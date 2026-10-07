# -*- coding: utf-8 -*-
"""禾医 · 田间伪标签脚手架（FixMatch 简化版第一阶段）

作用：对一批**无标注田间照片**用当前最佳模型打伪标签，产出训练用 items JSONL
（高置信子集）+ 人工复核清单（低置信/高危弃权子集）。
设计依据：docs/17 发现1（源域数据量⊥跨域精度）→ 提升方向是目标域数据；
本脚本是「采集 100~300 张真田间照片 → 自动伪标注 → 增量微调」管线的第一环。

用法:
  python model/pseudo_label.py --ckpt model/checkpoints/best_v3.pth \
      --data /path/to/unlabeled_photos --out results/pseudo_items.jsonl
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# 与 eval_v3 相同的推理变换（保证伪标签与评测协议一致）
TF = transforms.Compose([
    transforms.Resize(int(224 * 1.15)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")


class FolderDS(Dataset):
    def __init__(self, root):
        self.items = [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if f.lower().endswith(IMG_EXT)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], TF(Image.open(self.items[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/checkpoints/best_v3.pth")
    ap.add_argument("--data", required=True, help="无标注照片文件夹")
    ap.add_argument("--conf", type=float, default=0.85, help="伪标签置信门槛")
    ap.add_argument("--margin", type=float, default=0.20, help="top1-top2 最小间隔")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/pseudo_items.jsonl")
    args = ap.parse_args()

    device = "npu" if (getattr(torch, "npu", None) is not None and torch.npu.is_available()) else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    sd = {k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(sd, strict=True)
    net = net.eval().to(device)

    ds = FolderDS(args.data)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
    kept, review = [], []
    with torch.no_grad():
        for paths, x in loader:
            x = x.to(device)
            p = torch.softmax(net(x).float(), dim=1).cpu().numpy()
            top2 = np.sort(p, axis=1)[:, ::-1][:, :2]
            for path, probs, (t1, t2) in zip(paths, p, top2):
                idx = int(probs.argmax())
                rec = {"path": path, "class_id": classes[idx],
                       "confidence": round(float(t1), 4)}
                if t1 >= args.conf and (t1 - t2) >= args.margin:
                    rec["idx"] = idx
                    kept.append(rec)
                else:
                    rec["reason"] = ("low-conf" if t1 < args.conf else "ambiguous")
                    review.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    review_path = args.out.replace(".jsonl", "_review.jsonl")
    with open(review_path, "w", encoding="utf-8") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[pseudo] 总数={len(kept)+len(review)} 伪标签={len(kept)} "
          f"(conf>={args.conf}, margin>={args.margin}) 人工复核={len(review)} -> {review_path}")
    # 高危类弃权提示：进入复核清单的高危疑似照 must 人工优先
    hr = [r for r in review if "Late_blight" in r["class_id"] or "Mosaic" in r["class_id"]]
    print(f"[pseudo] 复核清单中疑似高危(晚疫/病毒类) {len(hr)} 张 —— 优先人工确认")


if __name__ == "__main__":
    main()
