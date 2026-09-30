# -*- coding: utf-8 -*-
"""
禾医 · v3 模型评测(结构感知版)
==============================
为什么还需要一个评测脚本
------------------------
不同版本训练脚本产出的 checkpoint 结构不同:
* v2 风格: 键为 "conv1.weight" / "layer1.0.conv1.weight" —— 标准 torchvision
* v3 风格: backbone 被包成子模块, 键为 "backbone.conv1.weight" ...
若评测端盲目用标准模型加载, 会全部 missing 而静默退化成随机权重。

本脚本**自动识别结构**: 扫描 checkpoint 的键前缀, 决定用
"标准模型" 还是 "Net(含 backbone 子模块)" 重建, 然后 **strict=True** 加载,
任何不匹配立即报错, 绝不静默降级。

同时输出与 v1 基线可比的完整指标: top1 / top3 / 作物级归因 / 每类 / 混淆。

用法
----
/opt/venv/bin/python model/eval_v5.py --ckpt model/checkpoints/best_v3.pth \
    --backbone resnet50 --split test --tta
"""
import argparse, collections, json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from prepare_plantdoc import PD2PV
from train_field_v3 import Net, CLASSES27


def crop_of(pv_class):
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


def detect_layout(keys):
    """根据键前缀判断 checkpoint 的网络结构"""
    if any(k.startswith("backbone.") for k in keys):
        return "nested"      # Net: backbone 是子模块
    return "flat"            # 标准 torchvision


def build_net(backbone, n_cls, layout):
    if layout == "nested":
        return Net(backbone, n_cls, mix_p=0.0)   # 评测时关闭 MixStyle
    net = getattr(models, backbone)(weights=None)
    if hasattr(net, "fc"):
        net.fc = nn.Linear(net.fc.in_features, n_cls)
    return net


class CropSet(Dataset):
    def __init__(self, root, tf, c2i):
        self.items, self.tf = [], tf
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d) or cls not in c2i:
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((os.path.join(d, f), c2i[cls]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def make_tf(size, flip=False):
    ops = [transforms.Resize(int(size * 1.15)), transforms.CenterCrop(size)]
    if flip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


@torch.no_grad()
def run(net, root, c2i, device, sizes, flips, batch=64, workers=8):
    prob_sum, labels = None, None
    for size in sizes:
        for flip in flips:
            ds = CropSet(root, make_tf(size, flip), c2i)
            loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
            probs, ys = [], []
            for x, y in loader:
                p = torch.softmax(net(x.to(device)).float(), 1).cpu().numpy()
                probs.append(p)
                ys.extend(y.tolist())
            p = np.concatenate(probs)
            prob_sum = p if prob_sum is None else prob_sum + p
            labels = np.array(ys)
    return prob_sum / (len(sizes) * len(flips)), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pd-root", default="/root/data/pd_v2")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    c2i = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)

    layout = detect_layout(ck["model"].keys())
    net = build_net(args.backbone, n_cls, layout)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"[eval-v5] 权重加载失败: layout={layout} "
                         f"missing={len(missing)}{missing[:3]} unexpected={len(unexpected)}{unexpected[:3]}")
    net.eval().to(device)
    print(f"[eval-v5] ckpt={args.ckpt} layout={layout} backbone={args.backbone} "
          f"classes={n_cls} split={args.split} tta={bool(args.tta)} device={device}")

    sizes = [224, 288] if args.tta else [224]
    flips = [False, True] if args.tta else [False]
    probs, gts = run(net, os.path.join(args.pd_root, args.split), c2i,
                     device, sizes, flips, args.batch)

    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    n = len(gts)
    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t for t, g in zip(top3, gts)) / n)

    gt_crop = np.array([crop_of(classes[g]) for g in gts])
    pr_crop = np.array([crop_of(classes[p]) for p in pred])
    ok = gt_crop == pr_crop
    crop_acc = float(ok.mean())
    dis_given = float((pred[ok] == gts[ok]).mean()) if ok.any() else 0.0

    per = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    for p, g in zip(pred, gts):
        per[classes[g]][1] += 1
        per[classes[g]][0] += int(p == g)
        if p != g:
            conf[f"{classes[g]} -> {classes[p]}"] += 1

    res = {
        "ckpt": args.ckpt, "backbone": args.backbone, "layout": layout,
        "n_classes": n_cls, "split": args.split, "tta": bool(args.tta),
        "protocol": "PlantDoc 原图级留出, 27类输出空间",
        "n_test": n, "top1": round(top1, 4), "top3": round(top3acc, 4),
        "crop_acc": round(crop_acc, 4),
        "disease_acc_given_crop_correct": round(dis_given, 4),
        "per_class": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
                      for k, v in sorted(per.items())},
        "top_confusions": [{"pair": k, "n": v} for k, v in conf.most_common(12)],
    }
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_class", "top_confusions")},
                     indent=2, ensure_ascii=False))
    print("\n--- 每类(升序) ---")
    for k, v in sorted(res["per_class"].items(), key=lambda kv: kv[1]["acc"]):
        print(f"  {k:46s} {v['acc']*100:5.1f}%  n={v['n']}")
    print("\n--- 高频混淆 ---")
    for c in res["top_confusions"][:10]:
        print(f"  {c['n']:3d}  {c['pair']}")

    out = args.out or f"/root/he-yi/docs/eval_{args.split}_{os.path.basename(args.ckpt).replace('.pth','')}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
