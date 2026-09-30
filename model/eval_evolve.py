# -*- coding: utf-8 -*-
"""
禾医 · 可进化闭环：自动回归评测
================================
赛题3「**可进化**决策智能体」的核心不是"模型多准"，而是
**每一次策略变更都能被量化、被比较、可回滚**。
本脚本就是那个"度量尺"：

1. 在**固定评测集**（PlantDoc test, 452 个标注叶片）上跑当前策略
2. 输出常规指标 + **高危病害召回率**（比总体准确率更关键）
3. 把结果**追加**到 `results/evolution_history.jsonl`
4. 打印完整进化曲线，直观展示每一版的改进与退步

为什么单独统计"高危病害召回率"
------------------------------
番茄/马铃薯晚疫病 3~7 天可毁田，病毒病无特效药，黄龙病需砍树。
这类病害**漏诊的代价远高于误诊**。总体准确率会把它们淹没在
"健康叶片"这类简单样本里，而召回率直接回答评委最关心的问题：
"最危险的那几种病，你漏了多少？"

用法
----
/opt/venv/bin/python model/eval_evolve.py --ckpt model/checkpoints/best_v3b.pth \
    --backbone resnet50 --note "v3b: 40ep+800/类"
"""
import argparse, collections, json, os, time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from prepare_plantdoc import PD2PV
from train_field_v3 import Net, CLASSES27

HISTORY_PATH = "/root/he-yi/results/evolution_history.jsonl"

# 高危病害：漏诊代价极高（暴发性 / 无特效药 / 需整株处理）
HIGH_RISK = {
    "Tomato___Late_blight",              # 番茄晚疫病：3~7天毁田
    "Potato___Late_blight",              # 马铃薯晚疫病：引发爱尔兰大饥荒
    "Tomato___Tomato_mosaic_virus",      # 病毒病：无特效药
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",  # 黄化曲叶病毒：无特效药
    "Corn___Northern_Leaf_Blight",       # 玉米大斑病：流行性强
}


def crop_of(pv_class):
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


def build_net(backbone, n_cls, keys):
    """按 checkpoint 的实际结构重建（v3 为嵌套 backbone，v2 为标准模型）"""
    if any(k.startswith("backbone.") for k in keys):
        return Net(backbone, n_cls, mix_p=0.0)
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
def run(net, root, c2i, device, sizes, flips, batch=64):
    prob_sum, labels = None, None
    for size in sizes:
        for flip in flips:
            ds = CropSet(root, make_tf(size, flip), c2i)
            loader = DataLoader(ds, batch_size=batch, num_workers=8, pin_memory=True)
            probs, ys = [], []
            for x, y in loader:
                p = torch.softmax(net(x.to(device)).float(), 1).cpu().numpy()
                probs.append(p)
                ys.extend(y.tolist())
            p = np.concatenate(probs)
            prob_sum = p if prob_sum is None else prob_sum + p
            labels = np.array(ys)
    return prob_sum / (len(sizes) * len(flips)), labels


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    out = []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pd-root", default="/root/data/pd_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--note", default="", help="本版策略说明，写入进化史")
    ap.add_argument("--no-save", action="store_true", help="只评测不写入历史")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    c2i = {c: i for i, c in enumerate(classes)}
    net = build_net(args.backbone, len(classes), ck["model"].keys())
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"权重加载失败: missing={len(missing)} unexpected={len(unexpected)}")
    net.eval().to(device)

    sizes = [224, 288] if args.tta else [224]
    flips = [False, True] if args.tta else [False]
    probs, gts = run(net, os.path.join(args.pd_root, args.split), c2i,
                     device, sizes, flips)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    n = len(gts)

    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t for t, g in zip(top3, gts)) / n)
    gt_crop = np.array([crop_of(classes[g]) for g in gts])
    pr_crop = np.array([crop_of(classes[p]) for p in pred])
    crop_acc = float((gt_crop == pr_crop).mean())

    # ---- 高危病害召回率（核心指标）----
    hr_idx = [i for i, c in enumerate(classes) if c in HIGH_RISK]
    hr_mask = np.isin(gts, hr_idx)
    hr_n = int(hr_mask.sum())
    hr_recall1 = float((pred[hr_mask] == gts[hr_mask]).mean()) if hr_n else None
    hr_recall3 = float(sum(g in t for t, g in
                           zip(top3[hr_mask], gts[hr_mask])) / hr_n) if hr_n else None
    # 高危漏诊率 = 真实高危但 top1 判成"健康"的比例（最危险的失败模式）
    healthy_idx = [i for i, c in enumerate(classes) if c.endswith("___healthy")]
    hr_as_healthy = float(np.isin(pred[hr_mask], healthy_idx).mean()) if hr_n else None

    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": os.path.basename(args.ckpt),
        "note": args.note,
        "split": args.split, "n": n, "tta": bool(args.tta),
        "top1": round(top1, 4), "top3": round(top3acc, 4),
        "crop_acc": round(crop_acc, 4),
        "high_risk_n": hr_n,
        "high_risk_recall@1": None if hr_recall1 is None else round(hr_recall1, 4),
        "high_risk_recall@3": None if hr_recall3 is None else round(hr_recall3, 4),
        "high_risk_missed_as_healthy": None if hr_as_healthy is None else round(hr_as_healthy, 4),
    }

    print(json.dumps(rec, indent=2, ensure_ascii=False))

    if not args.no_save:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 进化曲线 ----
    hist = load_history()
    if len(hist) > 1:
        print("\n=== 进化曲线（按时间）===")
        print(f"{'版本':<28}{'top1':>8}{'top3':>8}{'作物':>8}{'高危R@1':>10}{'高危判健康':>12}")
        for h in hist:
            print(f"{h.get('note') or h.get('ckpt',''):<28}"
                  f"{h.get('top1',0):>8.3f}{h.get('top3',0):>8.3f}"
                  f"{h.get('crop_acc',0):>8.3f}"
                  f"{str(h.get('high_risk_recall@1')):>10}"
                  f"{str(h.get('high_risk_missed_as_healthy')):>12}")
        best = max(hist, key=lambda h: h.get("top1", 0))
        print(f"\n当前最优: {best.get('note') or best.get('ckpt')} "
              f"(top1={best.get('top1')})")


if __name__ == "__main__":
    main()
