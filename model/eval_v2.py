# -*- coding: utf-8 -*-
"""
禾医 · 田间统一评测 v2
======================
与 v1 (eval_plantdoc.py) 的区别
-------------------------------
1. 直接读取 prepare_plantdoc.py 产出的 pd_v2/test(452 个裁叶, 原图级留出),
   不再每次从 parquet 重新解码, 保证与训练时的划分协议完全一致、可复现。
2. 自动兼容 **39 类(旧)** 与 **27 类(v2)** 两种输出空间:
   - 39 类模型默认把 logits 掩膜到 PlantDoc 对应的 27 类再 argmax,
     消除"判成无关作物"这类与病害识别能力无关的系统性失分;
   - 27 类模型直接评测。
3. **TTA**: 水平翻转 x 多尺度(224/256) 平均, 抵消单视角方差。
4. **作物级归因**: 同时报告"作物认对率"与"作物认对前提下病害认对率",
   判断误差主要来自物种混淆还是病害混淆, 指导下一步优化方向。

用法
----
/opt/venv/bin/python model/eval_v2.py --ckpt model/checkpoints/best_v2.pth
/opt/venv/bin/python model/eval_v2.py --ckpt model/checkpoints/best_finetuned.pth --no-tta
"""
import argparse, json, os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from prepare_plantdoc import PD2PV

CLASSES27 = sorted(set(PD2PV.values()))
C2I27 = {c: i for i, c in enumerate(CLASSES27)}


def crop_of(pv_class):
    """PlantVillage 类名 -> 作物名, 用于作物级归因"""
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


class CropDS(Dataset):
    """按目录名读取 27 类裁叶。目录名即 PlantVillage 类名。"""

    def __init__(self, root, tf):
        self.items = []
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d) or cls not in C2I27:
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((os.path.join(d, f), C2I27[cls]))
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def tf_val(size):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    n = len(classes)
    backbone = ck.get("backbone", "resnet18")
    builder = getattr(models, backbone, models.resnet18)
    net = builder(weights=None)
    if hasattr(net, "fc"):
        net.fc = nn.Linear(net.fc.in_features, n)
    elif hasattr(net, "classifier"):
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    net.eval().to(device)
    return net, classes, backbone, len(missing), len(unexpected)


@torch.no_grad()
def predict(net, ds, device, tta=True, mask27=None):
    """返回 (N, n_classes) 平均概率。mask27 非空时把 logits 限制在这些类。"""
    loader = DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True)
    sizes = [224, 256] if tta else [224]
    flips = [False, True] if tta else [False]
    acc = None
    for size in sizes:
        ds.tf = tf_val(size)
        for flip in flips:
            probs = []
            for x, _ in loader:
                x = x.to(device)
                if flip:
                    x = torch.flip(x, dims=[-1])
                logits = net(x)
                if mask27 is not None:
                    m = torch.full_like(logits, float("-inf"))
                    m[:, mask27] = logits[:, mask27]
                    logits = m
                probs.append(torch.softmax(logits.float(), dim=1).cpu())
            p = torch.cat(probs).numpy()
            acc = p if acc is None else acc + p
    return acc / (len(sizes) * len(flips))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/root/data/pd_v2/test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-mask", action="store_true",
                    help="39类模型不做27类掩膜(用于对比掩膜本身的收益)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, classes, backbone, n_missing, n_unexpected = load_model(args.ckpt, device)
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"[eval-v2] ckpt={args.ckpt}")
    print(f"[eval-v2] backbone={backbone} n_classes={len(classes)} "
          f"missing={n_missing} unexpected={n_unexpected} device={device} tta={not args.no_tta}")

    mask27 = None
    if len(classes) == 39 and not args.no_mask:
        mask27 = sorted(c2i[c] for c in CLASSES27 if c in c2i)
        print(f"[eval-v2] 39类模型 -> 掩膜到 {len(mask27)} 个 PlantDoc 有效类")

    ds = CropDS(args.data, tf_val(224))
    print(f"[eval-v2] 测试样本 {len(ds)}")

    gts = np.array([y for _, y in ds.items])
    probs = predict(net, ds, device, tta=not args.no_tta, mask27=mask27)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    # 掩膜后 top1 的类别索引可能属于 39 类空间, 统一映射回"PD 27 类"语义做作物归因
    def cls_of(i):
        return classes[i]

    n = len(gts)
    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t3 for t3, g in zip(top3, gts)) / n)

    gt_crop = np.array([crop_of(CLASSES27[g]) for g in gts])
    pred_crop = np.array([crop_of(cls_of(p)) for p in pred])
    crop_acc = float((gt_crop == pred_crop).mean())
    correct_crop = gt_crop == pred_crop
    disease_given_crop = float((pred[correct_crop] == gts[correct_crop]).mean()) \
        if correct_crop.any() else 0.0

    per_class = {}
    for i, c in enumerate(CLASSES27):
        sel = gts == i
        if sel.any():
            per_class[c] = {"acc": round(float((pred[sel] == i).mean()), 4), "n": int(sel.sum())}

    conf = Counter()
    for g, p in zip(gts, pred):
        if g != p:
            conf[f"{CLASSES27[g]} -> {cls_of(p)}"] += 1

    res = {
        "ckpt": args.ckpt,
        "n_classes": len(classes),
        "mask27": bool(mask27),
        "tta": not args.no_tta,
        "n_test": n,
        "top1": round(top1, 4),
        "top3": round(top3acc, 4),
        "crop_acc": round(crop_acc, 4),
        "disease_acc_given_crop_correct": round(disease_given_crop, 4),
        "per_class": per_class,
        "top_confusions": [{"pair": k, "n": v} for k, v in conf.most_common(12)],
    }
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_class", "top_confusions")},
                     indent=2, ensure_ascii=False))
    print("\n--- 每类 ---")
    for c, v in sorted(per_class.items(), key=lambda kv: kv[1]["acc"]):
        print(f"  {c:46s} {v['acc']*100:5.1f}%  n={v['n']}")
    print("\n--- 高频混淆 ---")
    for c in res["top_confusions"]:
        print(f"  {c['n']:3d}  {c['pair']}")

    out = args.out or args.ckpt.replace(".pth", "_eval.json").replace(
        "checkpoints", "evals") if False else (args.out or "/root/he-yi/docs/plantdoc_eval_v2.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
