# -*- coding: utf-8 -*-
"""
禾医 · 27 类田间模型评测 (含 TTA)
==================================
与 eval_v2.py 的区别: eval_v2 评的是 39 类 PlantVillage 模型(需掩膜),
本脚本评的是 train_field_v2.py 产出的 **27 类输出空间**模型。

评测特性
--------
* TTA(测试时增强): 多尺度(224/288) x 水平翻转 x 中心/五切, 概率平均。
  单视角推理方差大, TTA 是零训练成本的稳定增益。
* 报告 top1 / top3 / 作物级准确率 / 每类准确率 / 高频混淆对。
* 支持在 val(模型选择) 与 test(最终报告) 两个 split 上评测。

用法
----
/opt/venv/bin/python model/eval_v3.py --ckpt model/checkpoints/best_v2.pth \
    --split test --tta
"""
import argparse, collections, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from prepare_plantdoc import PD2PV

CLASSES27 = sorted(set(PD2PV.values()))
CROP_OF = {c: c.split("___")[0].lower().replace("pepper,_bell", "pepper")
           for c in CLASSES27}


def build_net(backbone, n_cls, ckpt_path, device):
    if backbone == "resnet50":
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n_cls)
    elif backbone == "resnet18":
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n_cls)
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_cls)
    else:
        raise ValueError(backbone)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    return net.to(device).eval()


def make_batch_tf(size, five_crop=False):
    base = [
        transforms.Resize(int(size * 1.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if five_crop:
        return transforms.Compose([
            transforms.Resize(int(size * 1.4)),
            transforms.FiveCrop(size),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    transforms.ToTensor()(c)) for c in crops])),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        *base[1:],
    ])


@torch.no_grad()
def predict(net, loader, device, tta=True):
    """返回 (N, C) 概率与 (N,) 标签"""
    probs, gts = [], []
    for x, y in loader:
        gts.extend(y.tolist())
        x = x.to(device)
        if x.dim() == 5:  # five-crop: (B, 5, C, H, W)
            B, K = x.shape[0], x.shape[1]
            p = net(x.view(B * K, *x.shape[2:])).view(B, K, -1).mean(1)
        else:
            p = net(x)
        probs.append(torch.softmax(p.float(), 1).cpu().numpy())
        gts = gts[:sum(len(q) for q in probs)] if False else gts
    return np.concatenate(probs), np.array(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pd-root", default="/root/data/pd_v2")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = os.path.join(args.pd_root, args.split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f"split 目录不存在: {split_dir}")

    variants = [("center", 224, False)]
    if args.tta:
        variants += [("center", 288, False), ("five", 224, True)]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    n_cls = len(classes)
    net = build_net(args.backbone, n_cls, args.ckpt, device)
    print(f"[eval-v3] ckpt={args.ckpt} backbone={args.backbone} classes={n_cls} "
          f"split={args.split} tta={bool(args.tta)} device={device}")

    acc_prob, labels = None, None
    for name, size, five in variants:
        tf = make_batch_tf(size, five)
        ds = datasets.ImageFolder(split_dir, transform=tf)
        # ImageFolder 目录顺序即 classes 顺序, 需与训练一致
        if list(ds.classes) != list(classes):
            remap = [classes.index(c) for c in ds.classes]
            ds.samples = [(p, remap[y]) for p, y in ds.samples]
            ds.targets = [remap[y] for y in ds.targets]
        loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
        p, y = predict(net, loader, device)
        acc_prob = p if acc_prob is None else acc_prob + p
        labels = y
    probs = acc_prob / len(variants)

    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    n = len(labels)
    top1 = float((pred == labels).mean())
    top3acc = float(sum(l in t for t, l in zip(top3, labels)) / n)

    crop_ok = sum(CROP_OF.get(classes[p], "?") == CROP_OF.get(classes[l], "?")
                  for p, l in zip(pred, labels))
    per_class, conf = collections.defaultdict(lambda: [0, 0]), collections.Counter()
    for p, l in zip(pred, labels):
        per_class[classes[l]][1] += 1
        per_class[classes[l]][0] += int(p == l)
        if p != l:
            conf[f"{classes[l]} -> {classes[p]}"] += 1

    res = {
        "ckpt": args.ckpt, "backbone": args.backbone, "split": args.split,
        "n_samples": n, "tta": bool(args.tta), "protocol": "27类输出空间",
        "top1": round(top1, 4), "top3": round(top3acc, 4),
        "crop_acc": round(crop_ok / n, 4),
        "per_class": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
                      for k, v in sorted(per_class.items())},
        "top_confusions": [{"pair": k, "n": v} for k, v in conf.most_common(12)],
    }
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_class", "top_confusions")}, indent=2))
    print("\n--- 每类准确率 ---")
    for k, v in res["per_class"].items():
        print(f"  {k:44s} {v['acc']*100:5.1f}%  n={v['n']}")
    print("\n--- 高频混淆 ---")
    for c in res["top_confusions"]:
        print(f"  {c['n']:3d}  {c['pair']}")

    out = args.out or f"/root/he-yi/docs/eval_{args.split}_{os.path.basename(args.ckpt).replace('.pth','')}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
