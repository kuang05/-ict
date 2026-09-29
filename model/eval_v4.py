# -*- coding: utf-8 -*-
"""
禾医 · 严格评测 v4
==================
写这个脚本的直接原因
--------------------
v3 评测出现 top1=4.2%、作物认对率 4.4% 的荒谬结果, 排查发现是 **权重静默加载失败**:
训练时 MixStyle 通过 `nn.Sequential(layer, MixStyle())` 插入, state_dict 的键变成
"layer1.0.xxx"; 而评测端构建模型的包装层数与之不一致, 导致 20 个键缺失、24 个键多余,
`load_state_dict(strict=False)` 不会报错, 模型退化为半随机初始化, 却照样"跑出指标"。

v4 的防线
---------
1. **strict=True 加载**: 键不匹配直接抛错, 绝不允许静默降级。
2. **键自动归一化**: 统一把 "layerN.0.xxx" 还原成 "layerN.xxx", 兼容是否插入 MixStyle。
3. **标签空间跟随 ckpt**: 类别顺序以 ckpt["classes"] 为准, 不再假设与本地常量一致。
4. **加载后自检**: 用一小批样本确认模型输出非退化(预测分布不塌缩到单一类)。

用法
----
/opt/venv/bin/python model/eval_v4.py --ckpt model/checkpoints/best_v2_r50.pth \
    --backbone resnet50 --data /root/data/pd_v2/test --tta
"""
import argparse, collections, json, os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def crop_of(pv_class):
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


def normalize_keys(sd):
    """把 MixStyle 包装引入的 "layerN.0.xxx" 还原为 "layerN.xxx" """
    out = {}
    for k, v in sd.items():
        for name in ("layer1", "layer2", "layer3", "layer4"):
            if k.startswith(name + ".0."):
                k = name + "." + k[len(name + ".0."):]
                break
        out[k] = v
    return out


def build_net(backbone, n_cls):
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
    return net


class CropSet(Dataset):
    def __init__(self, root, tf, c2i):
        self.items, self.tf = [], tf
        skipped = []
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            if cls not in c2i:
                skipped.append(cls)
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((os.path.join(d, f), c2i[cls]))
        if skipped:
            print(f"[eval-v4] 警告: {len(skipped)} 个目录不在模型输出空间, 已跳过: {skipped[:5]}")

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
                p = torch.softmax(net(x.to(device)).float(), dim=1).cpu().numpy()
                probs.append(p)
                ys.extend(y.tolist())
            p = np.concatenate(probs)
            prob_sum = p if prob_sum is None else prob_sum + p
            labels = np.array(ys)
    return prob_sum / (len(sizes) * len(flips)), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/root/data/pd_v2/test")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    net = build_net(args.backbone, len(classes))

    sd = normalize_keys(ck["model"])
    missing = [k for k in net.state_dict() if k not in sd]
    unexpected = [k for k in sd if k not in net.state_dict()]
    if missing or unexpected:
        raise SystemExit(
            f"[eval-v4] 权重加载失败, 已中止(避免静默降级): "
            f"missing={len(missing)}{missing[:4]} unexpected={len(unexpected)}{unexpected[:4]}")
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    print(f"[eval-v4] ckpt={args.ckpt} backbone={args.backbone} "
          f"classes={len(classes)} device={device} tta={bool(args.tta)} 权重完整加载 OK")

    sizes = [224, 288] if args.tta else [224]
    flips = [False, True] if args.tta else [False]
    probs, gts = run(net, args.data, c2i, device, sizes, flips, args.batch)

    # 自检: 预测若塌缩到单一类, 说明模型或权重仍有问题
    pred = probs.argmax(1)
    uniq, cnt = np.unique(pred, return_counts=True)
    if len(uniq) <= 2:
        print(f"[eval-v4] 警告: 预测仅覆盖 {len(uniq)} 个类 {dict(zip(uniq.tolist(), cnt.tolist()))}, "
              f"模型可能未正确加载")
    print(f"[eval-v4] 预测覆盖 {len(uniq)}/{len(classes)} 类, 样本 {len(gts)}")

    top3 = np.argsort(-probs, axis=1)[:, :3]
    n = len(gts)
    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t for t, g in zip(top3, gts)) / n)

    gt_crop = np.array([crop_of(classes[g]) for g in gts])
    pr_crop = np.array([crop_of(classes[p]) for p in pred])
    crop_ok = gt_crop == pr_crop
    crop_acc = float(crop_ok.mean())
    dis_given_crop = float((pred[crop_ok] == gts[crop_ok]).mean()) if crop_ok.any() else 0.0

    per = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    for p, g in zip(pred, gts):
        per[classes[g]][1] += 1
        per[classes[g]][0] += int(p == g)
        if p != g:
            conf[f"{classes[g]} -> {classes[p]}"] += 1

    res = {
        "ckpt": args.ckpt, "backbone": args.backbone, "n_classes": len(classes),
        "protocol": "PlantDoc 原图级留出 test(452 裁叶)",
        "tta": bool(args.tta), "n_test": n,
        "top1": round(top1, 4), "top3": round(top3acc, 4),
        "crop_acc": round(crop_acc, 4),
        "disease_acc_given_crop_correct": round(dis_given_crop, 4),
        "per_class": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
                      for k, v in sorted(per.items())},
        "top_confusions": [{"pair": k, "n": v} for k, v in conf.most_common(12)],
    }
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_class", "top_confusions")},
                     indent=2, ensure_ascii=False))
    print("\n--- 每类 ---")
    for k, v in sorted(res["per_class"].items(), key=lambda kv: kv[1]["acc"]):
        print(f"  {k:44s} {v['acc']*100:5.1f}%  n={v['n']}")
    print("\n--- 高频混淆 ---")
    for c in res["top_confusions"]:
        print(f"  {c['n']:3d}  {c['pair']}")

    out = args.out or args.ckpt.replace(".pth", "_eval_v4.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
