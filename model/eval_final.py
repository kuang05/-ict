# -*- coding: utf-8 -*-
"""
禾医 · 田间模型统一评测(最终版)
================================
设计目的
--------
训练时 MixStyle 以 nn.Sequential(layer, MixStyle()) 形式插在网络里, 导致
state_dict 的键变成 "layerN.0.xxx"; 而评测端若用标准 torchvision 模型加载,
键名不匹配 -> 权重静默丢失 -> 指标退化到随机水平(表现为 top1 个位数)。
本脚本在加载前统一做**键规范化**, 兼容"包装过 / 未包装"两种 checkpoint,
并在加载后打印缺失键数量, 杜绝静默失败。

同时支持:
* 39 类(旧 v1 输出空间) 与 27 类(v2 输出空间) 模型
* 39 类模型可选掩膜到 PlantDoc 27 个有效类
* TTA: 多尺度 x 水平翻转
* 作物级归因: 区分"认错作物"与"认错病害"

用法
----
/opt/venv/bin/python model/eval_final.py --ckpt model/checkpoints/best_v2_r50.pth \
    --backbone resnet50
"""
import argparse, collections, json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from prepare_plantdoc import PD2PV

CLASSES27 = sorted(set(PD2PV.values()))
C2I27 = {c: i for i, c in enumerate(CLASSES27)}
VALID27 = set(PD2PV.values())


def crop_of(pv_class):
    """PlantVillage 类名 -> 作物名"""
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


def normalize_state_dict(sd):
    """把 MixStyle 包装引入的 "layerN.0.xxx" 还原为 "layerN.xxx"。

    只处理以 layerN.0. 开头的键; 已是标准键的原样保留, 因此两种 checkpoint 都兼容。
    """
    out = {}
    for k, v in sd.items():
        for n in ("layer1", "layer2", "layer3", "layer4"):
            pre = n + ".0."
            if k.startswith(pre):
                k = n + "." + k[len(pre):]
                break
        out[k] = v
    return out


def build_net(backbone, n_cls):
    """构建**无 MixStyle 包装**的标准网络, 与规范化后的 checkpoint 对齐"""
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
        raise ValueError(f"unknown backbone: {backbone}")
    return net


class CropSet(Dataset):
    """按目录名读取裁叶, 标签用**模型输出空间**的下标(避免 39/27 类索引错位)"""

    def __init__(self, root, tf, c2i):
        self.items, skipped = [], []
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
        self.tf = tf
        if skipped:
            print(f"[eval] 跳过不在模型输出空间的目录: {skipped}")

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
def run(net, root, c2i, device, sizes, flips, mask27, batch=64, workers=8):
    prob_sum, labels = None, None
    for size in sizes:
        for flip in flips:
            ds = CropSet(root, make_tf(size, flip), c2i)
            loader = DataLoader(ds, batch_size=batch, num_workers=workers,
                                pin_memory=True)
            probs, ys = [], []
            for x, y in loader:
                x = x.to(device)
                logits = net(x)
                if mask27 is not None:
                    m = torch.full_like(logits, float("-inf"))
                    m[:, mask27] = logits[:, mask27]
                    logits = m
                probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
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
    ap.add_argument("--no-mask", action="store_true",
                    help="39类模型不做27类掩膜(用于对比掩膜收益)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    n_cls = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}

    net = build_net(args.backbone, n_cls)
    sd = normalize_state_dict(ck["model"])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval().to(device)

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] backbone={args.backbone} n_classes={n_cls} device={device} tta={bool(args.tta)}")
    print(f"[eval] 缺失键={len(missing)} 多余键={len(unexpected)}")
    if len(missing) > 0:
        print(f"[eval][警告] 有 {len(missing)} 个键未加载, 指标可能失真: {missing[:3]}")

    # 39 类模型 -> 掩膜到 PlantDoc 27 个有效类
    mask27 = None
    if n_cls == 39 and not args.no_mask:
        mask27 = sorted(c2i[c] for c in VALID27 if c in c2i)
        print(f"[eval] 39类模型 -> 掩膜到 {len(mask27)} 个 PlantDoc 有效类")

    sizes = [224, 288] if args.tta else [224]
    flips = [False, True] if args.tta else [False]
    probs, gts = run(net, args.data, c2i, device, sizes, flips, mask27,
                     args.batch)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    n = len(gts)
    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t for t, g in zip(top3, gts)) / n)

    gt_crop = np.array([crop_of(classes[g]) for g in gts])
    pred_crop = np.array([crop_of(classes[p]) for p in pred])
    crop_ok = gt_crop == pred_crop
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
        "ckpt": args.ckpt, "backbone": args.backbone, "n_classes": n_cls,
        "protocol": "PlantDoc 原图级留出 test(452 裁叶)",
        "mask27": bool(mask27), "tta": bool(args.tta), "n_test": n,
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
    print("\n--- 每类(升序) ---")
    for k, v in sorted(res["per_class"].items(), key=lambda kv: kv[1]["acc"]):
        print(f"  {k:46s} {v['acc']*100:5.1f}%  n={v['n']}")
    print("\n--- 高频混淆 ---")
    for c in res["top_confusions"][:10]:
        print(f"  {c['n']:3d}  {c['pair']}")

    out = args.out or args.ckpt.replace(".pth", "_eval_final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
