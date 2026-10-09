# -*- coding: utf-8 -*-
"""
禾医 · 田间模型评测 v3 (适配 27 类输出空间 + TTA)
==================================================
与 v1(eval_plantdoc.py) 的关键差异
----------------------------------
1. v1 只能评 39 类模型; v3 评 train_field_v2.py 产出的 **27 类** 模型。
2. 评测数据用 prepare_plantdoc.py 产出的 **原图级留出** test(452 裁叶),
   与训练划分协议严格一致, 可复现。
3. 支持 **TTA**(多尺度 x 翻转) —— 零训练成本的稳定增益。
4. 权重加载兼容 MixStyle 包装(训练时 layer1 被包成 Sequential,
   state_dict 键形如 "layer1.0.xxx"), 这里自动还原为标准键。

用法
----
/opt/venv/bin/python model/eval_v3.py --ckpt model/checkpoints/best_v2_r50.pth --backbone resnet50 --tta
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
C2I = {c: i for i, c in enumerate(CLASSES27)}


def crop_of(pv_class):
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


def unwrap_mixstyle_keys(sd):
    """把 MixStyle 包装引入的**三层**键 "layerN.0.0.xxx" 还原为 "layerN.0.xxx"。

    ⚠️ 2026-10-08 修复：原版用**两层**前缀 "layerN.0." 剥离，会把**本来就标准**的键
    "layer1.0.conv1.weight" 误伤成 "layer1.conv1.weight"，导致 60 个键缺失
    （权重汤 soup_*.pth 这类纯 backbone ckpt 必然中招）。
    MixStyle 包装后的键一定是三层 "layerN.0.0."，两层的不该动。
    """
    out = {}
    for k, v in sd.items():
        for name in ("layer1", "layer2", "layer3", "layer4"):
            pre = name + ".0.0."
            if k.startswith(pre):
                k = name + ".0." + k[len(pre):]
                break
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        out[k] = v
    return out


def build_net(backbone, n_cls):
    if backbone.startswith("timm_"):
        import timm
        return timm.create_model(backbone[5:], pretrained=False, num_classes=n_cls)
    if backbone == "resnet101":
        net = models.resnet101(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n_cls)
        return net
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
    def __init__(self, root, tf):
        self.items = []
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d) or cls not in C2I:
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((os.path.join(d, f), C2I[cls]))
        self.tf = tf

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
def run(net, ds_root, device, sizes, flips, batch=64, workers=8):
    prob_sum, labels = None, None
    for size in sizes:
        for flip in flips:
            ds = CropSet(ds_root, make_tf(size, flip))
            loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
            probs, ys = [], []
            for x, y in loader:
                p = torch.softmax(net(x.to(device)).float(), dim=1).cpu().numpy()
                probs.append(p); ys.extend(y.tolist())
            p = np.concatenate(probs)
            prob_sum = p if prob_sum is None else prob_sum + p
            labels = np.array(ys)
    return prob_sum / (len(sizes) * len(flips)), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/home/ma-user/data/pd_v3/test")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--sizes", default="224,288", help="TTA尺度列表")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    device = "npu" if getattr(torch, "npu", None) is not None and torch.npu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    # 关键: GT 索引空间必须与 checkpoint 的类表对齐(30类扩展后 CLASSES27 不再是权威)
    C2I.clear()
    C2I.update({c: i for i, c in enumerate(classes)})
    n_cls = len(classes)
    net = build_net(args.backbone, n_cls)
    sd = unwrap_mixstyle_keys(ck["model"])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # 工程纪律(2026-10-08): 禁止静默降级。
    # 缺失键 = 权重没真正加载上, 模型退化却照样"跑出指标"(v2 时代踩过: top1 4.2%)。
    # 因此 missing 非空直接终止; unexpected 通常来自 crop_head 等辅助头, 可忽略。
    if missing:
        raise SystemExit(f"[eval-v3] 权重缺失 {len(missing)} 个键 (前5: {list(missing)[:5]}), "
                         f"评测结果不可信 → 终止。请检查 --backbone 与 ckpt 是否匹配。")
    if unexpected:
        print(f"[eval-v3] 提示: 多余键 {len(unexpected)} (前3: {list(unexpected)[:3]}) "
              f"—— 通常来自 crop_head 等辅助头, 已忽略")
    net.eval().to(device)
    print(f"[eval-v3] ckpt={args.ckpt} backbone={args.backbone} classes={n_cls} "
          f"device={device} tta={bool(args.tta)}")

    sizes = [int(s) for s in args.sizes.split(",")] if args.tta else [int(args.sizes.split(",")[0])]
    flips = [False, True] if args.tta else [False]
    probs, gts = run(net, args.data, device, sizes, flips, args.batch)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    n = len(gts)
    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t for t, g in zip(top3, gts)) / n)
    crop_ok = np.array([crop_of(classes[p]) == crop_of(classes[g])
                        for p, g in zip(pred, gts)])
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
        "protocol": "PlantDoc 原图级留出 test(452 裁叶), 27类输出空间",
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

    out = args.out or (args.ckpt.replace(".pth", "_eval.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
