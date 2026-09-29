# -*- coding: utf-8 -*-
"""
禾医 · 两级推理评测 (作物 -> 病害)
==================================
动机(来自 v2 诊断)
------------------
旧模型在 PlantDoc test 上的错误归因:
  * 作物认对率       74.12%
  * 作物认对前提下病害准确率 76.42%
即约 1/4 的失分来自"把番茄叶判成马铃薯叶"这类**跨作物混淆**,
而这些类别在病害特征上本不相似 —— 是输出空间缺少结构约束导致的。

方法
----
把 27 个细粒度类按作物分组(apple/tomato/grape/... 共 13 组), 推理分两步:
  1. 作物级: crop_score[c] = logsumexp(logits of classes in crop c)
     用 logsumexp 而非 max, 表示该作物下所有病害证据的累积, 对单类噪声更稳健。
  2. 病害级: 在 argmax 作物的类子集内再取 argmax。
这样模型永远不可能输出"番茄叶的病却判成马铃薯"的越界结果。

本脚本不训练任何参数, 可与同权重的"扁平 argmax"直接对比,
因此是纯净的推理端增益(也可作为消融: --mode flat / two_stage)。

用法
----
/opt/venv/bin/python model/eval_two_stage.py \
    --ckpt model/checkpoints/best_finetuned.pth --backbone resnet18 --tta
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


def crop_of(pv_class):
    c = pv_class.split("___")[0].lower()
    return {"pepper,_bell": "pepper"}.get(c, c)


class CropSet(Dataset):
    def __init__(self, root, tf):
        self.items = []
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d) or cls not in C2I27:
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((os.path.join(d, f), C2I27[cls]))
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def make_tf(size, flip):
    ops = [transforms.Resize(int(size * 1.15)), transforms.CenterCrop(size)]
    if flip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


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


def unwrap_mixstyle_keys(sd):
    out = {}
    for k, v in sd.items():
        for name in ("layer1", "layer2", "layer3"):
            if k.startswith(name + ".0."):
                k = name + "." + k[len(name + ".0."):]
                break
        out[k] = v
    return out


@torch.no_grad()
def collect_logits(net, root, device, tta, batch=64, workers=8):
    """返回 (N, n_classes) 的 TTA 平均 logits 与 (N,) 27类标签"""
    sizes = [224, 288] if tta else [224]
    flips = [False, True] if tta else [False]
    acc, labels = None, None
    for size in sizes:
        for flip in flips:
            ds = CropSet(root, make_tf(size, flip))
            loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
            outs, ys = [], []
            for x, y in loader:
                outs.append(net(x.to(device)).float().cpu().numpy())
                ys.extend(y.tolist())
            l = np.concatenate(outs)
            acc = l if acc is None else acc + l
            labels = np.array(ys)
    return acc / (len(sizes) * len(flips)), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/root/data/pd_v2/test")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    n_cls = len(classes)
    net = build_net(args.backbone, n_cls)
    missing, unexpected = net.load_state_dict(unwrap_mixstyle_keys(ck["model"]), strict=False)
    net.eval().to(device)
    print(f"[two-stage] ckpt={args.ckpt} backbone={args.backbone} classes={n_cls} "
          f"tta={bool(args.tta)} device={device}")

    logits, gts = collect_logits(net, args.data, device, bool(args.tta))
    n_cls_model = logits.shape[1]

    # 定位 27 个有效类在模型输出空间中的下标
    if n_cls_model == len(CLASSES27):
        valid = list(range(n_cls_model))
        name_of = {i: CLASSES27[i] for i in valid}
    else:
        c2i = {c: i for i, c in enumerate(classes)}
        valid = sorted(c2i[c] for c in CLASSES27 if c in c2i)
        name_of = {i: classes[i] for i in valid}
    print(f"[two-stage] 有效类 {len(valid)} / 模型输出 {n_cls_model}")

    # 按作物分组
    groups = collections.defaultdict(list)
    for i in valid:
        groups[crop_of(name_of[i])].append(i)
    crop_names = sorted(groups)
    print(f"[two-stage] 作物 {len(crop_names)} 组: {crop_names}")

    # --- 模式1: 扁平 argmax(掩膜到 27 类) ---
    masked = np.full_like(logits, -1e9)
    masked[:, valid] = logits[:, valid]
    flat_pred = masked.argmax(1)

    # --- 模式2: 两级 (作物 -> 病害) ---
    # 作物分: 组内 logsumexp
    cl = np.full((logits.shape[0], len(crop_names)), -1e9)
    for ci, cn in enumerate(crop_names):
        idx = groups[cn]
        m = logits[:, idx].max(axis=1, keepdims=True)
        cl[:, ci] = (m.squeeze(1) + np.log(np.exp(logits[:, idx] - m).sum(axis=1)))
    best_crop = cl.argmax(1)
    two_pred = np.empty(logits.shape[0], dtype=int)
    for r in range(logits.shape[0]):
        idx = groups[crop_names[best_crop[r]]]
        two_pred[r] = idx[int(np.argmax(logits[r, idx]))]

    # 真实标签的作物(27 类索引 -> 作物)
    gt_crop_name = np.array([crop_of(CLASSES27[g]) for g in gts])

    def summarize(pred, tag):
        n = len(gts)
        top1 = float((pred == gts).mean())
        prob = np.exp(masked - masked.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        top3 = np.argsort(-masked, axis=1)[:, :3]
        top3acc = float(sum(g in t for t, g in zip(top3, gts)) / n)
        pc = np.array([crop_of(name_of[p]) for p in pred])
        crop_acc = float((pc == gt_crop_name).mean())
        ok = pc == gt_crop_name
        dis = float((pred[ok] == gts[ok]).mean()) if ok.any() else 0.0
        return {"mode": tag, "top1": round(top1, 4), "top3": round(top3acc, 4),
                "crop_acc": round(crop_acc, 4),
                "disease_acc_given_crop_correct": round(dis, 4)}

    res = {"ckpt": args.ckpt, "backbone": args.backbone, "tta": bool(args.tta),
           "n_test": len(gts), "protocol": "PlantDoc 原图级留出 test(452 裁叶)",
           "results": [summarize(flat_pred, "flat_argmax_27"),
                       summarize(two_pred, "two_stage_crop_then_disease")]}

    # 两级模式下的作物预测准确率(上界参考)
    bc_name = np.array([crop_names[b] for b in best_crop])
    res["crop_selection_acc"] = round(float((bc_name == gt_crop_name).mean()), 4)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    delta = res["results"][1]["top1"] - res["results"][0]["top1"]
    print(f"\n>>> 两级推理相对扁平 argmax 的 top1 变化: {delta:+.4f}")

    out = args.out or args.ckpt.replace(".pth", "_twostage.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
