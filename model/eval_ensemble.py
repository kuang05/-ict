# -*- coding: utf-8 -*-
"""多checkpoint概率集成评测(与 eval_v3 同协议: PlantDoc test 452 裁叶, TTA 多尺度x翻转)。

用法:
  python model/eval_ensemble.py --ckpt ckpt1.pth ckpt2.pth ... \
      --data /path/to/pd_v2/test --tta 1 --out results/eval_ensemble.json

原理: 每个 checkpoint 独立加载为裸 ResNet50(27类, 键剥离 backbone. 前缀),
对每个 TTA 视图(224/288 x 翻转)取各模型 softmax 平均, 再跨视图平均。
"""
import argparse, collections, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval_v3 import CropSet, make_tf, CLASSES27, crop_of


def load_net(path, backbone="resnet50", device="npu"):
    """加载一个 ckpt 为可前向的模型（返回病害 logits）。

    支持两类 ckpt：
      * 层次化 NetV4（arch=v4_hierarchical_multitask）：含 crop_head/head_dis，用 NetV4 重建；
      * 单头 Net（v3 系）：backbone.fc 即病害头。
    （2026-10-08 修复：原版对 timm 骨干用 num_classes=n 建模型，会多出 head.fc 键，
      与 NetV4 的 num_classes=0 结构不匹配 → 集成直接 RuntimeError。）
    """
    from torchvision import models
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    n = len(classes)
    tag = ck.get("tag", os.path.basename(path))
    is_hier = str(ck.get("arch", "")).startswith("v4_hierarchical")

    if is_hier:
        from train_hier import NetV4
        crops = ck.get("crops") or sorted(set(crop_of(c) for c in classes))
        net = NetV4(backbone, n, len(crops), mix_p=0.0, pretrained=False)
        missing, unexpected = net.load_state_dict(ck["model"], strict=False)
        real_missing = [k for k in missing if "mix" not in k]
        if real_missing:
            raise RuntimeError(f"{path} 权重缺失 {real_missing[:3]}")
        return net.eval().to(device), classes, tag

    if backbone == "resnet50":
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n)
    elif backbone == "resnet101":
        net = models.resnet101(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n)
    elif backbone == "resnet18":
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n)
    elif backbone.startswith("timm_"):
        # 扁平 timm 模型（train_field_v3.py 产出，arch 字段缺失，如 best_v9 的 convnext_tiny）
        import timm
        net = timm.create_model(backbone[5:], pretrained=False, num_classes=n)
    else:
        raise ValueError(backbone)
    sd = {k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    net.load_state_dict(sd, strict=True)
    return net.eval().to(device), classes, tag


@torch.no_grad()
def run_ensemble(nets, ds_root, device, sizes, flips, batch=64, workers=16):
    prob_sum, labels = None, None
    for size in sizes:
        for flip in flips:
            ds = CropSet(ds_root, make_tf(size, flip))
            loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
            probs, ys = [], []
            for x, y in loader:
                x = x.to(device)
                p = None
                for net in nets:
                    q = torch.softmax(net(x).float(), dim=1)
                    p = q if p is None else p + q
                probs.append((p / len(nets)).cpu().numpy())
                ys.extend(y.tolist())
            p = np.concatenate(probs)
            prob_sum = p if prob_sum is None else prob_sum + p
            labels = np.array(ys)
    return prob_sum / (len(sizes) * len(flips)), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
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
    device = "npu" if (getattr(torch, "npu", None) is not None and torch.npu.is_available()) else \
        ("cuda" if torch.cuda.is_available() else "cpu")

    nets, tags, classes = [], [], None
    for p in args.ckpt:
        _ck = torch.load(p, map_location="cpu", weights_only=False)
        bb = _ck.get("backbone", args.backbone)   # 异构集成: 优先用各 ckpt 自带的骨干
        net, cls, tag = load_net(p, bb, device)
        nets.append(net)
        tags.append(tag)
        if classes is None:
            classes = cls
        assert classes == cls, f"{p} 的类表与第一个 checkpoint 不一致"
    # 标签空间必须与 ckpt 的类表对齐（CropSet 用 eval_v3 的全局 C2I 给样本打标签；
    # 不同步的话，一旦 ckpt 类序不是默认 CLASSES27 就会静默错标）
    import eval_v3 as ev
    ev.C2I.clear()
    ev.C2I.update({c: i for i, c in enumerate(classes)})
    print(f"[ens] n_models={len(nets)} tags={tags} device={device} tta={bool(args.tta)}", flush=True)

    sizes = [int(s) for s in args.sizes.split(",")] if args.tta else [int(args.sizes.split(",")[0])]
    flips = [False, True] if args.tta else [False]
    probs, gts = run_ensemble(nets, args.data, device, sizes, flips, args.batch)
    pred = probs.argmax(1)
    n = len(gts)
    top1 = float((pred == gts).mean())
    top3acc = float(sum(g in t for t, g in zip(np.argsort(-probs, axis=1)[:, :3], gts)) / n)
    crop_ok = np.array([crop_of(classes[p]) == crop_of(classes[g]) for p, g in zip(pred, gts)])
    crop_acc = float(crop_ok.mean())
    dis_given = float((pred[crop_ok] == gts[crop_ok]).mean()) if crop_ok.any() else 0.0

    per = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    for p, g in zip(pred, gts):
        per[classes[g]][1] += 1
        per[classes[g]][0] += int(p == g)
        if p != g:
            conf[f"{classes[g]} -> {classes[p]}"] += 1

    # ---- 高危守门指标（对齐评奖"社会效益"维度：漏诊率随拒识阈值下降） ----
    hr_extra = {}
    try:
        from eval_hier import hr_metrics, gate_curve
        hr_extra["high_risk"] = hr_metrics(probs, pred, gts, classes)
        hr_extra["gate_curve"] = gate_curve(probs, gts, classes)
    except Exception as e:
        hr_extra["high_risk_note"] = f"跳过: {str(e)[:80]}"

    res = {
        "ensemble": tags, "backbone": args.backbone, "n_classes": len(classes),
        "protocol": "PlantDoc 原图级留出 test(452 裁叶), 27类输出空间, 概率集成",
        "tta": bool(args.tta), "n_test": n,
        "top1": round(top1, 4), "top3": round(top3acc, 4),
        "crop_acc": round(crop_acc, 4),
        "disease_acc_given_crop_correct": round(dis_given, 4),
        **hr_extra,
        "per_class": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
                      for k, v in sorted(per.items())},
        "top_confusions": [{"pair": k, "n": v} for k, v in conf.most_common(12)],
    }
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_class", "top_confusions")},
                     indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
