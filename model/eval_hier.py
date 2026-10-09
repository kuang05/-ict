# -*- coding: utf-8 -*-
"""
禾医 · 层次融合推理与评测（配 train_hier.py 的 v4 层次化架构）
============================================================
核心：把双头输出按**层次先验**融合

    p_final(class) ∝ p_disease(class) · p_crop(crop_of(class))^α

α=0 退化为普通单头推理；α>0 时用作物先验抑制**跨作物误判**
（如把"番茄晚疫"错判成"马铃薯晚疫"——v3 的高频混淆之一）。

α 在 PlantDoc **val** 上选（最大化 val top1），在 **test** 上报告，避免过拟合测试集。

另外报告**高危守门**指标：
  * high_risk_recall@1 —— 高危样本被正确识别比例（比赛 KPI：漏诊率 <5%）
  * 置信度阈值 τ 拒识后，保留样本的高危漏诊率如何下降
    （拒识样本转人工复核 —— 正是作品"筛查+决策支持"定位的量化表达）

用法
----
python model/eval_hier.py --ckpt model/checkpoints/best_hier.pth \
  --data /home/ma-user/data/pd_v2/test --val /home/ma-user/data/pd_v2/val \
  --tta 1 --sizes 224,288 --out results/eval_hier.json
"""
import argparse, collections, json, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v3 as ev  # noqa: E402
from eval_v3 import CropSet, make_tf, CLASSES27, crop_of  # noqa: E402
from train_hier import NetV4, HIGH_RISK, crop_of as _crop_of  # noqa: E402


def load_net(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    crops = ck.get("crops", sorted(set(crop_of(c) for c in classes)))
    backbone = ck.get("backbone", "resnet101")
    if not (backbone.startswith("resnet") or backbone.startswith("timm_")):
        raise SystemExit(f"不支持的 backbone: {backbone}")
    net = NetV4(backbone, len(classes), len(crops), mix_p=0.0, pretrained=False)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    real_missing = [k for k in missing if "mix" not in k]
    if real_missing:
        raise SystemExit(f"权重缺失 {len(real_missing)}: {real_missing[:5]}")
    return net.eval().to(device), classes, crops


@torch.no_grad()
def infer(net, ds_root, device, sizes, flips, batch=64, workers=16):
    """返回 (p_disease, p_crop, labels)，跨 TTA 视图平均后的概率。"""
    pd_sum = pc_sum = None
    labels = None
    for size in sizes:
        for flip in flips:
            ds = CropSet(ds_root, make_tf(size, flip))
            loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
            d_l, c_l, ys = [], [], []
            for x, y in loader:
                x = x.to(device)
                d, c = net(x, both=True)
                d_l.append(torch.softmax(d.float(), 1).cpu().numpy())
                c_l.append(torch.softmax(c.float(), 1).cpu().numpy())
                ys.extend(y.tolist())
            d = np.concatenate(d_l)
            c = np.concatenate(c_l)
            pd_sum = d if pd_sum is None else pd_sum + d
            pc_sum = c if pc_sum is None else pc_sum + c
            labels = np.array(ys)
    n = len(sizes) * len(flips)
    return pd_sum / n, pc_sum / n, labels


def fuse(pd, pc, classes, crops, alpha):
    if alpha <= 0:
        return pd
    cls_crop = np.array([crops.index(_crop_of(c)) for c in classes])
    return pd * (pc[:, cls_crop] ** alpha)


def metrics(probs, labels, classes):
    pred = probs.argmax(1)
    n = len(labels)
    top1 = float((pred == labels).mean())
    top3 = float(sum(g in t for t, g in zip(np.argsort(-probs, axis=1)[:, :3], labels)) / n)
    crop_ok = np.array([crop_of(classes[p]) == crop_of(classes[g]) for p, g in zip(pred, labels)])
    crop_acc = float(crop_ok.mean())
    dis_given = float((pred[crop_ok] == labels[crop_ok]).mean()) if crop_ok.any() else 0.0
    return pred, top1, top3, crop_acc, dis_given, crop_ok


def hr_metrics(probs, pred, labels, classes):
    hr_idx = [i for i, c in enumerate(classes) if c in HIGH_RISK]
    hr_names = [classes[i] for i in hr_idx]
    mask = np.isin(labels, hr_idx)
    hr_n = int(mask.sum())
    if hr_n == 0:
        return {"high_risk_n": 0}
    hr_rec1 = float((pred[mask] == labels[mask]).mean())
    top3 = np.argsort(-probs, axis=1)[:, :3]
    hr_rec3 = float(np.mean([labels[i] in top3[i] for i in np.where(mask)[0]]))
    healthy_idx = [i for i, c in enumerate(classes) if c.endswith("healthy")]
    missed_healthy = float(np.mean([pred[i] in healthy_idx for i in np.where(mask)[0]]))
    return {"high_risk_n": hr_n, "high_risk_classes": hr_names,
            "high_risk_recall@1": round(hr_rec1, 4),
            "high_risk_recall@3": round(hr_rec3, 4),
            "high_risk_missed_as_healthy": round(missed_healthy, 4)}


def gate_curve(probs, labels, classes, thresholds=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """拒识守门曲线：置信度 < τ 转人工，看高危风险如何随拒识率下降。

    同时报告两个口径（农业场景里"漏诊"通常指后者）：
      * hr_miss_rate_after_gate      —— 高危样本被判成**任意**错类（误判率）
      * hr_healthy_miss_after_gate   —— 高危样本被判成**健康**（最危险，真正的"漏诊"）
    """
    hr_idx = [i for i, c in enumerate(classes) if c in HIGH_RISK]
    healthy_idx = [i for i, c in enumerate(classes) if c.endswith("healthy")]
    mask = np.isin(labels, hr_idx)
    conf = probs.max(1)
    pred = probs.argmax(1)
    out = []
    for tau in thresholds:
        kept = conf >= tau
        rej_rate = float(1 - kept.mean())
        hr_kept = mask & kept
        n_hr = int(hr_kept.sum())
        if n_hr > 0:
            miss = float((pred[hr_kept] != labels[hr_kept]).mean())
            hmiss = float(np.isin(pred[hr_kept], healthy_idx).mean())
        else:
            miss = hmiss = 0.0
        out.append({"tau": tau, "reject_rate": round(rej_rate, 4), "hr_n_kept": n_hr,
                    "hr_miss_rate_after_gate": round(miss, 4),
                    "hr_healthy_miss_after_gate": round(hmiss, 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/home/ma-user/data/pd_v3/test")
    ap.add_argument("--val", default="/home/ma-user/data/pd_v3/val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--sizes", default="224,288")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    device = "npu" if (getattr(torch, "npu", None) is not None and torch.npu.is_available()) \
        else ("cuda" if torch.cuda.is_available() else "cpu")

    net, classes, crops = load_net(args.ckpt, device)
    # 标签空间必须与 ckpt 的类表对齐（CropSet 用 eval_v3 的全局 C2I 给样本打标签）
    ev.C2I.clear()
    ev.C2I.update({c: i for i, c in enumerate(classes)})
    sizes = [int(s) for s in args.sizes.split(",")] if args.tta else [int(args.sizes.split(",")[0])]
    flips = [False, True] if args.tta else [False]
    print(f"[hiereval] ckpt={args.ckpt} classes={len(classes)} crops={len(crops)} "
          f"device={device} sizes={sizes} flips={flips}", flush=True)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_alpha, best_val = 0.0, -1.0
    val_table = []
    if args.val and os.path.isdir(args.val):
        pdv, pcv, yv = infer(net, args.val, device, sizes, flips, args.batch, args.workers)
        for a in alphas:
            p = fuse(pdv, pcv, classes, crops, a)
            t1 = float((p.argmax(1) == yv).mean())
            val_table.append({"alpha": a, "val_top1": round(t1, 4)})
            if t1 > best_val:
                best_val, best_alpha = t1, a
        print(f"[hiereval] val α 搜索: {val_table} → best α={best_alpha} (val_top1={best_val:.4f})", flush=True)
    else:
        print("[hiereval] 未提供 val，α 固定为 0", flush=True)

    pdt, pct, yt = infer(net, args.data, device, sizes, flips, args.batch, args.workers)
    pred0, t1_0, t3_0, ca_0, dg_0, _ = metrics(pdt, yt, classes)
    pt = fuse(pdt, pct, classes, crops, best_alpha)
    pred, top1, top3, crop_acc, dis_given, crop_ok = metrics(pt, yt, classes)

    crop_pred = pct.argmax(1)
    crop_gt = np.array([crops.index(_crop_of(classes[g])) for g in yt])
    crop_head_acc = float((crop_pred == crop_gt).mean())

    per = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    for p, g in zip(pred, yt):
        per[classes[g]][1] += 1
        per[classes[g]][0] += int(p == g)
        if p != g:
            conf[f"{classes[g]} -> {classes[p]}"] += 1

    res = {
        "ckpt": args.ckpt, "arch": "v4_hierarchical_multitask",
        "protocol": "PlantDoc 原图级留出 test(452 裁叶), 27类输出空间, 层次融合",
        "tta": bool(args.tta), "sizes": sizes, "alpha": best_alpha,
        "val_alpha_table": val_table,
        "n_test": len(yt),
        "top1": round(top1, 4), "top3": round(top3, 4),
        "top1_alpha0": round(t1_0, 4), "top3_alpha0": round(t3_0, 4),
        "crop_acc": round(crop_acc, 4), "crop_head_acc": round(crop_head_acc, 4),
        "disease_acc_given_crop_correct": round(dis_given, 4),
        "high_risk": hr_metrics(pt, pred, yt, classes),
        "gate_curve": gate_curve(pt, yt, classes),
        "per_class": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
                      for k, v in sorted(per.items())},
        "top_confusions": [{"pair": k, "n": v} for k, v in conf.most_common(12)],
    }
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_class", "top_confusions")},
                     indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
