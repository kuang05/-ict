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
    from torchvision import models
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    n = len(classes)
    if backbone == "resnet50":
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n)
    elif backbone == "resnet101":
        net = models.resnet101(weights=None)
        net.fc = nn.Linear(net.fc.in_features, n)
    else:
        raise ValueError(backbone)
    sd = ck["model"]
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    net.load_state_dict(sd, strict=True)   # 键必须全匹配: 集成要求同一输出空间
    return net.eval().to(device), classes, ck.get("tag", os.path.basename(path))


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
    ap.add_argument("--data", default="/home/ma-user/data/pd_v2/test")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", type=int, default=1)
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
        net, cls, tag = load_net(p, args.backbone, device)
        nets.append(net)
        tags.append(tag)
        if classes is None:
            classes = cls
        assert classes == cls, f"{p} 的类表与第一个 checkpoint 不一致"
    print(f"[ens] n_models={len(nets)} tags={tags} device={device} tta={bool(args.tta)}", flush=True)

    sizes = [224, 288] if args.tta else [224]
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

    res = {
        "ensemble": tags, "backbone": args.backbone, "n_classes": len(classes),
        "protocol": "PlantDoc 原图级留出 test(452 裁叶), 27类输出空间, 概率集成",
        "tta": bool(args.tta), "n_test": n,
        "top1": round(top1, 4), "top3": round(top3acc, 4),
        "crop_acc": round(crop_acc, 4),
        "disease_acc_given_crop_correct": round(dis_given, 4),
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
