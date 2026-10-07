# -*- coding: utf-8 -*-
"""高危指标计算器（口径与 eval_evolve.py 完全一致，复用 eval_v3.run 保证协议一致）。

指标：
  high_risk_recall@1        真实高危样本 top1 命中率
  high_risk_recall@3        top3 命中率
  high_risk_missed_as_healthy  真实高危但被判成任意 *_healthy 的比例（最危险失败）
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_v3 import run, CLASSES27  # noqa: E402
# 自包含高危清单（与 retrieval/risk_policy.py 保持一致）
HIGH_RISK_CLASSES = frozenset({
    "Corn___Northern_Leaf_Blight", "Potato___Late_blight", "Tomato___Late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
})


def load_net(path, device, backbone="resnet50"):
    from torchvision import models
    import torch.nn as nn
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    if backbone == "resnet101":
        net = models.resnet101(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(classes))
        sd = {k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
        net.load_state_dict(sd, strict=True)
        return net.eval().to(device), classes
    if backbone == "resnet50":
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(classes))
    elif backbone.startswith("timm_"):
        import timm
        n_target = n_cls if "n_cls" in dir() else len(classes)
        net = timm.create_model(backbone[5:], pretrained=False, num_classes=n_target)
    elif backbone == "resnet101":
        net = models.resnet101(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(classes))
    elif backbone == "convnext_tiny":
        net = models.convnext_tiny(weights=None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, len(classes))
    else:
        raise ValueError(backbone)
    sd = {k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    net.load_state_dict(sd, strict=True)
    return net.eval().to(device), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--data", default="/home/ma-user/data/pd_v2/test")
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

    net, classes = load_net(args.ckpt, device, args.backbone)
    sizes = [224, 288] if args.tta else [224]
    flips = [False, True] if args.tta else [False]
    probs, gts = run(net, args.data, device, sizes, flips, args.batch)
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    hr_names = sorted(HIGH_RISK_CLASSES)
    hr_idx = [i for i, c in enumerate(classes) if c in HIGH_RISK_CLASSES]
    hr_mask = np.isin(gts, hr_idx)
    hr_n = int(hr_mask.sum())
    recall1 = float((pred[hr_mask] == gts[hr_mask]).mean())
    recall3 = float(sum(g in t for t, g in zip(top3[hr_mask], gts[hr_mask])) / hr_n)
    healthy_idx = [i for i, c in enumerate(classes) if c.endswith("___healthy")]
    missed_as_healthy = float(np.isin(pred[hr_mask], healthy_idx).mean())
    # 扩展口径: 高危样本被判成任何非高危病害(误判方向统计)
    hr_wrong_nonhr = float((~np.isin(pred[hr_mask], hr_idx) & ~np.isin(pred[hr_mask], healthy_idx)).mean())

    rec = {
        "ckpt": os.path.basename(args.ckpt), "n_test": len(gts), "tta": bool(args.tta),
        "top1": round(float((pred == gts).mean()), 4),
        "high_risk_n": hr_n, "high_risk_classes": hr_names,
        "high_risk_recall@1": round(recall1, 4),
        "high_risk_recall@3": round(recall3, 4),
        "high_risk_missed_as_healthy": round(missed_as_healthy, 4),
        "high_risk_pred_nonhr_disease": round(hr_wrong_nonhr, 4),
    }
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
