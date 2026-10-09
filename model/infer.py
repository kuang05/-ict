# -*- coding: utf-8 -*-
"""
禾医·统一推理引擎
==================
四级自适应策略:
  1) model/checkpoints/best.pth      -> PyTorch ResNet18 迁移模型 (生产路径, 98%+可达)
  2) model/checkpoints_ms/best.ckpt  -> MindSpore ResNet50        (昇腾训练产物)
  3) model/checkpoints_light/*.json  -> 轻量质心模型(Demo基线, 开箱即用, 见train_light.py指标)
  4) 均不存在                        -> 内置裸规则引擎(仅保底, 不建议演示用)
统一输出:
  {"engine": "torch|mindspore|light|heuristic",
   "predictions": [{"class_id","class_name","confidence"}...top3],
   "features": {...色度学特征(报告展示用)...}}
"""
import json, os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys_dir = HERE
import sys
if sys_dir not in sys.path:
    sys.path.insert(0, sys_dir)

LABELS = json.load(open(os.path.join(HERE, "labels.json"), encoding="utf-8"))["classes"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

from features import extract as _extract_feats, pretty as _pretty_feats   # noqa: E402


def _top3_from_scores(score: dict) -> list:
    items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:3]
    raw = np.array([v for _, v in items], dtype=np.float64)
    raw = np.clip(raw, 0, None) + 1e-6
    probs = raw / raw.sum()
    return [{"class_id": LABELS.index(c), "class_name": c, "confidence": round(float(p), 4)}
            for (c, _), p in zip(items, probs)]


# ---------------------------------------------------------------- 裸规则引擎(最后兜底)
def _heuristic_predict(img: Image.Image) -> dict:
    _, x = _preprocess(img)
    d, _ = _extract_feats(img)
    score = {
        "Apple___healthy": 3 * d["h3"] + 2 * d["h4"] - 2 * d["dark"] - 2 * d["yellow"],
        "Apple___Apple_scab": 3 * d["dark"] + 1.5 * d["cluster"] - d["h4"],
        "Apple___Black_rot": 2.2 * d["dark"] + 2 * d["rust"] - d["h4"],
        "Corn_(maize)___Common_rust_": 4 * d["rust"] - 2 * d["h4"],
        "Tomato___Late_blight": 2.5 * d["dark"] + 2 * d["yellow"] - d["h4"] + 0.5 * d["rust"],
        "Potato___Early_blight": 2 * d["dark"] + 3 * d["yellow"] - d["h4"] + 0.3 * d["rust"],
    }
    return {"engine": "heuristic", "predictions": _top3_from_scores(score), "features": _pretty_feats(d)}


# ---------------------------------------------------------------- 轻量质心模型
class _LightEngine:
    def __init__(self, path):
        m = json.load(open(path, encoding="utf-8"))
        self.classes = m["classes"]
        self.keys = m["feat_keys"]
        self.mu = np.array(m["mu"]); self.sd = np.array(m["sd"])
        self.centroids = np.array(m["centroids"])
        self.metrics = m["metrics"]
        self.picked = m.get("picked", "centroid")
        if self.picked.startswith("knn"):
            self.k = 3 if self.picked == "knn3" else 1
            self.train_Z = np.array(m["train_Z"]); self.train_Y = np.array(m["train_Y"])

    def predict(self, img: Image.Image) -> dict:
        d, vec = _extract_feats(img)
        z = (vec - self.mu) / self.sd
        if self.picked == "centroid":
            dist2 = ((self.centroids - z[None]) ** 2).sum(-1)
            p = np.exp(-np.clip(dist2, 0, 50) / (2 * np.median(np.abs(dist2)) + 1e-9))
        else:
            d2 = ((self.train_Z - z[None]) ** 2).sum(-1)
            idx = np.argsort(d2)[:self.k]
            votes = np.zeros(len(self.classes))
            for i in idx:
                votes[self.train_Y[i]] += 1.0 / self.k
            p = votes * 0.98 + 0.02 / len(self.classes)      # 平滑, 保证top3均有值
        p = p / p.sum()
        top = p.argsort()[::-1][:3]
        return {"engine": "light",
                "predictions": [{"class_id": LABELS.index(self.classes[i]),
                                 "class_name": self.classes[i],
                                 "confidence": round(float(p[i]), 4)} for i in top],
                "features": _pretty_feats(d)}


# ---------------------------------------------------------------- 深度模型
class _TorchEngine:
    """PyTorch 推理引擎。

    支持两类产物（此前写死 resnet18 + strict 加载，导致只能跑旧的 38 类模型，
    无法加载最优模型 v11(R101) —— 2026-10-09 修复）：
      * v3 扁平模型：键为 `backbone.*`（可能带 MixStyle 三层包装）
      * v4 层次化双头：另含 `crop_head.*` / `head_dis.*`，此处只取病害头，其余丢弃
    类表一律从 ckpt 自带的 `classes` 读取，避免与静态 labels.json 不一致。
    """

    def __init__(self, ckpt_path):
        import torch
        from torchvision import models, transforms as T
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes = ck.get("classes") or LABELS
        bb = ck.get("backbone", "resnet18")
        n = len(self.classes)
        builder = {"resnet18": models.resnet18, "resnet50": models.resnet50,
                   "resnet101": models.resnet101}.get(bb)
        if builder is None:
            raise ValueError(f"演示引擎暂不支持 backbone={bb}")
        net = builder(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, n)
        # 归一化键：丢辅助头 → 剥 MixStyle 三层包装 → 剥 backbone. 前缀
        sd = {}
        for k, v in ck["model"].items():
            if k.startswith(("crop_head.", "head_dis.", "aux_head.")):
                continue
            for nm in ("layer1", "layer2", "layer3", "layer4"):
                pre = nm + ".0.0."
                if k.startswith(pre):
                    k = nm + ".0." + k[len(pre):]
                    break
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            sd[k] = v
        missing, unexpected = net.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "mix" not in k]
        if real_missing:
            raise RuntimeError(f"权重缺失 {len(real_missing)} 键，拒绝静默降级: {real_missing[:3]}")
        net.eval()
        self.net = net
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())])

    def predict(self, img: Image.Image) -> dict:
        import torch
        x = self.tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(self.net(x)[0], dim=0).numpy()
        top = p.argsort()[::-1][:3]
        d, _ = _extract_feats(img)
        return {"engine": "torch",
                "predictions": [{"class_id": int(i), "class_name": self.classes[i],
                                 "confidence": round(float(p[i]), 4)} for i in top],
                "features": _pretty_feats(d)}


class _MindSporeEngine:
    def __init__(self, ckpt_path):
        import mindspore as ms
        from mindvision.classification.models import resnet50
        meta = json.load(open(os.path.join(HERE, "checkpoints_ms", "meta.json"), encoding="utf-8"))
        self.classes = meta["classes"]
        net = resnet50(pretrained=False, num_classes=len(self.classes))
        param = ms.load_checkpoint(ckpt_path, net)
        ms.load_param_into_net(net, param)
        net.set_train(False)
        self.net, self.ms = net, ms
        self.mean = IMAGENET_MEAN.reshape(3, 1, 1) * 255
        self.std = IMAGENET_STD.reshape(3, 1, 1) * 255

    def predict(self, img: Image.Image) -> dict:
        img = img.convert("RGB").resize((224, 224))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None]
        x = (x - self.mean) / self.std
        logits = self.net(self.ms.Tensor(x))
        p = self.ms.ops.softmax(logits)[0].asnumpy()
        top = p.argsort()[::-1][:3]
        d, _ = _extract_feats(img)
        return {"engine": "mindspore",
                "predictions": [{"class_id": int(i), "class_name": self.classes[i],
                                 "confidence": round(float(p[i]), 4)} for i in top],
                "features": _pretty_feats(d)}


def _preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    return img, np.asarray(img, dtype=np.float32) / 255.0


def get_engine():
    """按**精度优先**构建引擎并缓存。

    2026-10-09 修复：原实现只认 `checkpoints/best.pth`（9月30日的旧 38 类模型），
    演示跑的不是最优模型。现改为候选列表，按精度从高到低尝试，并打印实际加载的引擎。
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    cands = [
        ("checkpoints/best_v11.pth", "torch"),      # 最优单模型 67.92%（R101, 27类）
        ("checkpoints/best.pth", "torch"),          # 通用最优
        ("checkpoints_ms/best.ckpt", "mindspore"),  # 昇腾 MindSpore 训练产物
    ]
    for rel, kind in cands:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        try:
            _ENGINE = _TorchEngine(p) if kind == "torch" else _MindSporeEngine(p)
            print(f"[HeYi] 引擎={kind} <- {rel} ({len(_ENGINE.classes)} 类)")
            break
        except Exception as e:
            print(f"[HeYi] {rel} 加载失败，尝试下一个: {e}")
    light_dir = os.path.join(HERE, "checkpoints_light")
    if _ENGINE is None and os.path.isdir(light_dir):
        lm = os.path.join(light_dir, "light_model.json")
        if os.path.exists(lm):
            _ENGINE = _LightEngine(lm)
            print("[HeYi] using light model. metrics:", _ENGINE.metrics)
    if _ENGINE is None:
        _ENGINE = "heuristic"
    return _ENGINE


def predict(img: Image.Image) -> dict:
    eng = get_engine()
    if eng == "heuristic":
        return _heuristic_predict(img)
    return eng.predict(img)


_ENGINE = None

if __name__ == "__main__":
    import sys as _s
    print(json.dumps(predict(Image.open(_s.argv[1])), ensure_ascii=False, indent=2))
