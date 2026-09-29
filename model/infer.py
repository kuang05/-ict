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
    def __init__(self, ckpt_path):
        import torch
        from torchvision import models, transforms as T
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(self.classes))
        net.load_state_dict(ck["model"])
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
    """按优先级构建引擎并缓存。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    pth = os.path.join(HERE, "checkpoints", "best.pth")
    ck = os.path.join(HERE, "checkpoints_ms", "best.ckpt")
    light_dir = os.path.join(HERE, "checkpoints_light")
    for cand in [pth, ck]:
        try:
            if os.path.exists(pth) and cand == pth:
                _ENGINE = _TorchEngine(pth); break
            if os.path.exists(ck) and cand == ck:
                _ENGINE = _MindSporeEngine(ck); break
        except Exception as e:
            print("[HeYi] deep engine load failed:", e)
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
