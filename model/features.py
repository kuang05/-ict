# -*- coding: utf-8 -*-
"""
禾医·叶面特征提取（背景掩膜 + 色度学统计）
============================================
针对PlantVillage纯色摄影棚背景的针对性设计:
  1) 背景掩膜: 近白(S<0.15, V>0.82)像素剔除, 所有比例分母均为"叶面像素数"
  2) 特征组:
     - 病变占比: dark(坏死黑斑)/yellow(黄化)/rust(锈色孢子堆)/mildew(叶面白粉)
     - 色相直方图: 叶面像素10-bin归一化直方图
     - 空间聚集: dark密度的3x3网格标准差
输出固定长度向量(用于轻量模型) + 字典(用于报告展示)。
"""
import numpy as np
from PIL import Image

FEAT_KEYS = ["dark", "yellow", "rust", "mildew", "leaf_ratio",
             "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "cluster"]

def extract(img: Image.Image) -> tuple[dict, np.ndarray]:
    img224 = img.convert("RGB").resize((224, 224))
    hsv = np.asarray(img224.convert("HSV"), dtype=np.float32)
    H, S, V = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255

    bg = (S < 0.15) & (V > 0.82)                    # 摄影棚白底
    leaf = ~bg
    area = max(leaf.sum(), 1)

    dark = (S < 0.40) & (V < 0.32) & leaf
    yellow = (H >= 35) & (H < 60) & (S > 0.25) & (V > 0.35) & leaf
    rust = (H >= 5) & (H < 40) & (S > 0.40) & (V > 0.20) & (V <= 0.75) & leaf
    mildew = (S < 0.15) & (V > 0.70) & leaf

    hleaf = H[leaf & (S > 0.08)]
    hist, _ = np.histogram(hleaf, bins=10, range=(0, 360)) if hleaf.size else (
        np.zeros(10), None)
    hist = hist / max(hist.sum(), 1)

    # 黑腐病"蛙眼"斑偏红棕且带亮心, 黑星病偏橄榄绿褐: 用红棕占比+叶面明度方差区分
    brown_red = (((H >= 340) | (H < 20)) & (S > 0.30) & (V > 0.15) & (V < 0.60) & leaf)
    v_std = float(V[leaf].std()) if leaf.sum() > 10 else 0.0

    rows = np.array_split(dark.astype(np.float64), 3, axis=0)
    cells = [np.array_split(r, 3, axis=1) for r in rows]
    cluster = float(np.std([c.mean() for row in cells for c in row]))

    d = {"dark": dark.sum() / area, "yellow": yellow.sum() / area,
         "rust": rust.sum() / area, "mildew": mildew.sum() / area,
         "leaf_ratio": leaf.mean(), "brown_red": brown_red.sum() / area,
         "v_std": v_std, "cluster": cluster,
         **{f"h{i}": float(hist[i]) for i in range(10)}}
    d = {k: round(float(v), 5) for k, v in d.items()}
    vec = np.array([d[k] for k in FEAT_KEYS], dtype=np.float64)
    return d, vec

def pretty(d: dict) -> dict:
    """报告用: 百分比已乘好, 键名面向人读。"""
    return {"leaf_pct": round(d["leaf_ratio"] * 100, 1),
            "dark_pct": round(d["dark"] * 100, 1),
            "yellow_pct": round(d["yellow"] * 100, 1),
            "rust_pct": round(d["rust"] * 100, 1),
            "mildew_pct": round(d["mildew"] * 100, 1),
            "cluster": d["cluster"]}
