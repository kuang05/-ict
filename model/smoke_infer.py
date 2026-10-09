# -*- coding: utf-8 -*-
"""冒烟测试：确认演示后端能加载"最优模型"而不是旧的 38 类模型。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import infer

img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
r = infer.predict(img)
eng = infer.get_engine()
print("engine =", r.get("engine"))
if r.get("engine") != "heuristic":
    print("n_classes =", len(eng.classes))
    print("前3类 =", eng.classes[:3])
print("top1 =", r["predictions"][0])
print("SMOKE_OK" if r.get("engine") != "heuristic" else "SMOKE_FAIL(退化为启发式)")
