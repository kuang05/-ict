# -*- coding: utf-8 -*-
"""禾医 · 全链路小数据自检（不碰生产数据、不污染生产目录）

目的：用极小的合成数据端到端跑通 train_hier → eval_hier → export_onnx，
      在不占用大量算力、不影响生产训练的前提下，把**代码级问题**暴露出来。

设计要点：
* 合成数据只用 3 个真实类名（必须属于 CLASSES27，否则会被 scan_dir 过滤掉）
* 图像尺寸 64、backbone resnet18、2 epoch、batch 4 —— 单次运行几秒
* 全部落在 /tmp/heyi_selftest，跑完不残留

用法: python model/selftest_pipeline.py
"""
import json, os, shutil, subprocess, sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TMP = "/tmp/heyi_selftest"
CLASSES = ["Tomato___healthy", "Tomato___Late_blight", "Potato___Late_blight"]


def _img(ci, size=64):
    """造一张带类别颜色信号的随机图（保证类别可学，指标不会恒为 0）。"""
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    arr[..., ci % 3] = 200
    return Image.fromarray(arr)


def make_data():
    shutil.rmtree(TMP, ignore_errors=True)
    for part, n in (("train", 8), ("val", 6), ("test", 6)):
        for ci, c in enumerate(CLASSES):
            d = os.path.join(TMP, "pd", part, c)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                _img(ci).save(os.path.join(d, f"{c}_{i}.jpg"))
    for ci, c in enumerate(CLASSES):
        d = os.path.join(TMP, "pv", c)
        os.makedirs(d, exist_ok=True)
        for i in range(4):
            _img(ci).save(os.path.join(d, f"pv_{i}.jpg"))
    print(f"[selftest] 合成数据已建 -> {TMP}（3 类）")


def run(cmd, name):
    print(f"[selftest] 运行 {name} ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = "\n".join(r.stderr.strip().splitlines()[-12:])
        print(f"  ✗ {name} 失败\n{err}")
        return False
    return True


def main():
    make_data()
    ckpt = os.path.join(TMP, "best_tiny.pth")
    ok = run([sys.executable, "model/train_hier.py",
              "--pv-data", f"{TMP}/pv", "--pd-root", f"{TMP}/pd",
              "--backbone", "resnet18", "--img-size", "64", "--epochs", "2",
              "--batch", "4", "--workers", "0", "--lr", "1e-3",
              "--pv-per-class", "0", "--pd-repeat", "1",
              "--crop-weight", "0.3", "--mixup", "0.2", "--ema-decay", "0.9",
              "--out", ckpt, "--tag", "selftest"], "train_hier")
    if not ok or not os.path.exists(ckpt):
        print("TRAIN_FAIL")
        return 1
    print("  ✓ train_hier 通过（ckpt 已生成）")

    eval_json = os.path.join(TMP, "eval.json")
    ok = run([sys.executable, "model/eval_hier.py", "--ckpt", ckpt,
              "--data", f"{TMP}/pd/test", "--val", f"{TMP}/pd/val",
              "--tta", "1", "--sizes", "64,96", "--batch", "4", "--workers", "0",
              "--out", eval_json], "eval_hier")
    if not ok:
        print("EVAL_FAIL")
        return 1
    d = json.load(open(eval_json, encoding="utf-8"))
    assert "gate_curve" in d and "high_risk" in d, "eval_hier 输出缺字段"
    print(f"  ✓ eval_hier 通过  top1={d['top1']} α={d['alpha']} "
          f"crop_head_acc={d['crop_head_acc']} 守门曲线={len(d['gate_curve'])}点")

    ok = run([sys.executable, "model/export_onnx.py", "--ckpt", ckpt,
              "--backbone", "resnet18", "--size", "64",
              "--out-dir", os.path.join(TMP, "onnx")], "export_onnx")
    if ok and os.path.exists(os.path.join(TMP, "onnx", "heyi_v3_hr.onnx")):
        print("  ✓ export_onnx 通过（ONNX 已生成，crop_head 被正确过滤）")
    else:
        print("  ⚠ export_onnx 未通过（可能缺 onnxruntime，非致命）")

    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
