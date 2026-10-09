# -*- coding: utf-8 -*-
"""
禾医 · 模型导出 ONNX（端侧推理用）
====================================
为什么要导出 ONNX
----------------
往届特等奖作品几乎都有"端侧"实体。我们没有硬件（OrangePi 待购），
但可以**把手机变成端侧**：ONNX 模型 + ONNX Runtime Web，
手机浏览器即可本地推理，不依赖后端服务器。

这一步是"端-边-云"架构里"端"的技术底座：
* 无网也能诊断（真正的端侧推理）
* 演示时拔掉网线照样出结果 —— 这是纯云方案做不到的

产出
----
1. `heyi_v3_hr.onnx`            FP32 模型
2. `heyi_v3_hr_int8.onnx`       动态量化版（体积约 1/4，手机友好）
3. `heyi_onnx_meta.json`        类别表 + 预处理参数（前端要用）

用法
----
/opt/venv/bin/python model/export_onnx.py --ckpt model/checkpoints/best_v3_hr.pth
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_field_v3 import Net, CLASSES27  # noqa: E402


def build(ckpt, backbone):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    classes = ck.get("classes", CLASSES27)
    is_hier = str(ck.get("arch", "")).startswith("v4_hierarchical")
    # 层次化双头 ckpt：病害头可能落在 backbone.fc(resnet) 或 head_dis(timm)，统一用 NetV4
    if is_hier:
        from train_hier import NetV4, CROPS
        crops = ck.get("crops", CROPS)
        net = NetV4(backbone, len(classes), len(crops), mix_p=0.0, pretrained=False)
    elif any(k.startswith("backbone.") for k in ck["model"].keys()):
        # pretrained=False: 权重马上被 load_state_dict 覆盖, 无需下载 ImageNet
        net = Net(backbone, len(classes), mix_p=0.0, pretrained=False)
    else:
        from torchvision import models
        net = getattr(models, backbone)(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    if is_hier:
        # NetV4 自身含 crop_head，必须全量加载（ONNX 只导出病害头，crop_head 不参与前向）
        sd = dict(ck["model"])
    else:
        # 单头 Net：ckpt 里可能残留 crop_head.*，过滤掉
        sd = {k: v for k, v in ck["model"].items() if not k.startswith("crop_head.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"权重加载失败 missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()
    return net, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/checkpoints/best_v3_hr.pth")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--out-dir", default="model/onnx")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quantize", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    net, classes = build(a.ckpt, a.backbone)

    fp32 = os.path.join(a.out_dir, "heyi_v3_hr.onnx")
    dummy = torch.randn(1, 3, a.size, a.size)
    torch.onnx.export(
        net, dummy, fp32,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, do_constant_folding=True,
    )
    print(f"[export] FP32 -> {fp32}  ({os.path.getsize(fp32)/1e6:.1f} MB)")

    # 数值一致性校验：ONNX 与 PyTorch 输出必须一致，否则端侧结论不可信
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = net(dummy).numpy()
        got = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        print(f"[check] ONNX vs PyTorch 最大绝对误差 = {diff:.6f} "
              f"({'通过' if diff < 1e-3 else '⚠ 偏差偏大'})")
    except ImportError:
        print("[check] 未安装 onnxruntime，跳过一致性校验")

    # 动态量化（INT8）——体积约 1/4，手机端加载更快
    if a.quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8 = os.path.join(a.out_dir, "heyi_v3_hr_int8.onnx")
            quantize_dynamic(fp32, int8, weight_type=QuantType.QUInt8)
            print(f"[export] INT8 -> {int8}  ({os.path.getsize(int8)/1e6:.1f} MB)")
        except ImportError:
            print("[export] 未安装 onnxruntime.quantization，跳过量化")

    meta = {
        "classes": classes,
        "n_classes": len(classes),
        "input_size": a.size,
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "source_ckpt": os.path.basename(a.ckpt),
        "backbone": a.backbone,
        "note": "端侧推理用；预处理须与训练一致（Resize→CenterCrop→Normalize）",
    }
    mp = os.path.join(a.out_dir, "heyi_onnx_meta.json")
    json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[export] meta -> {mp}")


if __name__ == "__main__":
    main()
