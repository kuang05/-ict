# -*- coding: utf-8 -*-
"""禾医 · NPU 检查点转 CPU 格式（让昇腾训练的权重能在任意纯 CPU 环境加载）
==================================================================
背景：torch_npu 2.1 保存的 .pth 含 NPU storage 布局，原生 torch（任意版本）
     直接加载报 "record size does not match expected size ... dtype Byte"。
     本脚本在装了 torch_npu 的服务器上：加载 -> 全部 tensor 转 CPU -> 重存。
     转换后的文件可被纯 CPU torch（本地演示/评测）直接加载。

用法（服务器上，PyTorch-2.1.0 环境）:
  python convert_ckpt_to_cpu.py --in model/checkpoints/best_v11.pth \
      --out model/checkpoints/best_v11_cpu.pth
"""
import argparse
import torch


def to_cpu(o):
    if isinstance(o, dict):
        return {k: to_cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(to_cpu(v) for v in o)
    if torch.is_tensor(o):
        return o.detach().cpu()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu")
    ck = to_cpu(ck)
    torch.save(ck, args.dst, _use_new_zipfile_serialization=True)
    n = sum(1 for _ in [ck]) if isinstance(ck, dict) else 1
    keys = list(ck.keys()) if isinstance(ck, dict) else ["<tensor>"]
    print(f"converted: {args.src} -> {args.dst}")
    print(f"top-level keys: {keys[:8]}")


if __name__ == "__main__":
    main()
