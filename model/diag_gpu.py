# -*- coding: utf-8 -*-
"""GPU/推理速度诊断: 确认 ROCm 是否真正可用, 避免评测脚本静默退化到 CPU。"""
import time

import torch
import torchvision.models as m

print("torch:", torch.__version__, flush=True)
print("cuda avail:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    print("!!! GPU 不可用, 评测将退化到 CPU", flush=True)
    raise SystemExit(1)

print("device:", torch.cuda.get_device_name(0), flush=True)
t0 = time.time()
x = torch.randn(8, 3, 224, 224, device="cuda")
torch.cuda.synchronize()
print("alloc tensor sec:", round(time.time() - t0, 2), flush=True)

net = m.resnet18(weights=None)
net.fc = torch.nn.Linear(512, 39)
net = net.to("cuda").eval()

t0 = time.time()
with torch.no_grad():
    for _ in range(5):
        net(x)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"5x forward (batch8) sec: {dt:.3f} -> {dt/5*1000:.1f} ms/batch", flush=True)

# 模拟完整评测: 452 crops x 5 视图
n_total = 452 * 5
est = (n_total / 8) * (dt / 5)
print(f"预估 452 crops x 5 视图耗时: {est:.1f} s", flush=True)
