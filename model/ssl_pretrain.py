# -*- coding: utf-8 -*-
"""禾医 · 田间域自适应自监督预训练（DAP-SimCLR）

真正的问题在哪
--------------
本任务的核心困难是**域偏移**：源域 PlantVillage 是实验室均匀背景，
目标域 PlantDoc 是真实田间（杂乱背景 / 多变光照 / 遮挡 / 多叶片）。
IEEE ICETCI 2026 的系统研究（14 个模型变体）结论明确：
"存在**数据天花板**，把骨干放大 14 倍也**不提升**；出路是
**target-domain pretraining** 或能利用结构信息的对齐目标。"

我们手上有 **111,165 张真实田间图**（`hf_field`，孟加拉作物数据集）。
它们不属于我们的 27 类，**无法当有标签数据用**，但**自监督根本不需要标签**。

本脚本的做法
------------
从 ImageNet 权重出发，在这 11 万张田间图上做 **SimCLR 对比学习**
（= 领域自适应预训练，Domain-Adaptive Pretraining），让骨干先"见过田间"，
再把学到的 backbone 交给 `train_hier.py --init` 做有监督微调。

设计要点
--------
* 图像 160px（比微调的 224 小，换取 3 倍速度；域特征是全局的，降分辨率损失很小）
* 双视图增强：RandomResizedCrop + 翻转 + 强颜色抖动 + 灰度 + 模糊
* 2 层 MLP 投影头（训练后丢弃，只保留 backbone）
* NT-Xent 损失，τ=0.5；FP16 + GradScaler
* 产出 ckpt 的键为 `backbone.*`，可直接被 `train_hier.py --init` 加载

用法
----
python model/ssl_pretrain.py --data /home/ma-user/data/ssl_field \
  --backbone resnet50 --epochs 30 --batch 256 --size 160 \
  --out model/checkpoints/ssl_r50.pth
"""
import argparse, json, os, random, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_plantdoc import PD2PV  # noqa: E402

CLASSES27 = sorted(set(PD2PV.values()))


class TwoViewDS(Dataset):
    """无标签图像集：每次返回同一张图的两个增强视图。"""

    def __init__(self, root, tf):
        self.items = [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.open(self.items[i]).convert("RGB")
        return self.tf(img), self.tf(img)


def ssl_tf(size=160):
    """SimCLR 强增强：两份独立随机 → 同一图的两种"风格"。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class SimCLR(nn.Module):
    def __init__(self, backbone="resnet50", proj_dim=128, hidden=2048, pretrained=True):
        super().__init__()
        builder = {"resnet18": models.resnet18, "resnet50": models.resnet50,
                   "resnet101": models.resnet101}[backbone]
        weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                   "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
                   "resnet101": models.ResNet101_Weights.IMAGENET1K_V1}[backbone] \
            if pretrained else None
        net = builder(weights=weights)
        feat = net.fc.in_features
        net.fc = nn.Identity()          # 去掉分类头 → 纯特征
        self.backbone = net
        self.feat_dim = feat
        self.proj = nn.Sequential(
            nn.Linear(feat, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim))

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)


def nt_xent(z, tau=0.5):
    """z: (2N, d)，前 N 与后 N 互为正样本对。"""
    N2 = z.size(0)
    N = N2 // 2
    sim = z @ z.t() / tau
    sim = sim.masked_fill(torch.eye(N2, dtype=torch.bool, device=z.device), -1e9)
    targets = torch.cat([torch.arange(N, N2), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/ma-user/data/ssl_field")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张（调试用，0=全部）")
    ap.add_argument("--sample", type=int, default=0,
                    help="随机采样 N 张做预训练（0=全部）。用于控制时间预算。")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="墙钟时间预算（分钟），到点即停并保存当前权重（0=不限）")
    ap.add_argument("--out", default="model/checkpoints/ssl_r50.pth")
    ap.add_argument("--tag", default="ssl")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    device = "npu" if getattr(torch, "npu", None) is not None and torch.npu.is_available() \
        else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    ds = TwoViewDS(args.data, ssl_tf(args.size))
    if args.sample and args.sample < len(ds.items):
        ds.items = random.Random(42).sample(ds.items, args.sample)
    elif args.limit:
        ds.items = ds.items[:args.limit]
    print(f"[ssl] 无标签田间语料 {len(ds)} 张, backbone={args.backbone} "
          f"size={args.size} batch={args.batch} epochs={args.epochs}", flush=True)

    g = torch.Generator().manual_seed(42)
    pf = 6 if args.workers > 0 else None
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        generator=g, pin_memory=True, drop_last=True,
                        persistent_workers=(args.workers > 0), prefetch_factor=pf)

    model = SimCLR(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.npu.amp.GradScaler() if (device == "npu" and args.amp) else \
        torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.amp))

    def ctx():
        if not args.amp:
            import contextlib
            return contextlib.nullcontext()
        if device == "npu":
            return torch.npu.amp.autocast(enabled=True, dtype=torch.float16)
        return torch.amp.autocast("cuda", enabled=(device == "cuda"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t_start = time.time()

    def save_ckpt(ep, loss_val):
        sd = {f"backbone.{k}": v.cpu() for k, v in model.backbone.state_dict().items()}
        torch.save({"model": sd, "classes": list(CLASSES27),
                    "backbone": args.backbone, "arch": "ssl_dap_simclr",
                    "ssl_epoch": ep, "ssl_loss": float(loss_val)}, args.out)
        print(f"[ssl] saved {args.out} (ep{ep}, loss={loss_val:.4f})", flush=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run, nb = torch.zeros((), device=device), 0
        for va, vb in loader:
            x = torch.cat([va, vb], 0).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with ctx():
                z = model(x)
                loss = nt_xent(z, args.tau)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.detach()
            nb += 1
        sched.step()
        cur = (run / max(nb, 1)).item()
        print(f"[ssl] epoch{ep} loss={cur:.4f} "
              f"({time.time()-t0:.0f}s, {len(loader)} steps)", flush=True)
        if ep % 5 == 0 or ep == args.epochs:
            save_ckpt(ep, cur)
        if args.max_minutes and (time.time() - t_start) / 60.0 >= args.max_minutes:
            print(f"[ssl] 达到时间预算 {args.max_minutes:.0f} 分钟，提前结束", flush=True)
            save_ckpt(ep, cur)
            break
    else:
        save_ckpt(args.epochs, cur)

    rec = {"tag": args.tag, "args": vars(args), "n_images": len(ds)}
    with open(args.out.replace(".pth", "_record.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[ssl] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
