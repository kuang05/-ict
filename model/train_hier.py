# -*- coding: utf-8 -*-
"""
禾医 · 田间域适应训练 v4 —— 层次化多任务重构版
================================================
为什么 v4（相对 v3 的**架构级**改动，不是调参）
---------------------------------------------
v3 是单一 27 类 flat softmax。它的**硬上限在作物识别**（crop_acc 86.7%）：
一旦作物判错，病害必错。而"区分番茄/马铃薯/玉米"本身远比"区分早疫/晚疫"
简单 —— 让一个 flat 模型同时承担两件事，特征会被细粒度病害主导，
作物级判别力反而不够。

v4 重构为**层次化多任务**：

    backbone ──┬── 病害头(n_cls=27)    ← 主任务, 键 backbone.fc.*
               └── 作物头(n_crop)      ← 辅助任务, 键 crop_head.*

    L = CE_disease(类别权重 + 高危加权 + label_smoothing) + λ · CE_crop

辅助头的梯度迫使 backbone 保留"作物级"可分性，而作物先验正是层次推理的基础。

推理端配套**层次融合**（见 eval_v4.py）：
    p_final(class) ∝ p_disease(class) · p_crop(crop_of(class))^α
α>0 时用作物先验抑制跨作物误判（如把番茄晚疫误判成马铃薯晚疫）；α 在 PlantDoc val 上选。

设计上刻意保持 **backbone 键与 v3 完全一致**（病害头仍放在 backbone.fc），
因此 v3 的评测脚本可直接评测 v4 的病害头（crop_head.* 会被自动忽略）。

其余沿用 v3 的全部有效设计：MixStyle 外挂、目标域重复采样、类别平衡权重、
MixUp、EMA、用 PlantDoc val 选模型。

用法
----
python model/train_field_v4.py --pv-data ... --pd-root ... --extra-data ... \
  --backbone resnet101 --epochs 30 --batch 64 --crop-weight 0.3 \
  --out model/checkpoints/best_v4arch.pth --tag v4arch
"""
import argparse, json, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from prepare_plantdoc import PD2PV

CLASSES27 = sorted(set(PD2PV.values()))
C2I27 = {c: i for i, c in enumerate(CLASSES27)}


def crop_of(cls):
    """'Tomato___Late_blight' -> 'Tomato'"""
    return cls.split("___")[0]


CROPS = sorted(set(crop_of(c) for c in CLASSES27))
C2I_CROP = {c: i for i, c in enumerate(CROPS)}

# 高危病害：漏诊代价远高于误诊（暴发性 3~7 天毁田 / 无特效药 / 需整株处理）
HIGH_RISK = {
    "Tomato___Late_blight",
    "Potato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Corn___Northern_Leaf_Blight",
}


class MixStyle(nn.Module):
    """随机混合 batch 内样本的通道均值/方差（风格统计量）。

    Zhou et al., "Domain Generalization with MixStyle", CVPR 2021.
    仅训练时生效；无参数，不进入 state_dict。
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        if B < 2:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        var = x.var(dim=(2, 3), keepdim=True).detach()
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)
        lmda = float(np.clip(np.random.beta(self.alpha, self.alpha), 0.0, 1.0))
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha})"


class NetV4(nn.Module):
    """层次化双头网络。

    backbone 保持 torchvision 原生结构（键标准，可直接 strict 加载）；
    病害头复用 backbone.fc（键与 v3 一致），作物辅助头为独立的 crop_head。
    """

    def __init__(self, backbone="resnet101", n_cls=27, n_crop=10, mix_p=0.5, pretrained=True):
        super().__init__()
        self.is_timm = backbone.startswith("timm_")
        if self.is_timm:
            # 现代骨干（ConvNeXt/Swin 等，建议 ImageNet-22k 预训练）：
            # num_classes=0 → 纯特征提取器，分类头由本类自行提供（便于双头）
            import timm
            self.backbone = timm.create_model(backbone[5:], pretrained=pretrained,
                                              num_classes=0)
            feat = self.backbone.num_features
            self.head_dis = nn.Linear(feat, n_cls)     # 病害头
            self.crop_head = nn.Linear(feat, n_crop)   # 作物辅助头
        else:
            builder = {"resnet18": models.resnet18,
                       "resnet50": models.resnet50,
                       "resnet101": models.resnet101}.get(backbone)
            if builder is None:
                raise ValueError(f"不支持的 backbone: {backbone}")
            weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
                       "resnet101": models.ResNet101_Weights.IMAGENET1K_V1}[backbone] \
                if pretrained else None
            self.backbone = builder(weights=weights)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feat, n_cls)  # 病害头（键 backbone.fc.*，兼容 v3 评测端）
            self.crop_head = nn.Linear(feat, n_crop)   # 作物辅助头
        self.mix1 = MixStyle(mix_p)
        self.mix2 = MixStyle(mix_p)

    def features(self, x):
        b = self.backbone
        if self.is_timm:
            # ConvNeXt 系：stem -> stages[0..3]；MixStyle 外挂于 stage0/stage1 之后
            y = b.stem(x)
            y = self.mix1(b.stages[0](y))
            y = self.mix2(b.stages[1](y))
            y = b.stages[2](y)
            y = b.stages[3](y)
            return b.forward_head(y, pre_logits=True)   # pooled feature (B, num_features)
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = self.mix1(b.layer1(x))
        x = self.mix2(b.layer2(x))
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, both=False):
        f = self.features(x)
        d = self.head_dis(f) if self.is_timm else self.backbone.fc(f)
        if both:
            return d, self.crop_head(f)
        return d


def tf_train(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.15)], p=0.8),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def tf_val(size=224):
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)), transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ListDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def scan_dir(root, per_class=None, seed=42, c2i=None):
    c2i = c2i if c2i is not None else C2I27
    rng = random.Random(seed)
    items = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d) or cls not in c2i:
            continue
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        if per_class:
            files = files[:per_class]
        items.extend((p, c2i[cls]) for p in files)
    return items


import contextlib as _ctxlib


def amp_ctx(device, use_amp=True):
    """AMP 上下文。

    ⚠️ 2026-10-08 修复：910B 的 npu autocast **只支持 float16**，
    原版传 bfloat16 会被 torch 静默禁用（warning 后继续跑 FP32）→ 白丢约一倍速度。
    这里显式用 float16，并配 GradScaler 防梯度下溢。
    """
    if not use_amp:
        return _ctxlib.nullcontext()
    if device == "npu":
        return torch.npu.amp.autocast(enabled=True, dtype=torch.float16)
    if device == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return _ctxlib.nullcontext()


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def step(self, opt):
        opt.step()

    def update(self):
        pass


def amp_scaler(device, use_amp=True):
    if not use_amp:
        return _NoOpScaler()
    if device == "npu":
        return torch.npu.amp.GradScaler()
    return torch.amp.GradScaler("cuda", enabled=(device == "cuda"))


@torch.no_grad()
def evaluate(net, loader, device, use_amp=False):
    """返回 (病害top1, 作物top1)。作物准确率用 crop_head 直接算。"""
    was_training = net.training
    net.eval()
    d_ok = c_ok = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx(device, use_amp):
            d, c = net(x, both=True)
        d_ok += (d.argmax(1) == y).sum().item()
        crop_gt = torch.tensor([C2I_CROP[crop_of(CLASSES27[i])] for i in y.tolist()],
                               device=device)
        c_ok += (c.argmax(1) == crop_gt).sum().item()
        total += y.size(0)
    if was_training:
        net.train()
    return d_ok / max(total, 1), c_ok / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-data", default="/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation")
    ap.add_argument("--pd-root", default="/home/ma-user/data/pd_v2")
    ap.add_argument("--backbone", default="resnet101",
                    help="resnet18/50/101 或 timm_<model_name>（如 timm_convnext_tiny.fb_in22k_ft_in1k）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--pv-per-class", type=int, default=600)
    ap.add_argument("--extra-data", default="", help="外部田间数据根目录（按27类名组织）")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-weight", type=float, default=0.3, help="作物辅助头损失权重 λ")
    ap.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha(0=关闭)")
    ap.add_argument("--mixup-p", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA衰减(0=关闭)。⚠️本任务已证伪 EMA+MixUp(v15 61.28%% vs v11 67.92%%)，默认关闭。")
    ap.add_argument("--pd-repeat", type=int, default=3)
    ap.add_argument("--mix-p", type=float, default=0.5)
    ap.add_argument("--high-risk-weight", type=float, default=1.0)
    ap.add_argument("--no-mixstyle", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子。改种子可产出同配置不同权重的模型，用于集成增多样性。")
    ap.add_argument("--amp", action="store_true",
                    help="启用 NPU FP16 混合精度(含 GradScaler)。默认 FP32 更稳。")
    ap.add_argument("--init", default="", help="可选: 从 v3/v4 ckpt 初始化 backbone")
    ap.add_argument("--out", default="model/checkpoints/best_v4arch.pth")
    ap.add_argument("--tag", default="v4arch")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    device = "npu" if getattr(torch, "npu", None) is not None and torch.npu.is_available() \
        else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    classes = list(CLASSES27)
    c2i = {c: i for i, c in enumerate(classes)}
    pv = scan_dir(args.pv_data, args.pv_per_class, c2i=c2i)
    if args.extra_data and os.path.isdir(args.extra_data):
        pv += scan_dir(args.extra_data, 0, c2i=c2i)
        print(f"[v4] 外部田间数据并入: +{len(pv)} 项累计")
    pd_train = scan_dir(os.path.join(args.pd_root, "train"), c2i=c2i)
    pd_val = scan_dir(os.path.join(args.pd_root, "val"), c2i=c2i)
    if not pd_train:
        raise SystemExit("目标域训练集为空，请先运行 prepare_plantdoc.py")

    train_items = pv + pd_train * args.pd_repeat
    random.shuffle(train_items)
    print(f"[v4] device={device} backbone={args.backbone} 双头(病害{c2i and len(classes)}+作物{len(CROPS)}) "
          f"mixstyle={not args.no_mixstyle} crop_weight={args.crop_weight}")
    print(f"[v4] PV27={len(pv)} PD-train={len(pd_train)}x{args.pd_repeat} "
          f"PD-val={len(pd_val)} total={len(train_items)}")

    cnt = Counter(y for _, y in train_items)
    w = np.array([1.0 / np.sqrt(max(cnt[i], 1)) for i in range(len(classes))])
    w = w / w.mean()
    if args.high_risk_weight > 1.0:
        hr = [i for i, c in enumerate(classes) if c in HIGH_RISK]
        w[hr] *= args.high_risk_weight
        w = w / w.mean()
        print(f"[v4] 高危加权 x{args.high_risk_weight}: {[classes[i] for i in hr]}")
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    # 固定 shuffle 的随机源 → 训练顺序可复现（比赛要求代码可复现）
    g = torch.Generator().manual_seed(args.seed)
    # prefetch_factor 仅在多进程时可用；workers=0（调试/自检）时必须为 None
    pf = 6 if args.workers > 0 else None
    train_loader = DataLoader(ListDS(train_items, tf_train(args.img_size)), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, generator=g,
                              pin_memory=True, drop_last=True,
                              persistent_workers=(args.workers > 0), prefetch_factor=pf)
    val_loader = DataLoader(ListDS(pd_val, tf_val(args.img_size)), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=(args.workers > 0), prefetch_factor=pf)

    net = NetV4(args.backbone, len(classes), len(CROPS),
                0.0 if args.no_mixstyle else args.mix_p)

    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        raw = ck.get("model", ck)
        src = {k[len("backbone."):] if k.startswith("backbone.") else k: v
               for k, v in raw.items() if k.startswith("backbone.")}
        missing, unexpected = net.backbone.load_state_dict(src, strict=False)
        print(f"[v4] 从 {args.init} 初始化 backbone: missing={len(missing)} unexpected={len(unexpected)}")

    net = net.to(device)
    ema = None
    if args.ema_decay > 0:
        import copy
        ema = copy.deepcopy(net).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_dis = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    loss_crop = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = bool(args.amp)
    scaler = amp_scaler(device, use_amp)
    print(f"[v4] amp={'fp16' if use_amp else 'fp32'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, best_ep, history = 0.0, 0, []
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running_t, nb = torch.zeros((), device=device), 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            yc = torch.tensor([C2I_CROP[crop_of(classes[i])] for i in y.tolist()],
                              device=device)
            opt.zero_grad(set_to_none=True)
            use_mix = args.mixup > 0 and np.random.random() < args.mixup_p
            with amp_ctx(device, use_amp):
                if use_mix:
                    lam = float(np.random.beta(args.mixup, args.mixup))
                    perm = torch.randperm(y.size(0), device=y.device)
                    x = lam * x + (1 - lam) * x[perm]
                    d, c = net(x, both=True)
                    ld = lam * loss_dis(d, y) + (1 - lam) * loss_dis(d, y[perm])
                    lc = lam * loss_crop(c, yc) + (1 - lam) * loss_crop(c, yc[perm])
                else:
                    d, c = net(x, both=True)
                    ld = loss_dis(d, y)
                    lc = loss_crop(c, yc)
                loss = ld + args.crop_weight * lc
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                with torch.no_grad():
                    for q_e, q in zip(ema.parameters(), net.parameters()):
                        q_e.mul_(args.ema_decay).add_(q, alpha=1 - args.ema_decay)
                    for b_e, b in zip(ema.buffers(), net.buffers()):
                        b_e.copy_(b)
            running_t += loss.detach()
            nb += 1
        sched.step()
        eval_net = ema if ema is not None else net
        acc, cacc = evaluate(eval_net, val_loader, device)
        history.append({"epoch": ep, "loss": round((running_t / max(nb, 1)).item(), 4),
                        "pd_val_dis_acc": round(acc, 4), "pd_val_crop_acc": round(cacc, 4)})
        print(f"[v4] epoch{ep} loss={(running_t/max(nb,1)).item():.4f} "
              f"PDval_dis={acc:.4f} PDval_crop={cacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best, best_ep = acc, ep
            torch.save({"model": (ema if ema is not None else net).state_dict(),
                        "classes": classes, "crops": CROPS, "backbone": args.backbone,
                        "mix_p": args.mix_p, "crop_weight": args.crop_weight,
                        "pd_val_dis_acc": acc, "pd_val_crop_acc": cacc, "epoch": ep,
                        "arch": "v4_hierarchical_multitask",
                        "protocol": "PlantDoc 27类输出空间; 原图级划分 val"},
                       args.out)
            print(f"[v4] saved {args.out} (PDval_dis={acc:.4f} crop={cacc:.4f})", flush=True)

    rec = {"tag": args.tag, "args": vars(args), "best_pd_val_dis_acc": round(best, 4),
           "best_epoch": best_ep, "history": history}
    rec_path = args.out.replace(".pth", "_train_record.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[v4] done. best_PDval_dis={best:.4f} @ep{best_ep} -> {rec_path}")


if __name__ == "__main__":
    main()
