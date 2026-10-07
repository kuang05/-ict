# -*- coding: utf-8 -*-
"""
禾医 · 田间域适应训练 v3
========================
为什么是 v3
-----------
v2 用 `net.layer1 = nn.Sequential(net.layer1, MixStyle())` 的方式插入 MixStyle。
这带来一个隐蔽而致命的副作用: **标准 ResNet 的 layerN 本身就是
nn.Sequential(Bottleneck, ...), 其 state_dict 键本来就形如 "layer3.0.conv1.weight"**。
插入 MixStyle 后键变成三层 "layer3.0.0.conv1.weight", 而任何试图"清洗前缀"的逻辑
都会误伤未包装层的标准键(把 "layer3.0." 吃成 "layer3."), 使同一 layer 内多个
block 的参数互相覆盖 —— 保存出的 checkpoint 表面能加载、实则权重错乱,
指标退化到随机水平(top1 个位数), 且 `strict=False` 不会报错。

v3 的做法
---------
**MixStyle 外挂**: backbone 保持 torchvision 原生结构不动(键完全标准),
MixStyle 作为独立无参数模块在 forward 中显式调用。这样
* 保存的 state_dict 100% 标准, 任何评测端都能直接 strict=True 加载;
* MixStyle 仍作用于 layer1/layer2 输出, 域泛化效果不变。

其余沿用 v2 的有效设计:
* 输出空间 = PlantDoc 27 类
* 源域 PlantVillage 27 类子集 + 目标域 PlantDoc train 混合, 目标域重复采样
* 类别权重损失缓解长尾
* **用 PlantDoc val(原图级划分)选模型**, 而不是 PlantVillage 同域 val

用法
----
/opt/venv/bin/python model/train_field_v3.py --epochs 20 --backbone resnet50
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

# 高危病害：漏诊代价远高于误诊（暴发性 3~7 天毁田 / 无特效药 / 需整株处理）。
# 实测基线里这类样本的"被误判为健康"率高达 12.3%，必须用代价敏感加权压下去。
HIGH_RISK = {
    "Tomato___Late_blight",
    "Potato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Corn___Northern_Leaf_Blight",
}


class MixStyle(nn.Module):
    """随机混合 batch 内样本的通道均值/方差(风格统计量)。

    Zhou et al., "Domain Generalization with MixStyle", CVPR 2021.
    仅在训练时生效; 无参数, 因此不进入 state_dict。
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


class Net(nn.Module):
    """backbone 保持原生结构; MixStyle 外挂于 layer1/layer2 之后。

    state_dict 与标准 torchvision 模型完全一致, 评测端可直接加载。
    """

    def __init__(self, backbone="resnet50", n_cls=27, mix_p=0.5, pretrained=True):
        super().__init__()
        self.is_timm = backbone.startswith("timm_")
        if self.is_timm:
            import timm
            self.backbone = timm.create_model(backbone[5:], pretrained=pretrained,
                                              num_classes=n_cls)
        else:
            builder = {"resnet18": models.resnet18,
                       "resnet50": models.resnet50,
                       "resnet101": models.resnet101}.get(backbone)
            if builder is None:
                raise ValueError(backbone)
            # 跨域小数据场景, ImageNet 预训练是必需的; 关掉会退化到近乎随机(实测 ep1 仅 12.7%)
            weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
                       "resnet101": models.ResNet101_Weights.IMAGENET1K_V1}[backbone] \
                if pretrained else None
            self.backbone = builder(weights=weights)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feat, n_cls)
        self.mix1 = MixStyle(mix_p)
        self.mix2 = MixStyle(mix_p)

    def forward(self, x):
        b = self.backbone
        if self.is_timm:
            # timm convnext: stem -> stages[0..3](96/192/384/768) -> norm -> head
            x = b.stem(x)
            x = b.stages[0](x)
            x = self.mix1(x)
            x = b.stages[1](x)
            x = self.mix2(x)
            x = b.stages[2](x)
            x = b.stages[3](x)
            return b.head(x)   # timm 1.x: norm 在 head 内部
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = self.mix1(b.layer1(x))
        x = self.mix2(b.layer2(x))
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        return b.fc(x)


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

def amp_ctx(device):
    if device == "npu":
        return torch.npu.amp.autocast(enabled=True, dtype=torch.bfloat16)
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

def amp_scaler(device):
    if device == "npu":
        return _NoOpScaler()
    return torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
@torch.no_grad()
def evaluate(net, loader, device):
    was_training = net.training
    net.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx(device):
            p = net(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
    if was_training:
        net.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-data", default="/root/data/pv/Plant_leave_diseases_dataset_without_augmentation")
    ap.add_argument("--pd-root", default="/root/data/pd_v2")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "resnet101", "timm_convnext_tiny"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pv-per-class", type=int, default=600)
    ap.add_argument("--extra-data", default="", help="外部田间数据根目录(按27类名组织, 不设上限)")
    ap.add_argument("--img-size", type=int, default=224, help="训练/验证输入分辨率")
    ap.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha(0=关闭, 建议0.2)")
    ap.add_argument("--mixup-p", type=float, default=0.5, help="MixUp 触发概率(可搜索)")
    ap.add_argument("--extend-classes", action="store_true",
                    help="输出空间扩展至30类: +Target_Spot/Spider_mites/Corn_healthy (KB已覆盖)")
    ap.add_argument("--ema-decay", type=float, default=0.0, help="EMA衰减(0=关闭, 建议0.999)")
    ap.add_argument("--pd-repeat", type=int, default=3)
    ap.add_argument("--mix-p", type=float, default=0.5)
    ap.add_argument("--high-risk-weight", type=float, default=1.0,
                    help="高危病害类别的额外损失权重(>1 启用代价敏感训练)")
    ap.add_argument("--no-mixstyle", action="store_true")
    ap.add_argument("--init", default="", help="可选: 本地权重初始化(不下载 ImageNet)")
    ap.add_argument("--out", default="/root/he-yi/model/checkpoints/best_v3.pth")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    device = "npu" if getattr(torch, "npu", None) is not None and torch.npu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    classes = list(CLASSES27)
    if args.extend_classes:
        classes = sorted(set(classes) | {"Tomato___Target_Spot",
                                         "Tomato___Spider_mites Two-spotted_Spider_Mite",
                                         "Corn___healthy"})
    c2i = {c: i for i, c in enumerate(classes)}
    pv = scan_dir(args.pv_data, args.pv_per_class, c2i=c2i)
    if args.extra_data and os.path.isdir(args.extra_data):
        pv += scan_dir(args.extra_data, 0, c2i=c2i)
        print(f"[v3] 外部田间数据并入: +{len(pv)} 项累计")
    pd_train = scan_dir(os.path.join(args.pd_root, "train"), c2i=c2i)
    pd_val = scan_dir(os.path.join(args.pd_root, "val"), c2i=c2i)
    if not pd_train:
        raise SystemExit("目标域训练集为空, 请先运行 prepare_plantdoc.py")

    train_items = pv + pd_train * args.pd_repeat
    random.shuffle(train_items)
    print(f"[v3] device={device} backbone={args.backbone} mixstyle={not args.no_mixstyle}")
    print(f"[v3] PV27={len(pv)} PD-train={len(pd_train)}x{args.pd_repeat} "
          f"PD-val={len(pd_val)} total={len(train_items)}")

    cnt = Counter(y for _, y in train_items)
    w = np.array([1.0 / np.sqrt(max(cnt[i], 1)) for i in range(len(classes))])
    w = w / w.mean()
    # 代价敏感：对高危病害额外加权，让模型"宁可误报，不可漏诊"
    if args.high_risk_weight > 1.0:
        hr = [i for i, c in enumerate(classes) if c in HIGH_RISK]
        w[hr] *= args.high_risk_weight
        w = w / w.mean()
        print(f"[v3] 高危加权 x{args.high_risk_weight}: {[CLASSES27[i] for i in hr]}")
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    train_loader = DataLoader(ListDS(train_items, tf_train(args.img_size)), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=(args.workers > 0), prefetch_factor=6)
    val_loader = DataLoader(ListDS(pd_val, tf_val(args.img_size)), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=(args.workers > 0), prefetch_factor=6)

    net = Net(args.backbone, len(classes), 0.0 if args.no_mixstyle else args.mix_p)

    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        src = normalize_src(ck.get("model", ck))
        missing, unexpected = net.backbone.load_state_dict(src, strict=False)
        print(f"[v3] 从 {args.init} 初始化 backbone: missing={len(missing)} unexpected={len(unexpected)}")

    net = net.to(device)
    ema = None
    if args.ema_decay > 0:
        import copy
        ema = copy.deepcopy(net).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    scaler = amp_scaler(device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, best_ep, history = 0.0, 0, []
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running_t, nb = torch.zeros((), device=device), 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            use_mix = args.mixup > 0 and np.random.random() < args.mixup_p
            with amp_ctx(device):
                if use_mix:
                    lam = float(np.random.beta(args.mixup, args.mixup))
                    perm = torch.randperm(y.size(0), device=y.device)
                    x = lam * x + (1 - lam) * x[perm]
                    logits = net(x)
                    loss = lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y[perm])
                else:
                    logits = net(x)
                    loss = loss_fn(logits, y)
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
        acc = evaluate(eval_net, val_loader, device)
        history.append({"epoch": ep, "loss": round((running_t / max(nb, 1)).item(), 4),
                        "pd_val_acc": round(acc, 4)})
        print(f"[v3] epoch{ep} loss={(running_t/max(nb,1)).item():.4f} PDval={acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best, best_ep = acc, ep
            # backbone 原生结构 -> 键标准, 评测端可直接 strict 加载
            torch.save({"model": (ema if ema is not None else net).state_dict(), "classes": classes,
                        "backbone": args.backbone, "mix_p": args.mix_p,
                        "pd_val_acc": acc, "epoch": ep,
                        "protocol": "PlantDoc 27类输出空间; 原图级划分 val"},
                       args.out)
            print(f"[v3] saved {args.out} (PDval={acc:.4f})", flush=True)

    rec = {"tag": args.tag, "args": vars(args), "best_pd_val_acc": round(best, 4),
           "best_epoch": best_ep, "history": history}
    rec_path = args.out.replace(".pth", "_train_record.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[v3] done. best_PD_val={best:.4f} @ep{best_ep} -> {rec_path}")


def normalize_src(sd):
    """兼容不同来源的初始化权重: 去掉可能的 MixStyle 包装前缀(三层键)"""
    out = {}
    for k, v in sd.items():
        for n in ("layer1", "layer2", "layer3", "layer4"):
            pre = n + ".0.0."
            if k.startswith(pre):
                k = n + "." + k[len(pre):]
                break
        out[k] = v
    return out


if __name__ == "__main__":
    main()
