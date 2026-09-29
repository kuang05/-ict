# -*- coding: utf-8 -*-
"""
禾医 · 田间域适应训练 v2
========================
要解决的问题
------------
v1 的 PlantDoc(真实田间) top1 仅 ~56%, 而 PlantVillage(实验室) 高达 99.3% ——
典型的 **域偏移(domain shift)**: 实验室图像背景干净、光照均匀、单叶居中;
田间图像背景杂乱、光照多变、多叶重叠。

v1 的三处结构性缺陷
--------------------
1. 输出空间是 PlantVillage 39 类, 但田间任务只有 27 个有效类。模型把叶片判成
   无关作物(玉米/橙子/辣椒)时必然失分, 这部分损失与"病害识别能力"无关。
   -> v2 把输出空间收缩为 27 类。
2. 模型选择看 PlantVillage 同域 val, 与目标指标(田间泛化)脱节。
   -> v2 用 PlantDoc 留出的 val(原图级划分, 见 prepare_plantdoc.py)选模型。
3. 只靠颜色/几何增强对抗域差, 未做特征层域泛化。
   -> v2 引入 **MixStyle**(Zhou et al., CVPR 2021, arXiv:2104.02008):
      训练时随机混合 batch 内样本的特征通道均值/方差, 迫使分类器不依赖
      "背景/光照"这类域相关低层统计, 从而提升跨域泛化。实现仅需十几行,
      无需目标域标签, 是域泛化方向性价比最高的组件之一。

训练数据
--------
* 源域 PlantVillage(27 类子集, 每类上限 --pv-per-class 张): 提供病害判别力
* 目标域 PlantDoc train 裁叶: 提供田间分布
两者混合训练, 目标域重复采样以提升占比。

用法
----
/opt/venv/bin/python model/train_field_v2.py --epochs 20 --backbone resnet50
"""
import argparse, json, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from prepare_plantdoc import PD2PV

CLASSES27 = sorted(set(PD2PV.values()))
C2I = {c: i for i, c in enumerate(CLASSES27)}


# ---------------------------------------------------------------- MixStyle
class MixStyle(nn.Module):
    """MixStyle: 随机混合 batch 内样本的特征通道统计量(均值/方差)。

    Zhou et al., "Domain Generalization with MixStyle", CVPR 2021.
    仅在训练时生效; p 为触发概率, alpha 控制 Beta 分布形状。
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = torch.sqrt(var + self.eps)
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)
        lmda = np.random.beta(self.alpha, self.alpha)
        lmda = float(np.clip(lmda, 0.0, 1.0))
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha})"


# ---------------------------------------------------------------- 数据
def tf_train():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
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


def tf_val():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ListDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


def scan_dir(root, per_class=None, seed=42):
    """扫描类别目录, 返回 [(path, label27)]。目录名必须是 27 类之一。"""
    rng = random.Random(seed)
    items = []
    for cls in os.listdir(root):
        d = os.path.join(root, cls)
        if not os.path.isdir(d) or cls not in C2I:
            continue
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        if per_class:
            files = files[:per_class]
        items.extend((p, C2I[cls]) for p in files)
    return items


# ---------------------------------------------------------------- 模型
def build_model(backbone, use_mixstyle, init=None):
    """init: 本地权重文件(如 PlantVillage 预训练的 best_robust.pth)。
    它比 ImageNet 预训练更贴近病害特征; 分类头因输出空间变化会被跳过重建。"""
    if backbone == "resnet50":
        net = models.resnet50(weights=None if init else models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, len(CLASSES27))
        if use_mixstyle:
            net.layer1 = nn.Sequential(net.layer1, MixStyle())
            net.layer2 = nn.Sequential(net.layer2, MixStyle())
    elif backbone == "resnet18":
        net = models.resnet18(weights=None if init else models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, len(CLASSES27))
        if use_mixstyle:
            net.layer1 = nn.Sequential(net.layer1, MixStyle())
            net.layer2 = nn.Sequential(net.layer2, MixStyle())
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(CLASSES27))
        if use_mixstyle:
            # features 序列中 stage2/4/6 之后插入
            feats = list(net.features)
            net.features = nn.Sequential(
                feats[0], feats[1], feats[2], MixStyle(),
                feats[3], feats[4], MixStyle(),
                feats[5], feats[6], feats[7], feats[8],
            )
    else:
        raise ValueError(f"unknown backbone {backbone}")

    if init and os.path.exists(init):
        ck = torch.load(init, map_location="cpu", weights_only=False)
        src = ck.get("model", ck)
        own = net.state_dict()
        n_loaded = 0
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                n_loaded += 1
        net.load_state_dict(own, strict=False)
        print(f"[init] 从 {init} 迁移 {n_loaded}/{len(own)} 层(分类头按 27 类重建)")
    elif init:
        print(f"[init] 警告: {init} 不存在, 回退到随机初始化")
    return net


def clean_state_dict(sd):
    """去掉 MixStyle 包装引入的层级前缀。

    MixStyle 通过 nn.Sequential(layerN, MixStyle()) 插入, state_dict 的键会变成
    "layer1.0.conv1.weight" 等; 推理端用标准 torchvision 模型加载会 key 不匹配。
    这里把 "layerN.0." 还原为 "layerN.", 使落盘权重与标准结构一致。
    """
    out = {}
    for k, v in sd.items():
        for name in ("layer1", "layer2", "layer3"):
            if k.startswith(name + ".0."):
                k = k.replace(name + ".0.", name + ".", 1)
                break
        out[k] = v
    return out


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    per = np.zeros(len(CLASSES27))
    per_n = np.zeros(len(CLASSES27))
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = net(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.numel()
        for yi, pi in zip(y.cpu().tolist(), p.cpu().tolist()):
            per_n[yi] += 1
            per[yi] += int(yi == pi)
    acc = correct / max(total, 1)
    return acc, per, per_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-data", default="/root/data/pv/Plant_leave_diseases_dataset_without_augmentation")
    ap.add_argument("--pd-root", default="/root/data/pd_v2")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pv-per-class", type=int, default=600)
    ap.add_argument("--pd-repeat", type=int, default=3,
                    help="目标域数据重复采样次数(提升田间分布占比)")
    ap.add_argument("--init", default="",
                    help="本地预训练权重(如 PlantVillage 训练的 best_robust.pth); "
                         "给定时不再下载 ImageNet 权重")
    ap.add_argument("--no-mixstyle", action="store_true", help="消融: 关闭 MixStyle")
    ap.add_argument("--out", default="/root/he-yi/model/checkpoints/best_v2.pth")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    pv_items = scan_dir(args.pv_data, args.pv_per_class)
    pd_train = scan_dir(os.path.join(args.pd_root, "train"))
    pd_val = scan_dir(os.path.join(args.pd_root, "val"))
    if not pd_train:
        raise SystemExit(f"目标域训练集为空, 请先运行 prepare_plantdoc.py -> {args.pd_root}")

    train_items = pv_items + pd_train * args.pd_repeat
    random.shuffle(train_items)
    print(f"[v2] device={device} backbone={args.backbone} mixstyle={not args.no_mixstyle}")
    print(f"[v2] PV27={len(pv_items)} PD-train={len(pd_train)}x{args.pd_repeat} "
          f"PD-val={len(pd_val)} total-train={len(train_items)}")

    # 类别权重: 缓解长尾(按混合后频率的平方根倒数)
    cnt = Counter(y for _, y in train_items)
    w = np.array([1.0 / np.sqrt(max(cnt[i], 1)) for i in range(len(CLASSES27))])
    w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    train_loader = DataLoader(ListDataset(train_items, tf_train()), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(ListDataset(pd_val, tf_val()), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=True)

    net = build_model(args.backbone, not args.no_mixstyle).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = 0.0
    best_ep = 0
    history = []
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = loss_fn(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            nb += 1
        sched.step()

        acc, per, per_n = evaluate(net, val_loader, device)
        history.append({"epoch": ep, "loss": round(running / max(nb, 1), 4),
                        "pd_val_acc": round(acc, 4)})
        print(f"[v2] epoch{ep} loss={running/max(nb,1):.4f} "
              f"PDval_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if acc > best:
            best, best_ep = acc, ep
            torch.save({
                "model": clean_state_dict(net.state_dict()),
                "classes": CLASSES27,
                "backbone": args.backbone,
                "mixstyle": not args.no_mixstyle,
                "pd_val_acc": acc,
                "epoch": ep,
                "protocol": "PlantDoc 27类输出空间; 原图级划分 val",
            }, args.out)
            print(f"[v2] saved {args.out} (PDval={acc:.4f})", flush=True)

    # 保存训练记录
    rec = {"tag": args.tag, "args": vars(args), "best_pd_val_acc": round(best, 4),
           "best_epoch": best_ep, "history": history,
           "per_class_val_at_best": {CLASSES27[i]: round(float(per[i] / max(per_n[i], 1)), 4)
                                     for i in range(len(CLASSES27))}}
    rec_path = args.out.replace(".pth", "_train_record.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[v2] done. best_PD_val={best:.4f} @ep{best_ep} -> {rec_path}")


if __name__ == "__main__":
    main()
