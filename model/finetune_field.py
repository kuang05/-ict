# -*- coding: utf-8 -*-
"""
分类器田间域适应微调: PlantVillage(主) + PlantDoc训练集裁叶(域适应) -> best_finetuned.pth
====================================================================================
协议: PlantDoc test 严格留出; PD训练裁叶仅来自 train parquet。
策略: 加载PV预训练best.pth, 全模型小学习率(1e-4)微调3轮, 每轮在PV-val(5%)上监控。
"""
import argparse, json, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, models, transforms

def tf_train_strong():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

def tf_val():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-data", required=True, help="PlantVillage 39类根目录")
    ap.add_argument("--pd-crops", default="/root/data/pd_crops_train")
    ap.add_argument("--init", default="/root/he-yi/model/checkpoints/best.pth")
    ap.add_argument("--out", default="/root/he-yi/model/checkpoints/best_finetuned.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    classes = ck["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    print(f"[HeYi-FT] device={device} classes={len(classes)}")

    full = datasets.ImageFolder(args.pv_data, transform=tf_train_strong())
    # PV 类名可能与39类表有细微差异(Mendeley命名一致), 校验
    assert full.classes == classes, f"PV类别不一致: {set(full.classes) ^ set(classes)}"
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(full), generator=g).tolist()
    n_val = int(len(full) * 0.05)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    pv_train = Subset(datasets.ImageFolder(args.pv_data, transform=tf_train_strong()), train_idx)
    pv_val = Subset(datasets.ImageFolder(args.pv_data, transform=tf_val()), val_idx)

    pd_crops = datasets.ImageFolder(args.pd_crops, transform=tf_train_strong())
    # PD crops 类目录名必须与39类一致
    remap = {i: c2i[c] for c, i in pd_crops.class_to_idx.items()}
    pd_targets = [remap[y] for y in pd_crops.targets]
    pd_crops.targets = pd_targets
    pd_crops.samples = [(p, remap[y]) for (p, y) in pd_crops.samples]
    print(f"[HeYi-FT] PV-train={len(pv_train)} PV-val={len(pv_val)} PD-crops={len(pd_crops)}")

    train_loader = DataLoader(ConcatDataset([pv_train, pd_crops]), batch_size=args.batch,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(pv_val, batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(ck["model"])
    net = net.to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        net.train(); t0 = time.time(); running = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward(); opt.step()
            running += loss.item()
            if i % 150 == 0:
                print(f"  ep{ep} step{i}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        net.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (net(x).argmax(1) == y).sum().item(); total += y.numel()
        acc = correct / max(total, 1)
        print(f"[HeYi-FT] epoch{ep} loss={running/len(train_loader):.4f} pv_val_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if acc >= best:
            best = acc
            torch.save({"model": net.state_dict(), "classes": classes, "val_acc": acc,
                        "epoch": ep, "arch": "resnet18_ft_pv+plantdoc"},
                       args.out)
            print(f"[HeYi-FT] saved {args.out} (pv_val={acc:.4f})", flush=True)
    print(f"[HeYi-FT] done. best_pv_val={best:.4f}")

if __name__ == "__main__":
    main()
