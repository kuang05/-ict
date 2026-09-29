# -*- coding: utf-8 -*-
"""
禾医·PlantVillage 病虫害识别训练脚本（PyTorch 版）
====================================================
数据: PlantVillage 38类 (https://github.com/spMohanty/PlantVillage-Dataset)
策略: ImageNet预训练ResNet18迁移学习 —— 冻结浅层、微调layer4+分类头
      (参考公开SOTA: 同结构方案在PlantVillage上可达 98%+ Top-1)
用法:
    python train_pytorch.py --data /path/to/plantvillage/color --epochs 8 --batch 64
输出: model/checkpoints/best.pth  (含类别索引与准确率元数据)
"""
import argparse, json, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def build_loaders(data_dir, batch, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = datasets.ImageFolder(data_dir, transform=train_tf)
    classes = full.classes
    n_val = max(1, int(len(full) * 0.05))          # 5%做验证集(数据量大,5%足够稳健)
    # 随机划分(跨所有类别), 避免按目录序取尾部导致的类别偏斜
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(full), generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=train_tf), train_idx)
    val_set = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=val_tf), val_idx)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader, classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PlantVillage color目录(含38个类别子目录)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HeYi] device = {device}")

    train_loader, val_loader, classes = build_loaders(args.data, args.batch, args.workers)
    print(f"[HeYi] classes = {len(classes)}, train ~{len(train_loader.dataset)}, val ~{len(val_loader.dataset)}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():                    # 冻结全部
        p.requires_grad = False
    for p in model.layer4.parameters():             # 解冻layer4微调
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); running = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            running += loss.item()
            if i % 100 == 0:
                print(f"  ep{ep} step{i}/{len(train_loader)} loss={loss.item():.4f}")
        sched.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total += y.numel()
        acc = correct / max(total, 1)
        print(f"[HeYi] epoch{ep} loss={running/len(train_loader):.4f} val_acc={acc:.4f} ({time.time()-t0:.0f}s)")

        if acc > best:
            best = acc
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "val_acc": acc, "epoch": ep,
                "arch": "resnet18_transfer_layer4",
            }, os.path.join(args.out, "best.pth"))
            print(f"[HeYi] saved best.pth (val_acc={acc:.4f})")

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "best_val_acc": best, "arch": "resnet18_transfer_layer4"},
                  f, ensure_ascii=False, indent=2)
    print(f"[HeYi] done. best_val_acc={best:.4f}")

if __name__ == "__main__":
    main()
