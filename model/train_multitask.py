# -*- coding: utf-8 -*-
"""禾医 · 多任务辅助监督（从根本重构模型的学习信号）

为什么这样做（根本性改变，不是调参）
------------------------------------
本任务的根本困难是**监督信号不足**：我们的 27 类只有 PlantDoc 6720 个田间 crop
（来自 1874 张原图），模型必然过拟合；而 452 样本的评测置信区间约 ±4.3pp，
连"是否进步"都难以判断。

但我们手上有 **111,165 张真实田间图、94 个类别**（hf_field），
此前只把它们当**无监督**语料做 SSL。本脚本把它们变成**有判别力的额外监督**：

    backbone ──┬── 病害头(27类, 我们的任务)   ← 主任务
               ├── 作物头(13类)               ← 辅助(层次)
               └── 辅助头(94类, hf_field)      ← 额外监督(本脚本新增)

    L = CE_disease + λc·CE_crop + λa·CE_aux

辅助任务强制 backbone 学会"**田间条件下如何区分病斑形态**"——这正是
PlantVillage(实验室) → PlantDoc(田间) 迁移时缺失的能力。与 SSL 的区别：
SSL 只学"什么图相似"，本方案学的是"**什么病不一样**"，判别性更强。

产出 ckpt 与 `NetV4` 兼容（病害头在 backbone.fc / head_dis，另含 crop_head），
额外的 `aux_head.*` 在 `eval_hier.py` 中会被当作 unexpected 键安全忽略。

用法
----
python model/train_multitask.py --pv-data ... --pd-root ... --extra-data ... \
  --aux-root /home/ma-user/data/hf_aux --backbone resnet101 \
  --epochs 15 --aux-sample 40000 --aux-weight 0.5 \
  --out model/checkpoints/best_mt_r101.pth
"""
import argparse, json, os, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_plantdoc import PD2PV  # noqa: E402
from train_hier import (MixStyle, tf_train, tf_val, scan_dir, CLASSES27,  # noqa: E402
                        CROPS, C2I_CROP, crop_of, HIGH_RISK,
                        amp_ctx, amp_scaler)

N_AUX_DEFAULT = 94


class PairDS(Dataset):
    """主任务样本: (path, y_disease, y_crop)。"""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, yd, yc = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), yd, yc


class AuxDS(Dataset):
    """辅助任务样本: (path, y_aux)。"""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


class NetMT(nn.Module):
    """多任务网络：backbone + 病害头(主) + 作物头 + 辅助头。

    resnet 分支把病害头放在 backbone.fc（与 NetV4/v3 键兼容）；
    timm 分支把病害头放在 head_dis。
    """

    def __init__(self, backbone="resnet101", n_cls=27, n_crop=13, n_aux=94,
                 mix_p=0.5, pretrained=True):
        super().__init__()
        self.is_timm = backbone.startswith("timm_")
        if self.is_timm:
            import timm
            self.backbone = timm.create_model(backbone[5:], pretrained=pretrained,
                                              num_classes=0)
            feat = self.backbone.num_features
            self.head_dis = nn.Linear(feat, n_cls)
        else:
            builder = {"resnet50": models.resnet50, "resnet101": models.resnet101,
                       "resnet18": models.resnet18}[backbone]
            weights = {"resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
                       "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
                       "resnet18": models.ResNet18_Weights.IMAGENET1K_V1}[backbone] \
                if pretrained else None
            self.backbone = builder(weights=weights)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feat, n_cls)
        self.crop_head = nn.Linear(feat, n_crop)
        self.aux_head = nn.Linear(feat, n_aux)
        self.mix1 = MixStyle(mix_p)
        self.mix2 = MixStyle(mix_p)

    def features(self, x):
        b = self.backbone
        if self.is_timm:
            y = b.stem(x)
            y = self.mix1(b.stages[0](y))
            y = self.mix2(b.stages[1](y))
            y = b.stages[2](y)
            y = b.stages[3](y)
            return b.forward_head(y, pre_logits=True)
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = self.mix1(b.layer1(x))
        x = self.mix2(b.layer2(x))
        x = b.layer3(x); x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        f = self.features(x)
        d = self.head_dis(f) if self.is_timm else self.backbone.fc(f)
        return d, self.crop_head(f), self.aux_head(f)


def scan_aux(root, per_class=0, seed=42):
    rng = random.Random(seed)
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    c2i = {c: i for i, c in enumerate(classes)}
    items = []
    for cls in classes:
        d = os.path.join(root, cls)
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        if per_class:
            files = files[:per_class]
        items.extend((p, c2i[cls]) for p in files)
    return items, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-data", default="/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation")
    ap.add_argument("--pd-root", default="/home/ma-user/data/pd_v3")
    ap.add_argument("--extra-data", default="/home/ma-user/data/pv_extra")
    ap.add_argument("--aux-root", default="/home/ma-user/data/hf_aux")
    ap.add_argument("--backbone", default="resnet101")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--pv-per-class", type=int, default=600)
    ap.add_argument("--pd-repeat", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-weight", type=float, default=0.3)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--aux-sample", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model/checkpoints/best_mt_r101.pth")
    ap.add_argument("--tag", default="mt")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    device = "npu" if getattr(torch, "npu", None) is not None and torch.npu.is_available() \
        else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    classes = list(CLASSES27)
    c2i = {c: i for i, c in enumerate(classes)}
    pv = scan_dir(args.pv_data, args.pv_per_class, c2i=c2i)
    if args.extra_data and os.path.isdir(args.extra_data):
        pv += scan_dir(args.extra_data, 0, c2i=c2i)
    pd_train = scan_dir(os.path.join(args.pd_root, "train"), c2i=c2i)
    pd_val = scan_dir(os.path.join(args.pd_root, "val"), c2i=c2i)
    if not pd_train:
        raise SystemExit("目标域训练集为空")

    # 主任务样本：(path, 病害标签, 作物标签)
    main_items = []
    for p, y in pv + pd_train * args.pd_repeat:
        main_items.append((p, y, C2I_CROP[crop_of(classes[y])]))
    random.shuffle(main_items)

    aux_items, aux_classes = scan_aux(args.aux_root)
    random.shuffle(aux_items)
    if args.aux_sample and args.aux_sample < len(aux_items):
        aux_items = aux_items[:args.aux_sample]

    print(f"[mt] device={device} backbone={args.backbone} 主任务={len(main_items)} "
          f"辅助={len(aux_items)}({len(aux_classes)}类) aux_w={args.aux_weight}", flush=True)

    cnt = Counter(y for _, y, _ in main_items)
    w = np.array([1.0 / np.sqrt(max(cnt[i], 1)) for i in range(len(classes))])
    w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    g = torch.Generator().manual_seed(args.seed)
    pf = 6 if args.workers > 0 else None
    main_loader = DataLoader(PairDS(main_items, tf_train(args.img_size)),
                             batch_size=args.batch, shuffle=True, num_workers=args.workers,
                             generator=g, pin_memory=True, drop_last=True,
                             persistent_workers=(args.workers > 0), prefetch_factor=pf)
    aux_loader = DataLoader(AuxDS(aux_items, tf_train(args.img_size)),
                            batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed + 1),
                            pin_memory=True, drop_last=True,
                            persistent_workers=(args.workers > 0), prefetch_factor=pf)
    val_loader = DataLoader(PairDS([(p, y, C2I_CROP[crop_of(classes[y])]) for p, y in pd_val],
                                   tf_val(args.img_size)),
                            batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                            persistent_workers=(args.workers > 0), prefetch_factor=pf)

    net = NetMT(args.backbone, len(classes), len(CROPS), len(aux_classes)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lf_dis = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    lf_crop = nn.CrossEntropyLoss(label_smoothing=0.05)
    lf_aux = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = amp_scaler(device, False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, best_ep, history = 0.0, 0, []
    for ep in range(1, args.epochs + 1):
        net.train(); t0 = time.time()
        run, nb = torch.zeros((), device=device), 0
        it_aux = iter(aux_loader)
        for x, yd, yc in main_loader:
            # ---- 主任务 step（病害 + 作物）----
            x = x.to(device, non_blocking=True)
            yd = yd.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, False):
                d, c, _ = net(x)
                loss = lf_dis(d, yd) + args.crop_weight * lf_crop(c, yc)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.detach(); nb += 1
            # ---- 辅助任务 step（交替；避免同批混任务 → NPU 动态 shape → TBE 崩溃）----
            try:
                xa, ya = next(it_aux)
            except StopIteration:
                it_aux = iter(aux_loader)
                xa, ya = next(it_aux)
            xa = xa.to(device, non_blocking=True)
            ya = ya.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, False):
                _, _, a = net(xa)
                loss_a = args.aux_weight * lf_aux(a, ya)
            scaler.scale(loss_a).backward()
            scaler.step(opt); scaler.update()
            run += loss_a.detach(); nb += 1
        sched.step()
        # 验证只用主任务病害头
        net.eval(); ok = tot = 0
        with torch.no_grad():
            for x, yd, _ in val_loader:
                x, yd = x.to(device), yd.to(device)
                with amp_ctx(device, False):
                    d, _, _ = net(x)
                ok += (d.argmax(1) == yd).sum().item(); tot += yd.size(0)
        net.train()
        acc = ok / max(tot, 1)
        history.append({"epoch": ep, "loss": round((run / max(nb, 1)).item(), 4),
                        "pd_val_dis_acc": round(acc, 4)})
        print(f"[mt] epoch{ep} loss={(run/max(nb,1)).item():.4f} PDval_dis={acc:.4f} "
              f"({time.time()-t0:.0f}s, {len(main_loader)} 主步 + {len(aux_loader)} 辅步)", flush=True)
        if acc > best:
            best, best_ep = acc, ep
            torch.save({"model": net.state_dict(), "classes": classes, "crops": CROPS,
                        "backbone": args.backbone, "mix_p": 0.5,
                        "crop_weight": args.crop_weight, "aux_weight": args.aux_weight,
                        "n_aux": len(aux_classes), "aux_classes": aux_classes,
                        "pd_val_dis_acc": acc, "epoch": ep,
                        "arch": "v4_hierarchical_multitask",
                        "protocol": "PlantDoc 27类输出空间; 原图级划分 val; 多任务辅助监督"},
                       args.out)
            print(f"[mt] saved {args.out} (PDval_dis={acc:.4f})", flush=True)

    with open(args.out.replace(".pth", "_train_record.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "args": vars(args), "best_pd_val_dis_acc": round(best, 4),
                   "best_epoch": best_ep, "history": history}, f, ensure_ascii=False, indent=2)
    print(f"[mt] done. best_PDval_dis={best:.4f} @ep{best_ep}")


if __name__ == "__main__":
    main()
