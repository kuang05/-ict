# -*- coding: utf-8 -*-
"""禾医 · k 折原图级划分

为什么要做这个（玩法重构）
--------------------------
1. 现状：`train_hier.py` 固定用 pd_v3/train 训练、pd_v3/val 选模型 —— **20% 目标域数据从未参与训练**。
   而目标域数据本就稀缺（6723 crop / 1874 原图），这是纯浪费。
2. k 折后：每个模型用不同的 80% 训练，**k 个模型的并集覆盖 100% 数据**；
   同时 k 个模型天然多样化 → 集成质量更高。
3. 划分严格**按原图**（同一张原图的全部 crop 必落同一侧），杜绝泄漏。
   文件名格式 `i{原图行号}_c{crop序号}.jpg`，据此分组。

产出：`pd_v3_fold0..{k-1}/{train,val}`（硬链接，几乎不占磁盘）；test 仍统一用 `pd_v3/test_clean`。

用法: python model/make_folds.py --k 5
"""
import argparse, glob, os, random, shutil
from collections import defaultdict


def img_id(path):
    """i1055_c0.jpg -> i1055"""
    base = os.path.basename(path)
    return base.split("_c")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ma-user/data/pd_v3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 合并 train + val（两者都来自 PlantDoc train parquet，原图互斥）
    src = {}
    for part in ("train", "val"):
        for p in glob.glob(os.path.join(args.root, part, "**", "*.jpg"), recursive=True):
            src.setdefault(img_id(p), []).append((part, p))
    ids = sorted(src)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    print(f"[fold] 原图总数 {len(ids)}，crop 总数 {sum(len(v) for v in src.values())}")

    folds = [ids[i::args.k] for i in range(args.k)]
    for k in range(args.k):
        dst = os.path.join(os.path.dirname(args.root), f"pd_v3_fold{k}")
        shutil.rmtree(dst, ignore_errors=True)
        val_ids = set(folds[k])
        n_tr = n_va = 0
        for iid, items in src.items():
            part = "val" if iid in val_ids else "train"
            for orig_part, p in items:
                # ⚠️ 必须相对**原始 split 目录**取相对路径，否则会拼成 dst/train/train/...
                # （2026-10-09 修复：原实现用 relpath(p, root) 保留了 "train/" 前缀，
                #   导致 fold 目录下出现嵌套 train/train，scan_dir 找不到类别 → 训练集为空）
                rel = os.path.relpath(p, os.path.join(args.root, orig_part))
                d = os.path.join(dst, part, os.path.dirname(rel))
                os.makedirs(d, exist_ok=True)
                os.link(p, os.path.join(dst, part, rel))
                if part == "train":
                    n_tr += 1
                else:
                    n_va += 1
        print(f"[fold] fold{k}: train={n_tr} val={n_va} -> {dst}")


if __name__ == "__main__":
    main()
