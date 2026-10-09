# -*- coding: utf-8 -*-
"""禾医 · 田间域自监督预训练语料提取

把 hf_field 的 111k 张 parquet 田间图**降采样到 192px** 后落盘，
供领域自适应自监督预训练（DAP）使用。

为什么要降采样：
* 原图 17GB，磁盘只剩 34GB，直接解压会占满；
* SSL 预训练本来就在 160~192px 做，降采样**不损失任何可用信息**；
* 192px JPEG 约 25KB/张 → 111k 张仅约 3GB。

用法: python model/extract_ssl_corpus.py
"""
import glob, io, os, sys
from multiprocessing import Pool

import pyarrow.parquet as pq
from PIL import Image

SRC = "/home/ma-user/data/hf_field"
OUT = "/home/ma-user/data/ssl_field"
SIZE = 192
QUALITY = 88


def _init():
    global _N
    _N = None


def save_one(args):
    idx, data = args
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((SIZE, SIZE))
        im.save(os.path.join(OUT, f"f{idx:07d}.jpg"), quality=QUALITY)
        return 1
    except Exception:
        return 0


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.parquet")))
    files = [f for f in files if os.path.basename(f) != ".parquet"]
    print(f"[extract] {len(files)} 个 parquet → {OUT} @{SIZE}px", flush=True)

    idx, total = 0, 0
    pool = Pool(32)
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1000, columns=["image"]):
            imgs = batch.column("image").to_pylist()
            args = []
            for im in imgs:
                data = im["bytes"] if isinstance(im, dict) else im
                args.append((idx, data))
                idx += 1
            total += sum(pool.map(save_one, args, chunksize=64))
        print(f"[extract] {os.path.basename(f)} 完成, 累计 {total}/{idx}", flush=True)
    pool.close()
    pool.join()
    print(f"[extract] DONE 共保存 {total} 张（跳过 {idx - total} 张坏图）", flush=True)


if __name__ == "__main__":
    main()
