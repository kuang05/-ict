# -*- coding: utf-8 -*-
"""禾医 · 为 hf_field 的 111k 田间图重建标签索引（供多任务辅助学习）

背景
----
`extract_ssl_corpus.py` 把 hf_field 的 111,165 张图按**固定顺序**落盘为
`ssl_field/f{idx:07d}.jpg`（idx 从 0 递增，按 parquet 文件名排序 + 行序）。
因此无需重新解码图片，只要按**同样的顺序**回读 `label_name` 列，
就能把标签一一对上。

产出
----
`~/data/hf_aux/<sanitized_label>/<idx>.jpg`  —— 硬链接，不占额外磁盘。
并输出 `~/data/hf_aux/_index.json` 记录 类别→数量 与 总数，便于核对。

用法: python model/build_aux_index.py
"""
import glob, json, os
from collections import Counter

import pyarrow.parquet as pq

SRC = "/home/ma-user/data/hf_field"
IMG = "/home/ma-user/data/ssl_field"
OUT = "/home/ma-user/data/hf_aux"


def sanitize(name):
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.parquet")))
    files = [f for f in files if os.path.basename(f) != ".parquet"]
    labels = []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000, columns=["label_name"]):
            labels.extend(batch.column("label_name").to_pylist())
        print(f"[aux] {os.path.basename(f)} 累计标签 {len(labels)}", flush=True)

    imgs = sorted(glob.glob(os.path.join(IMG, "f*.jpg")))
    print(f"[aux] 图片 {len(imgs)} 张, 标签 {len(labels)} 条")
    if len(imgs) != len(labels):
        raise SystemExit(f"数量不一致（图 {len(imgs)} vs 标签 {len(labels)}），顺序假设不成立，终止。")

    os.makedirs(OUT, exist_ok=True)
    cnt = Counter()
    for p, lab in zip(imgs, labels):
        d = os.path.join(OUT, sanitize(lab))
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.basename(p))
        if not os.path.exists(dst):
            try:
                os.link(p, dst)
            except OSError:
                pass
        cnt[lab] += 1
    meta = {"n_classes": len(cnt), "n_images": sum(cnt.values()),
            "classes": dict(sorted(cnt.items(), key=lambda kv: -kv[1]))}
    with open(os.path.join(OUT, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[aux] DONE 类别 {len(cnt)}, 图片 {sum(cnt.values())} -> {OUT}")


if __name__ == "__main__":
    main()
