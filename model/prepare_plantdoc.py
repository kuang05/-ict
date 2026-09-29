# -*- coding: utf-8 -*-
"""
PlantDoc 严格评测协议数据集构建
================================
解决的问题
----------
1. 旧流程只有 test(236张) 用于评测, 没有留出验证集, 模型选择只能看 PlantVillage
   同域 val —— 与"田间泛化"目标脱节, 选出的模型在目标域未必最优。
2. 旧流程按 bbox 随机裁叶, 同一张原图的多个叶片可能同时落入 train/val,
   造成信息泄漏, 指标虚高。
3. 模型输出空间是 PlantVillage 39 类, 而 PlantDoc 只有 27 个有效类,
   预测到无关作物(如玉米/橙子)直接判错 —— 输出空间不匹配。

本协议
------
* 以**原图为单位**划分: train parquet 按 image 行序 80/20 划分 train/val,
  同一张图的全部 bbox 必然落在同一侧, 杜绝泄漏。
* test parquet 严格留出, 只在最终报告时评估一次。
* 输出 27 类目录结构, 可直接喂 torchvision ImageFolder。
* 同时产出 class_map.json 记录 PlantDoc 27 类 -> PlantVillage 类名的映射,
  供训练/评测共享, 保证输出空间一致。

用法
----
python model/prepare_plantdoc.py --parquet-dir /root/data/plantdoc --out /root/data/pd_v2
"""
import argparse, io, json, os, random
from collections import defaultdict

import pyarrow.parquet as pq
from PIL import Image

# PlantDoc 27 类 -> PlantVillage(Mendeley 命名) 类名
# 这是全项目唯一的标签映射真源, 训练/评测/知识库共用
PD2PV = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple leaf": "Apple___healthy",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell_pepper leaf": "Pepper,_bell___healthy",
    "bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry___healthy",
    "corn gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn___Northern_Leaf_Blight",
    "corn rust leaf": "Corn___Common_rust",
    "peach leaf": "Peach___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def iter_rows(parquet_dir, split):
    """按 split 迭代 parquet 行, 返回 (row_index, row)"""
    if split == "test":
        files = [os.path.join(parquet_dir, "test.parquet")]
    else:
        files = [
            os.path.join(parquet_dir, "train-00000-of-00002.parquet"),
            os.path.join(parquet_dir, "train-00001-of-00002.parquet"),
        ]
        files = [f for f in files if os.path.exists(f)]
    idx = 0
    for fp in files:
        if not os.path.exists(fp):
            print(f"[warn] missing {fp}")
            continue
        for row in pq.read_table(fp).to_pylist():
            yield idx, row
            idx += 1


def extract_crops(row):
    """从一行中解出 [(PIL.Image crop, plant_doc_label), ...]"""
    img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size
    objs = row.get("objects") or {}
    cats = objs.get("category") or []
    boxes = objs.get("bbox") or []
    out = []
    for cat, bbox in zip(cats, boxes):
        label = str(cat).strip().lower()
        if label not in PD2PV:
            continue
        x, y, w, h = bbox
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        out.append((img.crop((x0, y0, x1, y1)), label))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="/root/data/plantdoc")
    ap.add_argument("--out", default="/root/data/pd_v2")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-pixels", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = defaultdict(lambda: defaultdict(int))
    n_images = defaultdict(int)

    for split in ("train", "test"):
        rows = list(iter_rows(args.parquet_dir, split))
        # 图级划分: 先给每张图(行)定归属, 再把它所有 crop 写入同一侧
        n_val = int(len(rows) * args.val_ratio) if split == "train" else 0
        order = list(range(len(rows)))
        rng.shuffle(order)
        val_set = set(order[:n_val])

        for row_idx, row in rows:
            crops = extract_crops(row)
            if not crops:
                continue
            if split == "test":
                part = "test"
            else:
                part = "val" if row_idx in val_set else "train"
            n_images[part] += 1
            for cidx, (crop, label) in enumerate(crops):
                cls_dir = os.path.join(args.out, part, PD2PV[label])
                os.makedirs(cls_dir, exist_ok=True)
                crop.save(os.path.join(cls_dir, f"i{row_idx}_c{cidx}.jpg"), quality=92)
                stats[part][label] += 1

    meta = {
        "protocol": "PlantDoc 原图级划分: train parquet 80/20 -> train/val; test parquet 严格留出",
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_classes": len(PD2PV),
        "pd2pv": PD2PV,
        "counts": {p: dict(sorted(v.items())) for p, v in stats.items()},
        "n_images": dict(n_images),
        "n_crops": {p: sum(v.values()) for p, v in stats.items()},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: meta[k] for k in
                      ("n_classes", "n_images", "n_crops")}, indent=2))
    for part in ("train", "val", "test"):
        print(f"[{part}] classes={len(stats[part])}")
    print("saved ->", os.path.join(args.out, "class_map.json"))


if __name__ == "__main__":
    main()
