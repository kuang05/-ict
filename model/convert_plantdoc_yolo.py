# -*- coding: utf-8 -*-
"""
PlantDoc检测训练集 parquet -> YOLO单类叶片检测数据集 + 分类微调裁叶集
====================================================================
输出:
  /root/data/yolo_leaf/images/train/*.jpg
  /root/data/yolo_leaf/labels/train/*.txt   (单类0=leaf, YOLO归一化 cx cy w h)
  /root/data/pd_crops_train/<39类名>/*.jpg  (分类微调用裁叶, 映射到禾医39类)
  /root/data/yolo_leaf/stats.json
"""
import io, json, os, sys
import pyarrow.parquet as pq
from PIL import Image

OUT_YOLO = "/root/data/yolo_leaf"
OUT_CROPS = "/root/data/pd_crops_train"
PARQUETS = [
    "/root/data/plantdoc/train-00000-of-00002.parquet",
    "/root/data/plantdoc/train-00001-of-00002.parquet",
]
# 与 eval_plantdoc.py 相同的映射(PlantDoc category -> 禾医39类)
sys.path.insert(0, "/root/he-yi/model")
from eval_plantdoc import MAP  # noqa: E402

os.makedirs(f"{OUT_YOLO}/images/train", exist_ok=True)
os.makedirs(f"{OUT_YOLO}/labels/train", exist_ok=True)

n_img = n_box = n_crop = n_unmapped = 0
for pq_path in PARQUETS:
    t = pq.read_table(pq_path)
    for row in t.to_pylist():
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        img_id = row.get("image_id", f"img{n_img}")
        fname = f"{pq_path.split('/')[-1][:5]}_{img_id}.jpg".replace("/", "_")
        objs = row.get("objects") or {}
        cats = objs.get("category") or []
        bboxes = objs.get("bbox") or []
        lines = []
        for cat, bbox in zip(cats, bboxes):
            x, y, w, h = bbox
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(W), x + w), min(float(H), y + h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            nw, nh = (x1 - x0) / W, (y1 - y0) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
            # 分类微调裁叶(带10%边距)
            gt = MAP.get(str(cat).strip().lower())
            if gt:
                mx, my = 0.10 * (x1 - x0), 0.10 * (y1 - y0)
                cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
                cx1, cy1 = min(W, int(x1 + mx)), min(H, int(y1 + my))
                crop = img.crop((cx0, cy0, cx1, cy1))
                if min(crop.size) >= 32:
                    d = os.path.join(OUT_CROPS, gt)
                    os.makedirs(d, exist_ok=True)
                    crop.save(os.path.join(d, fname.replace(".jpg", f"_{n_crop}.jpg")), quality=92)
                    n_crop += 1
            else:
                n_unmapped += 1
        if lines:
            img.save(os.path.join(f"{OUT_YOLO}/images/train", fname), quality=92)
            with open(os.path.join(f"{OUT_YOLO}/labels/train", fname.replace(".jpg", ".txt")), "w") as f:
                f.write("\n".join(lines))
            n_img += 1

stats = {"images": n_img, "boxes": n_box, "crops_for_cls": n_crop, "unmapped_boxes": n_unmapped}
json.dump(stats, open(f"{OUT_YOLO}/stats.json", "w"), indent=2)
print(json.dumps(stats))
