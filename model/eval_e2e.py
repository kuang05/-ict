# -*- coding: utf-8 -*-
"""
端到端田间评测: 整图 -> YOLO叶片检测 -> 裁叶 -> 禾医分类器 -> 与GT比对
======================================================================
协议(PlantDoc test, 236图/452个GT叶目标):
  对每个GT目标, 取与其IoU最大的YOLO检测框:
    - IoU>=0.4: 用YOLO裁叶分类 (检测命中)
    - IoU<0.4 : 记为检测漏检(严格口径下该目标判错; 另报回退口径=用GT框分类)
  分类器: best_finetuned.pth (PV+PlantDoc-train微调) 或 best.pth (基线)
输出: docs/e2e_eval_result.json
"""
import io, json, argparse, collections
import pyarrow.parquet as pq
import torch, numpy as np
from PIL import Image
from torchvision import models, transforms

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
sys.path.insert(0, "/root/he-yi/model")
from eval_plantdoc import MAP  # noqa: E402

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    u = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / u if u > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/he-yi/model/checkpoints/best_finetuned.pth")
    ap.add_argument("--yolo-weights", default="/root/data/yolo_leaf/best_leaf.pt")
    ap.add_argument("--parquet", default="/root/data/plantdoc/test.parquet")
    ap.add_argument("--out", default="/root/he-yi/docs/e2e_eval_result.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-th", type=float, default=0.4)
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(args.yolo_weights)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(ck["model"]); net.eval().cuda()

    t = pq.read_table(args.parquet)
    rows = t.to_pylist()

    strict_ok = fallback_ok = n_gt = det_miss = 0
    per_class = collections.defaultdict(lambda: [0, 0])
    crops, gts, tags = [], [], []
    for row in rows:
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = img.size
        res = det.predict(np.asarray(img), conf=args.conf, verbose=False)[0]
        det_boxes = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            det_boxes.append((x0, y0, x1, y1))
        objs = row.get("objects") or {}
        for cat, bbox in zip(objs.get("category") or [], objs.get("bbox") or []):
            gt_name = MAP.get(str(cat).strip().lower())
            if gt_name is None or gt_name not in c2i:
                continue
            x, y, w, h = bbox
            gt_box = (x, y, min(W, x + w), min(H, y + h))
            n_gt += 1
            best_iou, best_box = 0.0, None
            for db in det_boxes:
                v = iou(gt_box, db)
                if v > best_iou:
                    best_iou, best_box = v, db
            if best_box is None or best_iou < args.iou_th:
                det_miss += 1
                crop_box = gt_box           # 回退口径: 用GT框
                tag = "fallback"
            else:
                crop_box = best_box
                tag = "detected"
            x0, y0, x1, y1 = [max(0, int(v)) for v in crop_box]
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 10 or y1 - y0 < 10:
                continue
            crops.append(VAL_TF(img.crop((x0, y0, x1, y1))))
            gts.append(c2i[gt_name])
            tags.append((tag, str(cat)))

    preds = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = torch.stack(crops[i:i+64]).cuda()
            preds.extend(net(x).argmax(1).cpu().tolist())

    for (tag, cat), p, g in zip(tags, preds, gts):
        hit = int(p == g)
        strict_ok += hit
        fallback_ok += hit
        per_class[cat][1] += 1
        per_class[cat][0] += hit

    result = {
        "protocol": "PlantDoc test end-to-end: full image -> YOLO leaf detection -> crop -> classifier",
        "classifier": args.ckpt.split("/")[-1],
        "detector": args.yolo_weights.split("/")[-1],
        "n_gt_objects": n_gt,
        "detection_recall@iou%.1f" % args.iou_th: round(1 - det_miss / max(n_gt, 1), 4),
        "strict_acc(miss=wrong)": round(strict_ok / max(len(gts), 1), 4),
        "fallback_acc(gt-crop for miss)": round(fallback_ok / max(len(gts), 1), 4),
        "per_class": {s: {"acc": round(a / b, 4), "n": b} for s, (a, b) in sorted(per_class.items())},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, ensure_ascii=False, indent=2))
    for s, v in result["per_class"].items():
        print(f"  {s:38s} {v['acc']*100:5.1f}%  n={v['n']}")
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
