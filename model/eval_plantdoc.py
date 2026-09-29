# -*- coding: utf-8 -*-
"""
禾医·真实田间域差评测（PlantDoc 检测版 test 集）
=================================================
协议: 236张真实田间照片, 按人工标注bbox裁出452个叶子目标 -> 训练模型(ResNet18迁移,
     PlantVillage 39类)直接零样本推理 -> 与映射标签比对。
这是"实验室->田间"域差的真实数字, 论文佐证: 该差距通常为20-40个百分点
(Jeon 2025, Sci.Direct S1574954125005886; 广东农业科学 2022)。
"""
import io, json, collections
import pyarrow.parquet as pq
import torch, numpy as np
from PIL import Image
from torchvision import models, transforms

CKPT = "/root/he-yi/model/checkpoints/best.pth"
PARQUET = "/root/data/plantdoc/test.parquet"
OUT = "/root/he-yi/docs/plantdoc_eval_result.json"

# PlantDoc test 27类 -> 训练39类(Mendeley命名)
MAP = {
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

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ck["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(ck["model"]); net.eval().cuda()

    t = pq.read_table(PARQUET)
    rows = t.to_pylist()
    crops, gts, srcs = [], [], []
    skipped = 0
    for ri, row in enumerate(rows):
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = img.size
        objs = row.get("objects") or {}
        cats = objs.get("category") or []
        for (cat, bbox) in zip(cats, objs.get("bbox") or []):
            gt = MAP.get(str(cat).strip().lower())
            if gt is None or gt not in c2i:
                skipped += 1; continue
            x, y, w, h = bbox
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, int(x + w)), min(H, int(y + h))
            if x1 - x0 < 10 or y1 - y0 < 10:
                skipped += 1; continue
            crops.append(VAL_TF(img.crop((x0, y0, x1, y1))))
            gts.append(c2i[gt]); srcs.append(str(cat))
    print(f"crops={len(crops)} skipped={skipped}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = torch.stack(crops[i:i+64]).cuda()
            p = net(x).argmax(1)
            preds.extend(p.cpu().tolist())

    correct = sum(int(p == g) for p, g in zip(preds, gts))
    per_class = collections.defaultdict(lambda: [0, 0])
    top2_hits = 0
    with torch.no_grad():
        pass
    for p, g, s in zip(preds, gts, srcs):
        per_class[s][1] += 1
        per_class[s][0] += int(p == g)
    # top2 (辅助指标: 排序前2命中)
    with torch.no_grad():
        all_top2 = []
        for i in range(0, len(crops), 64):
            x = torch.stack(crops[i:i+64]).cuda()
            top2 = torch.topk(net(x), k=2, dim=1).indices.cpu()
            all_top2.extend(top2.tolist())
    top2_hits = sum(int(g in t2) for t2, g in zip(all_top2, gts))

    result = {
        "protocol": "PlantDoc test 236 real-field images -> 452 bbox leaf crops -> zero-shot inference by PlantVillage-trained ResNet18",
        "n_crops": len(gts),
        "top1_acc": round(correct / len(gts), 4),
        "top2_acc": round(top2_hits / len(gts), 4),
        "per_class": {s: {"acc": round(a / b, 4), "n": b} for s, (a, b) in sorted(per_class.items())},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    for s, v in result["per_class"].items():
        print(f"  {s:38s} {v['acc']*100:5.1f}%  n={v['n']}")
    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
