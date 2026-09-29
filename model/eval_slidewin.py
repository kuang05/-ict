# -*- coding: utf-8 -*-
"""
路线B·滑窗端到端评测(无需检测器): 整图多尺度滑窗 -> 分类器逐窗推理 -> 与GT目标按IoU配对
==========================================================================
动机: ultralytics+ROCm7.2 曾把系统拖挂, 用纯torchvision栈绕开。
协议: PlantDoc test 236图; 对每图做 224/320/448 三尺度、步长半窗的滑窗,
      每窗分类取(类,置信度,框); 对每个GT目标取IoU最大的窗, 比较类别。
      窗口无目标度(无objectness), 命中率依赖窗口覆盖; 这是"无检测器下界"。
"""
import io, json, argparse, collections
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import models, transforms
import sys
sys.path.insert(0, "/root/he-yi/model")
from eval_plantdoc import MAP  # noqa: E402

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0

def sliding_windows(W, H, sizes=(224, 320, 448)):
    wins = []
    for S in sizes:
        if W <= S and H <= S:
            wins.append((0, 0, W, H)); continue
        step = S // 2
        ys = list(range(0, max(1, H - S + 1), step)) or [0]
        xs = list(range(0, max(1, W - S + 1), step)) or [0]
        if ys[-1] + S < H: ys.append(max(0, H - S))
        if xs[-1] + S < W: xs.append(max(0, W - S))
        for y in ys:
            for x in xs:
                wins.append((x, y, min(W, x + S), min(H, y + S)))
    return wins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/he-yi/model/checkpoints/best_finetuned.pth")
    ap.add_argument("--parquet", default="/root/data/plantdoc/test.parquet")
    ap.add_argument("--out", default="/root/he-yi/docs/e2e_slidewin_result.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]; c2i = {c: i for i, c in enumerate(classes)}
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(ck["model"]); net.eval().cuda()

    t = pq.read_table(args.parquet)
    strict_ok = n_gt = 0
    per_class = collections.defaultdict(lambda: [0, 0])
    miss = 0

    with torch.no_grad():
        for ri, row in enumerate(t.to_pylist()):
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            W, H = img.size
            wins = sliding_windows(W, H)
            x = torch.stack([VAL_TF(img.crop(w)) for w in wins]).cuda()
            conf, pred = torch.softmax(net(x), dim=1).max(1)
            # 每个GT目标: 取IoU最大的窗
            objs = row.get("objects") or {}
            for cat, bbox in zip(objs.get("category") or [], objs.get("bbox") or []):
                gt_name = MAP.get(str(cat).strip().lower())
                if gt_name is None or gt_name not in c2i:
                    continue
                gx, gy, gw, gh = bbox
                gbox = (gx, gy, min(W, gx + gw), min(H, gy + gh))
                best_iou, best_k = 0.0, -1
                for k, w in enumerate(wins):
                    v = iou(gbox, w)
                    if v > best_iou:
                        best_iou, best_k = v, k
                n_gt += 1
                per_class[str(cat)][1] += 1
                if best_k < 0:
                    miss += 1; continue
                hit = int(int(pred[best_k]) == c2i[gt_name])
                strict_ok += hit
                per_class[str(cat)][0] += hit
            if ri % 40 == 0:
                print(f"  img {ri}/236, gt_so_far={n_gt}", flush=True)

    result = {
        "protocol": "PlantDoc test sliding-window (no detector), classifier=best_finetuned",
        "classifier": args.ckpt.split("/")[-1],
        "n_gt_objects": n_gt,
        "acc(gt-object vs best-iou-window)": round(strict_ok / max(n_gt, 1), 4),
        "per_class": {s: {"acc": round(a / b, 4), "n": b} for s, (a, b) in sorted(per_class.items())},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
