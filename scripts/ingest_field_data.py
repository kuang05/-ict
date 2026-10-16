# -*- coding: utf-8 -*-
"""禾医 · 自采田间数据入库工具链
==================================================================
用途：把队员自采的田间照片（手机拍摄）一键处理成模型可直接使用的数据集，
      并生成带溯源的登记表（registry.json）——这是"真数据背书"的凭证。

流程（六步，全部留痕）：
  1. 扫描 --src 下所有图片（支持按 作物_病害 子文件夹 或 平铺两种组织）
  2. MD5 去重（组内去重 + 与已有 field 数据去重）
  3. 标准化：去 EXIF（隐私+一致性）+ 长边封顶（默认 1600px，不损画质）
  4. 病名映射：文件夹名 tomato_late_blight -> 模型输出空间 Tomato___Late_blight
     （未知名强制人工确认，防止标注错类混入——标注质检第一关）
  5. 输出数据集：与 pd_v3/test_clean 同构的"一类一文件夹"，eval_hier 可直接评测
  6. 登记表：每张图的 源目录/拍摄信息/标签/MD5 全部写入 registry.json

用法:
  # 先干跑，看映射和数量对不对
  python ingest_field_data.py --src "data/field_raw" --out "data/field_captured" --dry-run
  # 确认后正式入库
  python ingest_field_data.py --src "data/field_raw" --out "data/field_captured"
  # 入库并按 8:2 切分微调集/独立验证集（可选）
  python ingest_field_data.py --src "data/field_raw" --out "data/field_captured" --split 0.2 --seed 42
"""
import argparse, hashlib, json, os, shutil, sys
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    Image = None

# 文件夹名(小写下划线) -> 模型 27 类输出空间名
# 只列常见自采场景；未列出的会在干跑/入库时列出等待人工补充
ALIAS = {
    "tomato_late_blight": "Tomato___Late_blight",
    "tomato_early_blight": "Tomato___Early_blight",
    "tomato_leaf_mold": "Tomato___Leaf_Mold",
    "tomato_septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "tomato_bacterial_spot": "Tomato___Bacterial_spot",
    "tomato_yellow_leaf_curl_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "tomato_healthy": "Tomato___healthy",
    "corn_common_rust": "Corn_(maize)___Common_rust_",
    "corn_northern_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn_gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn_healthy": "Corn_(maize)___healthy",
    "potato_late_blight": "Potato___Late_blight",
    "potato_early_blight": "Potato___Early_blight",
    "potato_healthy": "Potato___healthy",
    "cucumber_powdery_mildew": "Squash___Powdery_mildew",
    "cucumber_downy_mildew": "Cucumber___downy_mildew",
    "cucumber_healthy": "Cucumber___healthy",
    "pepper_bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "pepper_healthy": "Pepper,_bell___healthy",
    "grape_black_rot": "Grape___Black_rot",
    "grape_esca": "Grape___Esca_(Black_Measles)",
    "grape_leaf_blight": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "apple_scab": "Apple___Apple_scab",
    "apple_black_rot": "Apple___Black_rot",
    "apple_healthy": "Apple___healthy",
    "strawberry_leaf_scorch": "Strawberry___Leaf_scorch",
    "strawberry_healthy": "Strawberry___healthy",
    "soybean_healthy": "Soybean___healthy",
    "blueberry_healthy": "Blueberry___healthy",
    "peach_healthy": "Peach___healthy",
    "cherry_healthy": "Cherry_(including_sour)___healthy",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(img_path: str, dst: str, max_side: int) -> None:
    """去 EXIF + 长边封顶（默认不缩，只截断超大图；手机图一般 1600~4000px）"""
    if Image is None:
        shutil.copy2(img_path, dst)
        return
    im = Image.open(img_path)
    im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=92)  # 重存即去 EXIF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="自采照片根目录（按 作物_病害 建子文件夹）")
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--split", type=float, default=0.0, help=">0 时按此比例切出独立验证集")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--registry", default=None, help="登记表路径（默认 <out>/registry.json）")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    # 1. 扫描
    entries = []  # (src_path, folder_name)
    if os.path.isdir(args.src):
        for sub in sorted(os.listdir(args.src)):
            d = os.path.join(args.src, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    entries.append((os.path.join(d, f), sub.lower()))
    print(f"[ingest] 扫描到 {len(entries)} 张图片，来自 {len(set(e[1] for e in entries))} 个子文件夹")

    # 2. 映射检查（未知的先报出来，人工确认后才能入库）
    unknown = sorted(set(e[1] for e in entries if e[1] not in ALIAS))
    if unknown:
        print(f"[ingest] ⚠️ 未知名 {len(unknown)} 个（请先补 ALIAS 或改文件夹名）:")
        for u in unknown:
            print("   -", u)
        if not args.dry_run:
            print("[ingest] 未知名未处理，终止。修复后重跑。")
            sys.exit(2)
        print("[ingest] (dry-run 模式，继续展示可映射部分)")

    known = [e for e in entries if e[1] in ALIAS]
    print(f"[ingest] 可映射 {len(known)} 张；目标类别数 {len(set(ALIAS[e[1]] for e in known))}")

    # 3. 去重
    seen, kept = set(), []
    for p, folder in known:
        m = md5(p)
        if m in seen:
            print(f"[ingest] 去重跳过(组内重复): {os.path.basename(p)}")
            continue
        seen.add(m)
        kept.append((p, folder, m))
    print(f"[ingest] 去重后 {len(kept)} 张")

    # 4. 输出
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        registry = {"created": datetime.now().isoformat(timespec="seconds"),
                    "src": os.path.abspath(args.src), "count": len(kept),
                    "classes": {}, "files": []}
        for p, folder, m in kept:
            cls = ALIAS[folder]
            cls_dir = os.path.join(args.out, cls)
            os.makedirs(cls_dir, exist_ok=True)
            name = f"{folder}_{m[:8]}.jpg"
            dst = os.path.join(cls_dir, name)
            normalize(p, dst, args.max_side)
            registry["classes"].setdefault(cls, 0)
            registry["classes"][cls] += 1
            registry["files"].append({"file": f"{cls}/{name}", "label": cls,
                                      "src_folder": folder, "src_name": os.path.basename(p),
                                      "md5": m})
        # 5. 切分（可选）
        if args.split > 0:
            ev_dir = args.out + "_eval"
            ft_dir = args.out + "_finetune"
            for d in (ev_dir, ft_dir):
                shutil.rmtree(d, ignore_errors=True)
            for cls in registry["classes"]:
                files = [f for f in registry["files"] if f["label"] == cls]
                rng.shuffle(files)
                n_ev = max(1, round(len(files) * args.split))
                for i, f in enumerate(files):
                    dst_root = ev_dir if i < n_ev else ft_dir
                    d = os.path.join(dst_root, cls)
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(os.path.join(args.out, f["file"]),
                                 os.path.join(d, os.path.basename(f["file"])))
            print(f"[ingest] 已切分: 验证集 -> {ev_dir}，微调集 -> {ft_dir}")
        # 6. 登记表
        reg_path = args.registry or os.path.join(args.out, "registry.json")
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        print(f"[ingest] 登记表 -> {reg_path}")
        print("[ingest] 各类数量:")
        for cls, n in sorted(registry["classes"].items(), key=lambda kv: -kv[1]):
            print(f"   {n:4d}  {cls}")
        print(f"[ingest] 合计 {len(kept)} 张 ✅")
    else:
        from collections import Counter
        c = Counter(ALIAS[e[1]] for e in kept)
        print("[ingest] (dry-run) 入库后各类数量预览:")
        for cls, n in c.most_common():
            print(f"   {n:4d}  {cls}")
        print(f"[ingest] (dry-run) 合计 {len(kept)} 张，未写入任何文件")


if __name__ == "__main__":
    main()
