# -*- coding: utf-8 -*-
"""
禾医·PlantVillage 全量数据集下载脚本
=====================================
用法:
    python data/download_dataset.py                       # 下载到 data/plantvillage
    python data/download_dataset.py --dest D:/data/pv     # 指定目录
    python data/download_dataset.py --subset 20           # 每类只下N张(快速试用)
说明:
  - 源: spMohanty/PlantVillage-Dataset (raw/color, 38类, ~54,306张)
  - raw.githubusercontent 不可达时自动切换 jsDelivr CDN
  - 断点续传: 已存在且>10KB的文件自动跳过, 可反复执行
"""
import argparse, json, os, subprocess, urllib.parse

API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
CDN = "https://cdn.jsdelivr.net/gh/spMohanty/PlantVillage-Dataset@master/raw/color/"

def curl(url, out=None, retries=3):
    cmd = ["curl", "-s", "-L", "--max-time", "90"]
    if out:
        cmd += ["-o", out]
    cmd.append(url)
    for _ in range(retries):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout if not out else True
    return None if not out else False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "plantvillage"))
    ap.add_argument("--subset", type=int, default=0, help="每类下载数量, 0=全量")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    classes = json.loads(curl(API))
    if not isinstance(classes, list):
        raise SystemExit("获取类别列表失败, 请检查网络")
    print(f"共 {len(classes)} 类")
    total = 0
    for ci, cls in enumerate(classes, 1):
        name = cls["name"]
        d = os.path.join(args.dest, name)
        os.makedirs(d, exist_ok=True)
        files = json.loads(curl(API + "/" + urllib.parse.quote(name)))
        if not isinstance(files, list):
            print(f"[{ci}/{len(classes)}] {name}: 列表获取失败, 跳过"); continue
        want = files if not args.subset else files[:args.subset]
        got = 0
        for f in want:
            dst = os.path.join(d, f["name"])
            if os.path.exists(dst) and os.path.getsize(dst) > 10240:
                got += 1; continue
            url = CDN + urllib.parse.quote(name) + "/" + urllib.parse.quote(f["name"])
            if curl(url, out=dst) and os.path.getsize(dst) > 10240:
                got += 1
            elif os.path.exists(dst):
                os.remove(dst)
        total += got
        print(f"[{ci}/{len(classes)}] {name}: {got}/{len(want)}")
    print(f"完成, 共 {total} 张 -> {args.dest}")

if __name__ == "__main__":
    main()
