# -*- coding: utf-8 -*-
"""禾医 · pd_v3 重建脚本（服务器家目录被容器重置后的恢复工具）
==================================================================
背景：2026-10-15 ModelArts 笔记本容器重建，~/data 易失层被清。
      pd_v2 幸存于持久卷 ~/work/data。本脚本从 pd_v2 重建清洗版 pd_v3。

逻辑（与 docs/18 记载一致，MD5 级自校验）：
  1. 硬链接复制 pd_v2 -> pd_v3（不占额外空间）
  2. 计算 test 全体文件 MD5 索引；train/val 中凡是 MD5 命中 test 的文件即泄漏样本，删除
     （预期：train∩test 3 张 -> i1055_c0/c1 + i2048_c0；val∩test 5 张 -> i1269_c0~c4）
  3. 复验 train∩test = val∩test = 0
  4. test_clean = test 的硬链接副本（评测专用目录）

用法: python rebuild_pd_v3.py [--src ~/work/data/pd_v2] [--out ~/work/data/pd_v3]
"""
import argparse, hashlib, os, shutil


def md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_md5(root: str) -> dict:
    out = {}
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            out[md5(os.path.join(d, f))] = f"{cls}/{f}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/ma-user/work/data/pd_v2")
    ap.add_argument("--out", default="/home/ma-user/work/data/pd_v3")
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(args.src, args.out, copy_function=os.link)
    print(f"[pd_v3] 硬链接复制完成: {args.src} -> {args.out}")

    test_idx = index_md5(os.path.join(args.out, "test"))
    print(f"[pd_v3] test 索引 {len(test_idx)} 个文件")

    removed = []
    for part in ("train", "val"):
        base = os.path.join(args.out, part)
        for cls in sorted(os.listdir(base)):
            d = os.path.join(base, cls)
            for f in sorted(os.listdir(d)):
                m = md5(os.path.join(d, f))
                if m in test_idx:
                    os.remove(os.path.join(d, f))
                    removed.append(f"{part}/{cls}/{f}  <=>  test/{test_idx[m]}")
    print(f"[pd_v3] 删除泄漏样本 {len(removed)} 个（预期 8）:")
    for r in removed:
        print("   -", r)

    for part in ("train", "val"):
        left = index_md5(os.path.join(args.out, part))
        leak = set(left) & set(test_idx)
        print(f"[pd_v3] 复验 {part}∩test = {len(leak)}")
        assert not leak, f"{part} 仍有泄漏!"

    td = os.path.join(args.out, "test_clean")
    if os.path.exists(td):
        shutil.rmtree(td)
    shutil.copytree(os.path.join(args.out, "test"), td, copy_function=os.link)
    # 协议说明（2026-10-17 核实）：test_clean = 452 张（与官方 test 一致）。
    # 8 张泄漏样本是**从 train/val 剔除**的（train 6723→6720、val 1719→1714），
    # test 保持不变——剔除后 test 中的对应 8 张已无训练侧重复，属干净样本。
    # （早期文档误写"444 叶片"，是把 train/val 剔除的 8 张错算到 test 上，已修正。）
    print(f"[pd_v3] test_clean = {len(index_md5(td))} 张（协议：452）-> {td}")
    print("[pd_v3] 重建完成 ✅")


if __name__ == "__main__":
    main()
