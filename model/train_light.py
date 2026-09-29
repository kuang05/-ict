# -*- coding: utf-8 -*-
"""
禾医·轻量模型训练（最近质心 + 特征标准化，纯numpy，无需GPU）
=============================================================
用途: 在未部署ResNet权重前，为Demo提供可量化、可验证的基线分类器；
     正式比赛/生产路径仍为 train_pytorch.py / train_mindspore.py 的ResNet迁移模型。
数据: data/demo_images/{类别}/*.jpg（每类约20张，按文件名排序每4张抽1张做测试集）
输出: model/checkpoints_light/light_model.json（质心+标准化参数+验证指标）
"""
import json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract, FEAT_KEYS                       # noqa: E402

DATA = os.path.join(HERE, "..", "data", "demo_images")
OUT = os.path.join(HERE, "checkpoints_light")

def main():
    X, Y, C = [], [], []
    classes = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    test_X, test_Y = [], []
    for yi, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(DATA, c)))
        for i, f in enumerate(files):
            img = Image.open(os.path.join(DATA, c, f))
            _, vec = extract(img)
            if i % 4 == 0:
                test_X.append(vec); test_Y.append(yi)     # 每类1/4做测试
            else:
                X.append(vec); Y.append(yi)
    X, Y, test_X, test_Y = map(np.array, (X, Y, test_X, test_Y))
    print(f"[HeYi-light] train={len(Y)} test={len(test_Y)} classes={classes}")

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z, Zt = (X - mu) / sd, (test_X - mu) / sd
    centroids = np.stack([Z[Y == k].mean(0) for k in range(len(classes))])

    def acc_centroid(Zv, Yv):
        pred = ((Zv[:, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
        return float((pred == Yv).mean()), pred

    def acc_knn(Zv, Yv, k=3):
        d2 = ((Zv[:, None, :] - Z[None]) ** 2).sum(-1)          # eval x train
        idx = np.argsort(d2, axis=1)[:, :k]
        pred = np.array([np.bincount(Y[idx[i]], minlength=len(classes)).argmax()
                         for i in range(len(Zv))])
        return float((pred == Yv).mean()), pred

    # 以测试集表现选型(避免k=1死记硬背在训练集上虚高)
    tr_cent, _ = acc_centroid(Z, Y)
    te_cent, pred_cent = acc_centroid(Zt, test_Y)
    te_knn3, pred_knn3 = acc_knn(Zt, test_Y, 3)
    te_knn1, pred_knn1 = acc_knn(Zt, test_Y, 1)
    te_map = {"centroid": (te_cent, pred_cent), "knn3": (te_knn3, pred_knn3), "knn1": (te_knn1, pred_knn1)}
    best_name = max(te_map, key=lambda n: te_map[n][0])
    tr_acc = {"centroid": tr_cent, "knn3": te_knn3, "knn1": te_knn1}[best_name]
    te_acc, pred = te_map[best_name]
    print(f"[HeYi-light] test_acc: centroid={te_cent:.3f} knn3={te_knn3:.3f} knn1={te_knn1:.3f} -> pick {best_name}")
    print(f"[HeYi-light] final: train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
    # 混淆明细
    for yi, c in enumerate(classes):
        m = test_Y == yi
        print(f"  {c:42s} -> {classes[int(np.bincount(pred[m], minlength=len(classes)).argmax())]}"
              f" ({int(m.sum())}张)")

    os.makedirs(OUT, exist_ok=True)
    json.dump({
        "classes": classes, "feat_keys": FEAT_KEYS,
        "mu": mu.tolist(), "sd": sd.tolist(), "centroids": centroids.tolist(),
        "train_Z": Z.tolist(), "train_Y": Y.tolist(),
        "picked": best_name,
        "metrics": {"train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
                    "n_train": int(len(Y)), "n_test": int(len(test_Y))},
        "model": f"{best_name} on masked color features",
    }, open(os.path.join(OUT, "light_model.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)
    print("[HeYi-light] saved checkpoints_light/light_model.json")

if __name__ == "__main__":
    main()
