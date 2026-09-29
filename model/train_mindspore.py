# -*- coding: utf-8 -*-
"""
禾医·PlantVillage 病虫害识别训练脚本（MindSpore 版 —— 大赛主用，体现昇思技术栈）
================================================================================
数据: PlantVillage 38类, 目录结构 raw/color/{类别名}/*.JPG
策略: ImageNet预训练ResNet50(昇思模型仓 mindvision)迁移学习 + 余弦退火 + 标签平滑
      训练产物可导出 AIR/ONNX 供 ModelArts 或 昇腾310 部署
用法:
    pip install mindspore==2.3.1 mindvision
    python train_mindspore.py --data /path/to/plantvillage/color --epochs 8 --batch 64
输出: model/checkpoints_ms/best.ckpt + meta.json
昇腾环境(可选):
    在装有CANN的昇腾服务器上, 设置 export ASCEND_HOME_PATH=... 后运行,
    MindSpore 将自动使用 NPU；训练后执行 `ms_model_export` 说明可转OM模型上310推理。
"""
import argparse, json, math, os, time

import mindspore as ms
from mindspore import nn, ops, Tensor, Model
from mindspore.common.initializer import HeUniform

def build_dataset(data_dir, batch, image_size=224, train_ratio=0.95, seed=42):
    """ImageFolderDataset 直接按子目录名作为标签；训练/验证切分。"""
    from mindspore import dataset as ds
    mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
    full = ds.ImageFolderDataset(data_dir, class_indexing=None, decode=True)
    class_names = full.get_class_indexing()          # dict{name: idx}
    total = full.get_dataset_size()

    # 基本增广: 随机裁剪/翻转/色彩抖动(通过RandomColorAdjust)
    train_tf = ms.dataset.vision.transforms.Compose([
        ms.dataset.vision.transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        ms.dataset.vision.transforms.RandomHorizontalFlip(),
        ms.dataset.vision.transforms.RandomColorAdjust(brightness=0.15, contrast=0.15, saturation=0.15),
        ms.dataset.vision.transforms.Normalize(mean=mean, std=std),
        ms.dataset.vision.transforms.HWC2CHW(),
    ])
    val_tf = ms.dataset.vision.transforms.Compose([
        ms.dataset.vision.transforms.Resize(int(image_size * 256 / 224)),
        ms.dataset.vision.transforms.CenterCrop(image_size),
        ms.dataset.vision.transforms.Normalize(mean=mean, std=std),
        ms.dataset.vision.transforms.HWC2CHW(),
    ])

    train_ds = full.train_split([train_ratio, 1 - train_ratio], randomize=True)  # MindSpore>=2.2
    train_set, val_set = train_ds[0], train_ds[1]
    train_set = train_set.map(operations=train_tf, input_columns="image")
    val_set = val_set.map(operations=val_tf, input_columns="image")
    train_set = train_set.batch(batch, drop_remainder=True).repeat(1)
    val_set = val_set.batch(batch)
    return train_set, val_set, [k for k, v in sorted(class_names.items(), key=lambda x: x[1])]

def create_model(num_classes):
    """基于 mindvision 的 ResNet50 预训练权重做迁移学习。"""
    from mindvision.classification.models import resnet50
    net = resnet50(pretrained=True, num_classes=1000)
    # 替换分类头, 其余保持ImageNet权重(小数据集上全冻结或小学习率微调皆可)
    net.head = nn.Dense(net.head.in_channels, num_classes,
                        weight_init=HeUniform(), has_bias=True)
    return net

def forward_with_loss(net, loss_fn):
    """封装前向+损失, 供Model训练。"""
    class WithLoss(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net = net
            self.loss = loss_fn
        def construct(self, images, labels):
            out = self.net(images)
            return self.loss(out, labels)
    return WithLoss()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints_ms"))
    args = ap.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend" if os.environ.get("ASCEND_HOME_PATH") else "GPU")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    os.makedirs(args.out, exist_ok=True)
    train_set, val_set, classes = build_dataset(args.data, args.batch)
    num_classes = len(classes)
    print(f"[HeYi-MS] device={ms.get_context('device_target')} classes={num_classes}")

    net = create_model(num_classes)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps_per_epoch = max(1, math.ceil(train_set.get_dataset_size()))
    lr = nn.cosine_decay_lr(args.lr, 0.0, args.epochs * steps_per_epoch, steps_per_epoch, args.epochs)
    opt = nn.AdamWeightDecay([p for p in net.trainable_params()],
                             learning_rate=lr, weight_decay=1e-4)

    model = Model(forward_with_loss(net, loss_fn),
                  eval_network=None, optimizer=opt)   # 简化: 每epoch手动评估
    eval_model = Model(net, loss_fn=loss_fn, metrics={"acc": nn.Accuracy()})

    best = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(train_set, epochs=1, callbacks=[ms.train.TimeMonitor()])
        res = eval_model.eval(val_set, dataset_sink_mode=False)
        acc = float(res["acc"])
        print(f"[HeYi-MS] epoch{ep} val_acc={acc:.4f} ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc
            ms.save_checkpoint(net, os.path.join(args.out, "best.ckpt"))
            print(f"[HeYi-MS] saved best.ckpt (val_acc={acc:.4f})")

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "best_val_acc": best,
                   "arch": "resnet50_transfer_ms", "framework": f"mindspore-{ms.__version__}"},
                  f, ensure_ascii=False, indent=2)
    print(f"[HeYi-MS] done. best_val_acc={best:.4f}")

if __name__ == "__main__":
    main()
