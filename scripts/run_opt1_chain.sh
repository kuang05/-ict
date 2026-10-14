#!/bin/bash
# 禾医 · 全面优化批次1 —— 骨干 x 分辨率 x 正则 对照实验
# 设计原则: 单变量归因, 每组训完立即多尺度TTA评测 + 高危召回
#   A(v14) R101@288 20ep            隔离"分辨率"因素(vs v11 R101@224 67.92%)
#   B(v15) R101@288 30ep+MixUp+EMA  在A基础上叠加"长训练+正则"
#   C(v16) ConvNeXt-T@288 30ep+MixUp+EMA  换骨干
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
PD=/home/ma-user/data/pd_v2
EX=/home/ma-user/data/pv_extra
LOG=logs/opt1_chain.log

echo "=== OPT1 START $(date) ===" >> $LOG

# ---------------- A: v14  ResNet101 @288, 20ep ----------------
echo "[A] v14 train start $(date)" >> $LOG
python model/train_field_v3.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone resnet101 --img-size 288 --epochs 20 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 \
  --out model/checkpoints/best_v14.pth --tag v14 > logs/train_v14.log 2>&1
echo "[A] v14 train done $(date)" >> $LOG
python model/eval_v3.py --ckpt model/checkpoints/best_v14.pth --data $PD/test \
  --backbone resnet101 --tta 1 --sizes 224,288,352 \
  --out results/eval_test_best_v14.json > logs/eval_v14.log 2>&1
python model/compute_hr.py --ckpt model/checkpoints/best_v14.pth --backbone resnet101 --tta 1 \
  --data $PD/test --out results/hr_v14.json > logs/hr_v14.log 2>&1
echo "[A] v14 ALL DONE $(date)" >> $LOG

# ---------------- B: v15  ResNet101 @288, 30ep + MixUp0.2 + EMA0.999 ----------------
echo "[B] v15 train start $(date)" >> $LOG
python model/train_field_v3.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone resnet101 --img-size 288 --epochs 30 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 \
  --mixup 0.2 --mixup-p 0.5 --ema-decay 0.999 \
  --out model/checkpoints/best_v15.pth --tag v15 > logs/train_v15.log 2>&1
echo "[B] v15 train done $(date)" >> $LOG
python model/eval_v3.py --ckpt model/checkpoints/best_v15.pth --data $PD/test \
  --backbone resnet101 --tta 1 --sizes 224,288,352 \
  --out results/eval_test_best_v15.json > logs/eval_v15.log 2>&1
python model/compute_hr.py --ckpt model/checkpoints/best_v15.pth --backbone resnet101 --tta 1 \
  --data $PD/test --out results/hr_v15.json > logs/hr_v15.log 2>&1
echo "[B] v15 ALL DONE $(date)" >> $LOG

# ---------------- C: v16  ConvNeXt-Tiny @288, 30ep + MixUp0.2 + EMA0.999 ----------------
echo "[C] v16 train start $(date)" >> $LOG
python model/train_field_v3.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone timm_convnext_tiny --img-size 288 --epochs 30 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 \
  --mixup 0.2 --mixup-p 0.5 --ema-decay 0.999 \
  --out model/checkpoints/best_v16.pth --tag v16 > logs/train_v16.log 2>&1
echo "[C] v16 train done $(date)" >> $LOG
python model/eval_v3.py --ckpt model/checkpoints/best_v16.pth --data $PD/test \
  --backbone timm_convnext_tiny --tta 1 --sizes 224,288,352 \
  --out results/eval_test_best_v16.json > logs/eval_v16.log 2>&1
python model/compute_hr.py --ckpt model/checkpoints/best_v16.pth --backbone timm_convnext_tiny --tta 1 \
  --data $PD/test --out results/hr_v16.json > logs/hr_v16.log 2>&1
echo "[C] v16 ALL DONE $(date)" >> $LOG

echo "=== OPT1 ALL DONE $(date) ===" >> $LOG
