#!/bin/bash
# 禾医 · fold0 严格 A/B 实验：基线配方 vs 高危加权（--high-risk-weight 2.5）
# 目的：高危5类召回@1 仅 58.72% 是农事决策最大瓶颈；代价敏感在旧协议(v4b=3.0)试过，
#       但从未与 kfold 协议结合。同划分同种子，唯一变量是 high_risk_weight。
# 产出：results/eval_fold0.json（基线） vs results/eval_fold0_hr.json（加权）直接对比。
#
# 前置：scripts/rebuild_pd_v3.py 已重建 pd_v3；model/make_folds.py 已生成 pd_v3_fold0-4。
cd /home/ma-user/work/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com   # timm 权重必须走镜像（HF 直连被墙）

PY=/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python
PV=/home/ma-user/work/data/pv/Plant_leave_diseases_dataset_without_augmentation
EX=/home/ma-user/work/data/pv_extra
FOLD0=/home/ma-user/work/data/pd_v3_fold0
TEST=/home/ma-user/work/data/pd_v3/test_clean
LOG=logs/ab_hrfold_chain.log

echo "=== AB-HRFOLD CHAIN START $(date) ===" >> $LOG

# ---- 1. fold0 基线（与原 kfold fold0 完全相同的配方，作对照锚点）----
$PY model/train_hier.py --pv-data $PV --pd-root $FOLD0 --extra-data $EX \
  --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 --seed 42 \
  --out model/checkpoints/best_fold0.pth --tag fold0 > logs/train_fold0.log 2>&1
echo "[ab] fold0 基线训练完成 $(date)" >> $LOG
$PY model/eval_hier.py --ckpt model/checkpoints/best_fold0.pth \
  --data $TEST --val $FOLD0/val --tta 1 --sizes 192,224,288,352,416 \
  --out results/eval_fold0.json > logs/eval_fold0.log 2>&1
echo "[ab] fold0 -> $(grep -h '\"top1\"' results/eval_fold0.json | head -1)" >> $LOG

# ---- 2. fold0 高危加权（唯一变量：high-risk-weight 1.0 -> 2.5）----
$PY model/train_hier.py --pv-data $PV --pd-root $FOLD0 --extra-data $EX \
  --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 --seed 42 \
  --high-risk-weight 2.5 \
  --out model/checkpoints/best_fold0_hr.pth --tag fold0_hr > logs/train_fold0_hr.log 2>&1
echo "[ab] fold0_hr 训练完成 $(date)" >> $LOG
$PY model/eval_hier.py --ckpt model/checkpoints/best_fold0_hr.pth \
  --data $TEST --val $FOLD0/val --tta 1 --sizes 192,224,288,352,416 \
  --out results/eval_fold0_hr.json > logs/eval_fold0_hr.log 2>&1
echo "[ab] fold0_hr -> $(grep -h '\"top1\"' results/eval_fold0_hr.json | head -1)" >> $LOG

echo "=== AB-HRFOLD ALL DONE $(date) ===" >> $LOG
