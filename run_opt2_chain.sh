#!/bin/bash
# 禾医 · 全面优化批次2 —— 在批次1最优模型上做增强
# 用法: bash run_opt2_chain.sh <BEST_CKPT> <BACKBONE> <TAG>
#   例: bash run_opt2_chain.sh model/checkpoints/best_v15.pth resnet101 v15
# 内容:
#   2a 高危代价敏感微调(直击 high-risk recall@1, 从 BEST 继续而非从 ImageNet 重来)
#   2b 异构概率集成(BEST + v11 + v10 + v3, 各 ckpt 自带骨干)
#   2c 权重汤(仅同骨干时执行)
set -u
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

BEST=${1:?需要最优 ckpt 路径}
BB=${2:?需要 backbone}
TAG=${3:?需要 tag}
PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
PD=/home/ma-user/data/pd_v2
EX=/home/ma-user/data/pv_extra
LOG=logs/opt2_chain.log

echo "=== OPT2 START $(date) BEST=$BEST BB=$BB TAG=$TAG ===" >> $LOG

# ---------------- 2a: 高危代价敏感微调 ----------------
echo "[2a] hr-finetune start $(date)" >> $LOG
python model/train_field_v3.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone $BB --img-size 288 --epochs 8 --batch 64 --workers 48 \
  --lr 1e-4 --pv-per-class 600 --pd-repeat 3 --high-risk-weight 2.5 \
  --init $BEST \
  --out model/checkpoints/best_hr_$TAG.pth --tag hr_$TAG > logs/train_hr_$TAG.log 2>&1
python model/eval_v3.py --ckpt model/checkpoints/best_hr_$TAG.pth --data $PD/test \
  --backbone $BB --tta 1 --sizes 224,288,352 \
  --out results/eval_test_hr_$TAG.json > logs/eval_hr_$TAG.log 2>&1
python model/compute_hr.py --ckpt model/checkpoints/best_hr_$TAG.pth --backbone $BB --tta 1 \
  --data $PD/test --out results/hr_hr_$TAG.json > logs/hrhr_$TAG.log 2>&1
echo "[2a] done $(date)" >> $LOG

# ---------------- 2b: 异构概率集成 ----------------
echo "[2b] ensemble start $(date)" >> $LOG
python model/eval_ensemble.py \
  --ckpt $BEST model/checkpoints/best_v11.pth model/checkpoints/best_v10.pth model/checkpoints/best_v3.pth \
  --tta 1 --sizes 224,288,352 \
  --out results/eval_ens_opt2_$TAG.json > logs/ens_$TAG.log 2>&1
echo "[2b] done $(date)" >> $LOG

# ---------------- 2c: 权重汤(仅同骨干) ----------------
if [ "$BB" = "resnet101" ] && [ -f model/checkpoints/best_v11.pth ]; then
  echo "[2c] soup start $(date)" >> $LOG
  python model_soup.py $BEST model/checkpoints/best_v11.pth model/checkpoints/soup_$TAG.pth > logs/soup_$TAG.log 2>&1
  python model/eval_v3.py --ckpt model/checkpoints/soup_$TAG.pth --data $PD/test \
    --backbone $BB --tta 1 --sizes 224,288,352 \
    --out results/eval_soup_$TAG.json > logs/evalsoup_$TAG.log 2>&1
  echo "[2c] done $(date)" >> $LOG
else
  echo "[2c] skipped (BB=$BB 非 resnet101 或缺少 v11)" >> $LOG
fi

echo "=== OPT2 ALL DONE $(date) ===" >> $LOG
