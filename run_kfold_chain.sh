#!/bin/bash
# 禾医 · k 折交叉训练链（玩法重构：让 100% 目标域数据都参与训练）
#
# 现状问题：固定 80/20 划分 → 20% 目标域数据从未参与训练，而目标域本就稀缺。
# 本链：5 折各自训练一个模型（每折用不同 80%），并集覆盖 100% 数据；
#       5 个模型天然多样化 → 集成更强；且全部在 pd_v3 上、无泄漏。
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
EX=/home/ma-user/data/pv_extra
TEST=/home/ma-user/data/pd_v3/test_clean
LOG=logs/kfold_chain.log

echo "=== KFOLD CHAIN START $(date) ===" >> $LOG

# 等第二轮（hier_plus / w2_s123）跑完，避免抢 GPU
while ! grep -qa "=== ROUND2 ALL DONE" logs/round2_chain.log 2>/dev/null; do sleep 60; done
# 等多任务辅助监督训练跑完（单卡，不能并行）
while pgrep -f "train_multitask" >/dev/null 2>&1; do sleep 60; done
echo "[kfold] 第二轮 + 多任务均完成，开始 k 折 $(date)" >> $LOG

# 生成 5 折划分（原图级）
python model/make_folds.py --k 5 --seed 42 >> $LOG 2>&1

for k in 0 1 2 3 4; do
  PD=/home/ma-user/data/pd_v3_fold$k
  [ -d "$PD/train" ] || { echo "[kfold] fold$k 缺失，跳过" >> $LOG; continue; }
  python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
    --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
    --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 --seed 42 \
    --out model/checkpoints/best_fold$k.pth --tag fold$k > logs/train_fold$k.log 2>&1
  echo "[kfold] fold$k 训练完成 $(date)" >> $LOG
  python model/eval_hier.py --ckpt model/checkpoints/best_fold$k.pth \
    --data $TEST --val $PD/val --tta 1 --sizes 192,224,288,352,416 \
    --out results/eval_fold$k.json > logs/eval_fold$k.log 2>&1
  echo "[kfold] fold$k -> $(grep -h '"top1"' results/eval_fold$k.json | head -1)" >> $LOG
done

# 5 折集成：并集覆盖 100% 目标域数据，且天然多样化
CK=""
for k in 0 1 2 3 4; do
  [ -f model/checkpoints/best_fold$k.pth ] && CK="$CK model/checkpoints/best_fold$k.pth"
done
if [ -n "$CK" ]; then
  python model/eval_ensemble.py --ckpt $CK --data $TEST \
    --tta 1 --sizes 192,224,288,352,416 \
    --out results/eval_kfold_ensemble.json > logs/eval_kfold_ens.log 2>&1
  echo "[kfold] 5折集成 -> $(grep -h '"top1"' results/eval_kfold_ensemble.json | head -1)" >> $LOG
fi

echo "=== KFOLD CHAIN ALL DONE $(date) ===" >> $LOG
