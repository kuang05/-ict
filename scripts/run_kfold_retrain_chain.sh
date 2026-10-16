#!/bin/bash
# 禾医 · fold1-4 重训链（恢复 5 折交叉集成主口径的完整证据链）
# 背景：2026-10-15 实例容器重建，~/he-yi 全部丢失（fold0-4 权重 + 评测 JSON）。
#       fold0 已于 10-16 重训完毕（eval_fold0.json, top1=0.6903）。
#       本链重训 fold1-4 并做 5 折集成评测，产出：
#       results/eval_fold{1..4}.json + results/eval_kfold_ensemble.json
# 纪律：单卡串行，一次只跑一个训练。
cd /home/ma-user/work/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com   # timm 权重必须走镜像（HF 直连被墙）

PY=/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python
PV=/home/ma-user/work/data/pv/Plant_leave_diseases_dataset_without_augmentation
EX=/home/ma-user/work/data/pv_extra
TEST=/home/ma-user/work/data/pd_v3/test_clean
LOG=logs/kfold_retrain_chain.log

echo "=== KFOLD-RETRAIN CHAIN START $(date) ===" >> $LOG

for k in 1 2 3 4; do
  FOLD=/home/ma-user/work/data/pd_v3_fold$k
  $PY model/train_hier.py --pv-data $PV --pd-root $FOLD --extra-data $EX \
    --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
    --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 --seed 42 \
    --out model/checkpoints/best_fold$k.pth --tag fold$k > logs/train_fold$k.log 2>&1
  echo "[retrain] fold$k 训练完成 $(date)" >> $LOG
  $PY model/eval_hier.py --ckpt model/checkpoints/best_fold$k.pth \
    --data $TEST --val $FOLD/val --tta 1 --sizes 192,224,288,352,416 \
    --out results/eval_fold$k.json > logs/eval_fold$k.log 2>&1
  echo "[retrain] fold$k -> $(grep -h '\"top1\"' results/eval_fold$k.json | head -1)" >> $LOG
done

# 5 折集成（fold0 为新重训版，与 fold1-4 同协议）
CK=""
for k in 0 1 2 3 4; do
  [ -f model/checkpoints/best_fold$k.pth ] && CK="$CK model/checkpoints/best_fold$k.pth"
done
$PY model/eval_ensemble.py --ckpt $CK --data $TEST \
  --tta 1 --sizes 192,224,288,352,416 \
  --out results/eval_kfold_ensemble.json > logs/eval_kfold_ens.log 2>&1
echo "[retrain] 5折集成 -> $(grep -h '\"top1\"' results/eval_kfold_ensemble.json | head -1)" >> $LOG

echo "=== KFOLD-RETRAIN ALL DONE $(date) ===" >> $LOG
