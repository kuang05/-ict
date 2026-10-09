#!/bin/bash
# 禾医 · 世界水平冲击链 —— 现代骨干 + ImageNet-22k 预训练 + 层次化双头
# 依据（IEEE ICETCI 2026 的 UDA 系统研究）：PlantVillage→PlantDoc 存在"数据天花板",
#   单纯加大模型无用, 出路是 target-domain pretraining 与能利用结构的对齐目标。
#   本作品已用目标域监督 + MixStyle + 层次结构, 现补齐"现代骨干 + 大规模预训练"这一环。
#
# 流程：等 hier 完成 → w1(ConvNeXt-T/22k) → w2(ConvNeXt-S/22k)
#   每个：训练(224, 30ep, MixUp0.2, EMA0.999, 层次双头 λ=0.3) → eval_hier(多尺度TTA + α搜索 + 守门曲线)
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
PD=/home/ma-user/data/pd_v3
EX=/home/ma-user/data/pv_extra
LOG=logs/world_chain.log

echo "=== WORLD CHAIN START $(date) ===" >> $LOG

while ! grep -qa "=== HIER CHAIN ALL DONE" logs/hier_chain.log 2>/dev/null; do sleep 60; done
echo "[world] hier 完成, 开始现代骨干实验 $(date)" >> $LOG

# ---------------- w1: ConvNeXt-Tiny (ImageNet-22k) + 层次双头 ----------------
echo "[world] w1 ConvNeXt-T train start $(date)" >> $LOG
python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone timm_convnext_tiny.fb_in22k_ft_in1k --img-size 224 --epochs 20 \
  --batch 64 --workers 48 --lr 3e-4 --pv-per-class 600 --pd-repeat 3 \
  --crop-weight 0.3 \
  --out model/checkpoints/best_w1_convnextt.pth --tag w1_convnextt > logs/train_w1.log 2>&1
echo "[world] w1 train done $(date)" >> $LOG
python model/eval_hier.py --ckpt model/checkpoints/best_w1_convnextt.pth \
  --data $PD/test --val $PD/val --tta 1 --sizes 224,288,352 \
  --out results/eval_w1.json > logs/eval_w1.log 2>&1
echo "[world] w1 ALL DONE $(date)" >> $LOG
grep -a '"top1"\|"crop_head_acc"\|"alpha"' logs/eval_w1.log >> $LOG 2>/dev/null

# ---------------- w2: ConvNeXt-Small (ImageNet-22k) + 层次双头 ----------------
echo "[world] w2 ConvNeXt-S train start $(date)" >> $LOG
python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone timm_convnext_small.fb_in22k_ft_in1k --img-size 224 --epochs 20 \
  --batch 48 --workers 48 --lr 3e-4 --pv-per-class 600 --pd-repeat 3 \
  --crop-weight 0.3 \
  --out model/checkpoints/best_w2_convnexts.pth --tag w2_convnexts > logs/train_w2.log 2>&1
echo "[world] w2 train done $(date)" >> $LOG
python model/eval_hier.py --ckpt model/checkpoints/best_w2_convnexts.pth \
  --data $PD/test --val $PD/val --tta 1 --sizes 224,288,352 \
  --out results/eval_w2.json > logs/eval_w2.log 2>&1
echo "[world] w2 ALL DONE $(date)" >> $LOG
grep -a '"top1"\|"crop_head_acc"\|"alpha"' logs/eval_w2.log >> $LOG 2>/dev/null

echo "=== WORLD CHAIN ALL DONE $(date) ===" >> $LOG
