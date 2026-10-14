#!/bin/bash
# 禾医 · 第二轮：数据增补（补强高危弱类）+ 集成多样性
#
# 依据第一轮发现：
#   * Corn___Northern_Leaf_Blight 训练集仅 280 张、集成识别率仅 41.67%（高危类最差）
#     → 并入 kaggle 玉米 Blight 1146 张（数据量 ×5），定向补强
#   * 集成关键在多样性 → 训练同配置不同种子的模型
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
PD=/home/ma-user/data/pd_v3
EX=/home/ma-user/data/pv_extra
LOG=logs/round2_chain.log

echo "=== ROUND2 START $(date) ===" >> $LOG

# 1) hier_plus：含增补数据（与 hier 同配方同种子，唯一差异=数据）
python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 --seed 42 \
  --out model/checkpoints/best_hier_plus.pth --tag hier_plus > logs/train_hier_plus.log 2>&1
echo "[r2] hier_plus 训练完成 $(date)" >> $LOG

# 2) w2_s123：ConvNeXt-S 换种子（集成多样性）
python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone timm_convnext_small.fb_in22k_ft_in1k --img-size 224 --epochs 20 \
  --batch 48 --workers 48 --lr 3e-4 --pv-per-class 600 --pd-repeat 3 \
  --crop-weight 0.3 --seed 123 \
  --out model/checkpoints/best_w2_s123.pth --tag w2_s123 > logs/train_w2_s123.log 2>&1
echo "[r2] w2_s123 训练完成 $(date)" >> $LOG

# 3) 评测（统一 5 尺度 TTA）
for t in hier_plus w2_s123; do
  python model/eval_hier.py --ckpt model/checkpoints/best_$t.pth --data $PD/test --val $PD/val \
    --tta 1 --sizes 192,224,288,352,416 \
    --out results/eval_$t.json > logs/eval_$t.log 2>&1
  echo "[r2] $t -> $(grep -h '"top1"' results/eval_$t.json | head -1)" >> $LOG
done

echo "=== ROUND2 ALL DONE $(date) ===" >> $LOG
