#!/bin/bash
# 禾医 · 真正的解决办法：田间域自适应自监督预训练 + 微调
#
# 逻辑链：
#   PlantVillage(实验室) → PlantDoc(田间) 的域偏移是核心困难。
#   IEEE ICETCI 2026：存在数据天花板，加大模型无效；出路是 target-domain pretraining。
#   我们手上有 111,165 张真实田间图（hf_field，非本任务类别，但 SSL 不需要标签）。
#   → 在这些田间图上做 SimCLR 领域自适应预训练(DAP)，再微调。
#
# 流程：等 world 链完成 → SSL 预训练(R50, 160px) → 用 SSL 权重微调层次化模型 → 评测
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
PD=/home/ma-user/data/pd_v3
EX=/home/ma-user/data/pv_extra
SSL=/home/ma-user/data/ssl_field
LOG=logs/ssl_chain.log

echo "=== SSL CHAIN START $(date) ===" >> $LOG

while ! grep -qa "=== WORLD CHAIN ALL DONE" logs/world_chain.log 2>/dev/null; do sleep 60; done
echo "[ssl] world 链完成, 开始田间自监督预训练 $(date)" >> $LOG

# ---- 1) 田间域自适应自监督预训练（SimCLR, ResNet50, 160px） ----
# 实测吞吐 ~9s/step(含争用)，全量 111k×20ep 需 20h+ → 采样 6 万张 + 150 分钟硬预算
python model/ssl_pretrain.py --data $SSL --backbone resnet50 \
  --sample 60000 --epochs 12 --max-minutes 150 \
  --batch 256 --size 160 --lr 1e-3 --workers 48 \
  --out model/checkpoints/ssl_r50.pth --tag ssl_r50 > logs/ssl_pretrain.log 2>&1
echo "[ssl] 预训练完成 $(date)" >> $LOG
grep -a "^\[ssl\] epoch\|saved\|预算" logs/ssl_pretrain.log | tail -4 >> $LOG

# ---- 2) 用 SSL 权重微调层次化模型（R50, 与 v3 的 R50 基线可比） ----
python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone resnet50 --init model/checkpoints/ssl_r50.pth \
  --img-size 224 --epochs 20 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 \
  --out model/checkpoints/best_ssl_ft_r50.pth --tag ssl_ft_r50 > logs/train_ssl_ft.log 2>&1
echo "[ssl] 微调完成 $(date)" >> $LOG

python model/eval_hier.py --ckpt model/checkpoints/best_ssl_ft_r50.pth \
  --data $PD/test --val $PD/val --tta 1 --sizes 224,288,352 \
  --out results/eval_ssl_ft_r50.json > logs/eval_ssl_ft.log 2>&1
echo "[ssl] 评测完成 $(date)" >> $LOG
grep -a '"top1"\|"crop_head_acc"\|"alpha"' logs/eval_ssl_ft.log >> $LOG 2>/dev/null

echo "=== SSL CHAIN ALL DONE $(date) ===" >> $LOG
