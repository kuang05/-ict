#!/bin/bash
cd /home/ma-user/he-yi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# 等 v5 主进程退出
while pgrep -f "tag v5 " > /dev/null; do sleep 20; done
sleep 10
# v5 评测
python model/eval_v3.py --ckpt model/checkpoints/best_v5.pth --data /home/ma-user/data/pd_v2/test \
  --backbone resnet50 --tta 1 --out results/eval_test_best_v5.json > logs/eval_v5.log 2>&1
echo V5_EVAL_DONE >> logs/train_v5.log
# v6: v5配置 + HF田间图（pv_extra 已合并两源）
python model/train_field_v3.py \
  --pv-data /home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation \
  --extra-data /home/ma-user/data/pv_extra \
  --pd-root /home/ma-user/data/pd_v2 \
  --epochs 20 --batch 64 --workers 32 \
  --out model/checkpoints/best_v6.pth --tag v6 > logs/train_v6.log 2>&1
python model/eval_v3.py --ckpt model/checkpoints/best_v6.pth --data /home/ma-user/data/pd_v2/test \
  --backbone resnet50 --tta 1 --out results/eval_test_best_v6.json > logs/eval_v6.log 2>&1
echo V6_ALL_DONE >> logs/train_v6.log
