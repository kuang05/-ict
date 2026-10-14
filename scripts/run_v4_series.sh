#!/bin/bash
cd /home/ma-user/he-yi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
P=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
D=/home/ma-user/data/pd_v2
python model/train_field_v3.py --pv-data $P --pd-root $D --epochs 20 --batch 64 --workers 32 \
  --pv-per-class 3000 --out model/checkpoints/best_v4a.pth --tag v4a > logs/train_v4a.log 2>&1
python model/eval_v3.py --ckpt model/checkpoints/best_v4a.pth --data $D/test --backbone resnet50 --tta 1 \
  --out results/eval_test_best_v4a.json > logs/eval_v4a.log 2>&1
python model/train_field_v3.py --pv-data $P --pd-root $D --epochs 20 --batch 64 --workers 32 \
  --pv-per-class 3000 --high-risk-weight 3.0 --out model/checkpoints/best_v4b.pth --tag v4b > logs/train_v4b.log 2>&1
python model/eval_v3.py --ckpt model/checkpoints/best_v4b.pth --data $D/test --backbone resnet50 --tta 1 \
  --out results/eval_test_best_v4b.json > logs/eval_v4b.log 2>&1
echo V4_ALL_DONE
