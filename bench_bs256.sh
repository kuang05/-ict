#!/bin/bash
cd /home/ma-user/he-yi
# 等 v4 系列结束(v4b评测json出现即完成)
while [ ! -f results/eval_test_best_v4b.json ]; do sleep 30; done
sleep 10
source /usr/local/Ascend/ascend-toolkit/set_env.sh
P=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
D=/home/ma-user/data/pd_v2
python model/train_field_v3.py --pv-data $P --pd-root $D --epochs 2 --batch 256 --workers 48 \
  --lr 1.2e-3 --pv-per-class 3000 --out model/checkpoints/bench_bs256.pth \
  --tag v4_bs256_bench > logs/train_bs256_bench.log 2>&1
echo BS256_BENCH_DONE >> logs/train_bs256_bench.log
