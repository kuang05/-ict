#!/bin/bash
cd /home/ma-user/he-yi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
while ! grep -qa "\[v3\] done" logs/train_v13.log 2>/dev/null; do sleep 60; done
sleep 15
python model/eval_v3.py --ckpt model/checkpoints/best_v13.pth --data /home/ma-user/data/pd_v2/test \
  --backbone resnet50 --tta 1 --sizes 224,288 --out results/eval_test_best_v13.json > logs/eval_v13.log 2>&1
python model/compute_hr.py --ckpt model/checkpoints/best_v13.pth --tta 1 \
  --data /home/ma-user/data/pd_v2/test --out results/hr_v13.json > logs/hr_v13.log 2>&1
echo V13_EVAL_DONE >> logs/eval_v13.log
