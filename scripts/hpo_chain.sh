#!/bin/bash
cd /home/ma-user/he-yi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# 等 v13 训练+评测完成(评测json出现)
while [ ! -f results/eval_test_best_v13.json ]; do sleep 60; done
echo "v13完成, HPO 启动: $(date)" >> logs/hpo.log
python scripts/hpo_optuna.py > logs/hpo.log 2>&1
echo HPO_ALL_DONE >> logs/hpo.log
