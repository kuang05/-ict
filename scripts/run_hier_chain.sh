#!/bin/bash
# 禾医 · 层次化重构链（等 v15 完成后自动接续，跳过 v16 换骨干实验）
#   1) 等 v15 训练+评测完成
#   2) v4-hier 训练: R101@224 30ep + MixUp0.2 + EMA0.999 + 双头(作物辅助, λ=0.3)
#      —— 与 v15 训练配方完全相同, 唯一差异是双头架构 → 架构贡献可干净归因
#   3) 层次融合评测 + 高危守门曲线
#   4) ONNX -> ATC -> OM(Ascend310B4) 端侧产物
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PV=/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation
PD=/home/ma-user/data/pd_v3
EX=/home/ma-user/data/pv_extra
LOG=logs/hier_chain.log

echo "=== HIER CHAIN START $(date) ===" >> $LOG

# 1) 等 v15 训练完成（原 run_opt1b_chain.sh 已被终止, v15 评测由本链接手）
while ! grep -qa "\[v3\] done" logs/train_v15.log 2>/dev/null; do sleep 60; done
sleep 15
echo "[hier] 检测到 v15 训练完成 $(date), 先补跑 v15 评测" >> $LOG
python model/eval_v3.py --ckpt model/checkpoints/best_v15.pth --data $PD/test \
  --backbone resnet101 --tta 1 --sizes 224,288,352 \
  --out results/eval_test_best_v15.json > logs/eval_v15.log 2>&1
python model/compute_hr.py --ckpt model/checkpoints/best_v15.pth --backbone resnet101 --tta 1 \
  --data $PD/test --out results/hr_v15.json > logs/hr_v15.log 2>&1
echo "[hier] v15 评测完成 $(date)" >> $LOG
grep -a '"top1"\|"crop_acc"\|"disease_acc' logs/eval_v15.log >> $LOG 2>/dev/null

# 2) v4-hier 训练（配方与 v11 完全一致, 唯一差异 = 双头；EMA/MixUp 已证伪见下）
python model/train_hier.py --pv-data $PV --pd-root $PD --extra-data $EX \
  --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
  --lr 3e-4 --pv-per-class 600 --pd-repeat 3 --crop-weight 0.3 \
  --out model/checkpoints/best_hier.pth --tag hier > logs/train_hier.log 2>&1
echo "[hier] 训练完成 $(date)" >> $LOG

# 3) 层次融合评测（α 在 val 上选）+ 高危守门曲线
python model/eval_hier.py --ckpt model/checkpoints/best_hier.pth \
  --data $PD/test --val $PD/val --tta 1 --sizes 224,288,352 \
  --out results/eval_hier.json > logs/eval_hier.log 2>&1
echo "[hier] 评测完成 $(date)" >> $LOG
grep -a '"top1"\|"crop_head_acc"\|"alpha"\|high_risk_recall@1' logs/eval_hier.log >> $LOG 2>/dev/null

# 4) ONNX + ATC -> OM（端侧 310B）
python model/export_onnx.py --ckpt model/checkpoints/best_hier.pth --backbone resnet101 \
  --size 224 --out-dir model/onnx_hier > logs/export_hier.log 2>&1
if [ -f model/onnx_hier/heyi_v3_hr.onnx ]; then
  atc --model=model/onnx_hier/heyi_v3_hr.onnx --framework=5 \
    --output=model/onnx_hier/best_hier_Ascend310B4 --soc_version=Ascend310B4 \
    --input_shape="input:1,3,224,224" --log=error >> logs/export_hier.log 2>&1
  echo "[hier] OM 导出完成 $(date)" >> $LOG
else
  echo "[hier] ONNX 导出失败, 跳过 ATC" >> $LOG
fi

echo "=== HIER CHAIN ALL DONE $(date) ===" >> $LOG
