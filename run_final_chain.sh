#!/bin/bash
# 禾医 · 收尾链（对齐评奖规则的全部四个维度）
#   创新性   → 逐个评测所有候选模型 + 异构集成
#   完成度   → ONNX → ATC → OM(Ascend310B4) 端侧产物
#   华为栈   → 昇腾训练(已全程) + ATC 转换 + 端侧推理耗时
#   社会效益 → 高危守门曲线（漏诊率随拒识阈值下降），目标 <5%
cd /home/ma-user/he-yi || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_ENDPOINT=https://hf-mirror.com

PD=/home/ma-user/data/pd_v3
LOG=logs/final_chain.log

echo "=== FINAL CHAIN START $(date) ===" >> $LOG
while ! grep -qa "=== SSL CHAIN ALL DONE" logs/ssl_chain.log 2>/dev/null; do sleep 60; done
echo "[final] 上游链全部完成 $(date)" >> $LOG

# ---------- 1) 逐个评测（层次化模型用 eval_hier，自带高危守门） ----------
for tag in hier w1_convnextt w2_convnexts ssl_ft_r50; do
  ck=model/checkpoints/best_$tag.pth
  [ -f "$ck" ] || { echo "[final] 跳过 $tag (无 ckpt)" >> $LOG; continue; }
  python model/eval_hier.py --ckpt $ck --data $PD/test --val $PD/val \
    --tta 1 --sizes 224,288,352 \
    --out results/final_eval_$tag.json > logs/final_eval_$tag.log 2>&1
  echo "[final] $tag -> $(grep -h '"top1"' results/final_eval_$tag.json | head -1)" >> $LOG
done

# ---------- 2) 异构概率集成（各 ckpt 自带 backbone） ----------
CKPTS=""
for tag in hier w1_convnextt w2_convnexts ssl_ft_r50; do
  [ -f model/checkpoints/best_$tag.pth ] && CKPTS="$CKPTS model/checkpoints/best_$tag.pth"
done
if [ -n "$CKPTS" ]; then
  python model/eval_ensemble.py --ckpt $CKPTS --tta 1 --sizes 224,288,352 \
    --out results/final_ensemble.json > logs/final_ensemble.log 2>&1
  echo "[final] ensemble -> $(grep -h '"top1"' results/final_ensemble.json | head -1)" >> $LOG
fi

# ---------- 3) 选出最优单模型 → ONNX → ATC → OM ----------
BEST_CK=""; BEST_BB=""; BEST_TAG=""; BEST_ACC=0
for tag in hier w1_convnextt w2_convnexts ssl_ft_r50; do
  f=results/final_eval_$tag.json
  [ -f "$f" ] || continue
  acc=$(python -c "import json;print(json.load(open('$f')).get('top1',0))" 2>/dev/null)
  ck=model/checkpoints/best_$tag.pth
  bb=$(python -c "import torch;print(torch.load('$ck',map_location='cpu',weights_only=False).get('backbone','resnet101'))" 2>/dev/null)
  if python -c "import sys;sys.exit(0 if float('$acc')>float('$BEST_ACC') else 1)"; then
    BEST_ACC=$acc; BEST_CK=$ck; BEST_BB=$bb; BEST_TAG=$tag
  fi
done
echo "[final] 最优单模型: $BEST_TAG top1=$BEST_ACC backbone=$BEST_BB" >> $LOG

if [ -n "$BEST_CK" ]; then
  python model/export_onnx.py --ckpt $BEST_CK --backbone $BEST_BB \
    --size 224 --out-dir model/onnx_final > logs/final_export.log 2>&1
  if [ -f model/onnx_final/heyi_v3_hr.onnx ]; then
    atc --model=model/onnx_final/heyi_v3_hr.onnx --framework=5 \
      --output=model/onnx_final/${BEST_TAG}_Ascend310B4 --soc_version=Ascend310B4 \
      --input_shape="input:1,3,224,224" --log=error >> logs/final_export.log 2>&1
    echo "[final] OM 产物: $(ls -la model/onnx_final/*.om 2>/dev/null | awk '{print $9, $5}')" >> $LOG
  fi
fi

echo "=== FINAL CHAIN ALL DONE $(date) ===" >> $LOG
