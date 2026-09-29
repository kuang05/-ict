#!/bin/bash
cd /root/he-yi
echo STEP1 > /root/evals.progress
pkill -9 -f "yolo detect" 2>/dev/null
sed -i "s|checkpoints/best.pth|checkpoints/best_finetuned.pth|; s|plantdoc_eval_result_robust.json|plantdoc_eval_result_finetuned.json|; s/\.cuda()/.to(\"cpu\")/g" model/eval_plantdoc.py
sed -i "s/\.cuda()/.to(\"cpu\")/g" model/eval_slidewin.py
echo STEP2 >> /root/evals.progress
/opt/venv/bin/python model/eval_plantdoc.py 2>/dev/null | tail -40 > /root/eval_ft.log
echo STEP3 >> /root/evals.progress
/opt/venv/bin/python model/eval_slidewin.py 2>/dev/null | tail -40 > /root/eval_sw.log
echo ALL-DONE >> /root/evals.progress
