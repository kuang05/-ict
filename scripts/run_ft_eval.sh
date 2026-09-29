#!/bin/bash
/opt/venv/bin/python /root/he-yi/model/eval_plantdoc.py 2>/dev/null | tail -40 > /root/eval_ft.log
echo FT-DONE >> /root/evals.progress
