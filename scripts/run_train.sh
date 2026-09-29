#!/bin/bash
cd /root/he-yi
exec /opt/venv/bin/python model/train_pytorch.py --data "/root/data/pv/Plant_leave_diseases_dataset_without_augmentation" --epochs 8 --batch 128 --workers 16
