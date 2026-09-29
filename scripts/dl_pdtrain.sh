#!/bin/bash
export https_proxy=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
cd /root/data/plantdoc
for f in train-00000-of-00002.parquet train-00001-of-00002.parquet; do
  curl -sL --retry 5 -C - -o $f "https://hf-mirror.com/datasets/agyaatcoder/PlantDoc/resolve/main/data/$f"
  echo "done $f $(stat -c%s $f)" >> dl_train.status
done
echo ALL-DONE >> dl_train.status
