#!/bin/bash
export https_proxy=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
curl -sL --retry 8 --retry-delay 3 -C - -o /root/data/pv.zip "https://www.modelscope.cn/api/v1/datasets/OmniData/PlantVillage/repo?Revision=master&FilePath=raw/Plant_leaf_diseases_dataset_without_augmentation.zip"
echo "CURL-EXIT=$? size=$(stat -c%s /root/data/pv.zip 2>/dev/null)" >> /root/data/dl.status
