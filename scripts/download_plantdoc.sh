#!/bin/bash
export https_proxy=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
curl -sL --retry 5 -C - -o /root/data/plantdoc/PlantDoc.zip "https://hf-mirror.com/datasets/LamTNguyen/PlantDoc/resolve/main/PlantDoc.zip"
echo "EXIT=$? size=$(stat -c%s /root/data/plantdoc/PlantDoc.zip 2>/dev/null)" >> /root/data/plantdoc/dl.status
