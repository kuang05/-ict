#!/bin/bash
# ---------------------------------------------------------------
# 禾医 · 云服务器环境一键恢复脚本
# 适用: 实例重建/重装后, 在华为云控制台的「远程登录 / CloudShell」里执行
# 作用: 装 sshd -> 装 Python 依赖 -> 拉数据集 -> 建严格划分 -> 训练 -> 评测
# ---------------------------------------------------------------
set -e

# ===== 0. 出口代理(华为云 AMD 开发机专用, 下载权重/数据集需要) =====
export http_proxy=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
export https_proxy=$http_proxy

# ===== 1. 安装并启动 openssh-server =====
echo "[1/6] 安装 openssh-server ..."
apt-get update -y
apt-get install -y openssh-server curl git

# 允许 root 密钥登录
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# 写入本机公钥(cloud_server_ed25519)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
grep -q "cloud-server" ~/.ssh/authorized_keys 2>/dev/null || \
  echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA7pT3g+yZ0MFsIq4MN/LBJyFT/zwwEjCfO88Zv/NLBQ cloud-server' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

mkdir -p /run/sshd
if command -v systemctl >/dev/null 2>&1 && systemctl status ssh >/dev/null 2>&1; then
  systemctl restart ssh
else
  pgrep -x sshd >/dev/null || /usr/sbin/sshd
fi
echo "[1/6] sshd 监听端口:"
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep sshd || echo "  (未看到监听, 请手动 /usr/sbin/sshd)"

# ===== 2. Python 依赖 =====
echo "[2/6] 安装 Python 依赖 ..."
PY=/opt/venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
$PY -m pip install --quiet timm pyarrow pillow numpy 2>&1 | tail -2 || true

# ===== 3. 数据集 =====
echo "[3/6] 下载数据集 ..."
mkdir -p /root/data/plantdoc
cd /root/he-yi 2>/dev/null || { echo "  请先上传 he-yi 代码目录到 /root/he-yi"; exit 1; }
bash scripts/download_plantvillage.sh
bash scripts/download_plantdoc.sh
bash scripts/download_plantdoc_parquet.sh

# ===== 4. 严格划分(原图级 train/val/test) =====
echo "[4/6] 构建 PlantDoc 严格划分 ..."
cd /root/he-yi
$PY model/prepare_plantdoc.py

# ===== 5. 训练 =====
echo "[5/6] 训练 v3 (ResNet50 + MixStyle 外挂 + 27类) ..."
cd /root/he-yi
$PY model/train_field_v3.py \
    --epochs 40 --pv-per-class 800 --backbone resnet50 \
    --out model/checkpoints/best_v3b.pth --tag v3b_r50

# ===== 6. 评测 =====
echo "[6/6] 在 PlantDoc test(452 裁叶) 上评测 ..."
$PY model/eval_v5.py --ckpt model/checkpoints/best_v3b.pth \
    --backbone resnet50 --split test --tta 1 \
    --out /root/he-yi/docs/eval_test_best_v3b.json

echo "全部完成。"
