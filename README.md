# 禾医 HeYi —— 农作物病虫害智能诊断与防控决策平台

> 第十一届华为ICT大赛创新赛 · **拍照 → 诊断 → 开方 → 留痕**
> 端-云-边三级推理 + 防幻觉农事智能体 · 全程**昇腾 910B** 训练

---

## 一、这个项目解决的真问题：**实验室 99% ≠ 田间能用**

PlantVillage 上训出的模型在**实验室均匀背景**下能到 99%，
但拿到**真实田间**（杂乱背景、多变光照、遮挡、多叶片）上，**掉到 28.3%**。

**这就是本项目的主战场**：把跨域泛化从 **28.3% 提到 69.82%**（+41.5pp，**2.47×**）。

> **口径声明（诚实优先）**：对外主口径为 **69.82%** —— **原图级 5 折交叉集成**
> （每折用 80% 数据训练、**用留出集选模型**，5 折并集覆盖 **100% 训练数据**），
> 在**无泄漏**干净测试集（444 叶片）上测得；单折 **66.67% ± 1.05%**（n=5）。
> 对照：8 模型先验集成 69.59%；含测试集信息可达 70.05%（仅作技术上限，不作对外口径）。
> 详见 `docs/20` 第二节。

| 方法 / 设定 | PlantDoc 准确率 | 来源 |
|---|---|---|
| PlantDoc 原论文自训 | 31% | arXiv:1911.10317 |
| 纯迁移 VGG16（PV→PlantDoc） | 19.73% | IEEE ICETCI 2026 |
| 纯迁移 ViT-Base | 48% | IEEE ICETCI 2026 |
| UDA（无目标域标签，严格 inductive） | 28.42% | IEEE ICETCI 2026 |
| 少样本域适应（PV_100→PlantDoc） | 54.24% | *Sensors* 2026, 26(12):3632 |
| **禾医（5 折交叉集成，对外主口径）** | **69.82%** | 本项目 |

> 评测协议：PlantDoc **原图级留出 test = 444 个 bbox 裁叶**（已剔除 8 张与训练集同内容的泄漏样本），
> 27 类输出空间，多尺度 TTA × 水平翻转。数据用 `pd_v3`（无泄漏清洗版）。

---

## 二、对齐评奖四个维度

| 维度 | 交付 | 量化 |
|---|---|---|
| **创新性** | 跨域泛化 **28.3% → 69.82%**（5 折交叉集成，单折 66.67% ± 1.05%）；层次化双头；田间自监督预训练；代价敏感高危守门 | 对比少样本域适应 54.24%、UDA 28.42% |
| **完成度** | 端到端可跑：FastAPI 后端 + PWA 前端 + 农事智能体；**Ascend310B4 OM 产物** | `heyi_v11_Ascend310B4.om`（86MB） |
| **华为技术栈** | 全程 **昇腾 910B** 训练（R50/R101/ConvNeXt 系列）+ **ATC** 转换 | 训练日志 + OM 产物 |
| **社会效益** | **高危漏诊（判成健康）11.4% → 5.50%；守门后 → 0.00%** | 见下方守门曲线 |

**高危守门曲线**（置信度 < τ 转人工复核）：

| τ | 拒识率 | 高危**判成健康**率 |
|---|---|---|
| 0（不守门） | 0% | 5.50% |
| 0.3 | 11.5% | 3.33% |
| **0.4** | **21.8%** | **0.00%** ✅ |

→ 守门到 τ=0.4，**最危险的一类错误（把病叶判成健康）被完全消除**，78.2% 样本自动出结果。

---

## 三、30 秒上手（本地演示）

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

后端启动时会打印实际加载的引擎，例如：
```
[HeYi] 引擎=torch <- checkpoints/best_v11.pth (27 类)
```
推理引擎**按精度优先**自动选择：`best_v11.pth`（最优单模型 67.92%）→ `best.pth` → MindSpore 昇腾产物 → 轻量兜底。
类表从 ckpt 自带的 `classes` 读取，**不会出现模型与标签不匹配**。

---

## 四、技术路线（每一步都有实测依据）

### 1. 数据：把"能用的"全部用上

| 数据 | 规模 | 用途 |
|---|---|---|
| PlantVillage | 54k | 源域（实验室） |
| PlantDoc | 6,720 crop | **目标域**（田间，原图级划分） |
| pv_extra | 23,934 | 外部田间图（Tomato/Corn 等 13 类） |
| hf_field | **111,165 张 / 94 类** | 田间域预训练 + **多任务辅助监督** |
| kaggle corn Blight | 1,146 | 补强高危弱类 `Corn___Northern_Leaf_Blight` |

**泄漏排查**（MD5 级）：外部数据与 test 重叠 **0**；PlantDoc 官方 split 自带的 8 张原图重叠已剔除 → `pd_v3`。

### 2. 模型：层次化双头 + 域泛化

- **层次化双头**：病害头(27) + 作物头(13)，`L = CE_dis + 0.3·CE_crop`
  （作物识别是硬上限，显式建模它能让病害判别更聚焦）
- **MixStyle**（外挂式）在 layer1/layer2 做特征层风格混合，提升域泛化
- **现代骨干**：ConvNeXt-T/S（ImageNet-22k 预训练）

### 3. 预训练：田间域自适应自监督（DAP）

源域是实验室、目标域是田间 → 从 ImageNet 权重出发，在 **11 万张田间图**上做 SimCLR 对比学习，
让骨干先"见过田间"再微调。对应 IEEE ICETCI 2026 指出的出路：*"存在数据天花板，加大模型无效，
出路是 **target-domain pretraining**"*。

### 4. 训练信号：多任务辅助监督（根本性重构）

hf_field 的 94 类**自有标签**此前完全没用。现加一个辅助头：

```
backbone ──┬── 病害头(27类)   ← 主任务
           ├── 作物头(13类)   ← 层次辅助
           └── 辅助头(94类)   ← hf_field 自有标签
L = CE_dis + 0.3·CE_crop + 0.5·CE_aux
```

与 SSL 的区别：SSL 只学"什么图相似"，多任务学的是"**什么病不一样**"，判别性更强。

### 5. 推理：9 模型异构集成 + 多尺度 TTA

集成成员刻意覆盖不同骨干/年代/训练范式：
`R50 + R50@288 + R101 + R101@288 + R101双头 + ConvNeXt-T/22k + ConvNeXt-S/22k ×2种子 + R50+SSL-DAP`

**两条正交增益**：① 模型集成（多样性）② 多尺度 TTA（推理端）。合计 67.92% → **70.05%**。

---

## 五、已证伪的路线（避免后人重复踩坑）

| 尝试 | 结果 | 结论 |
|---|---|---|
| 分辨率 224→288 | 66.81% < 67.92% | 无益 |
| 更长训练（30ep） | 61.28% | 有害 |
| MixUp + EMA | 61.28% / 64.38% | **有害**（EMA 的 buffers 直接 copy → 权重与 BN 统计量不匹配） |
| 30 类扩展 + pd_repeat=1 | 64.38% | 有害 |
| 权重汤 soup | 62.83% | **无效**（不同脚本/不同头训练 → 权重不在同一损失盆地） |
| 同族模型堆数量（13 模型） | 68.81% < 9 模型 70.05% | **集成关键是多样性，不是数量** |

**方法学发现**：PDval 最高的模型（w2, 0.7211）test 反而偏中游 → **val-test 分布错配**；
且 444 样本的 95% 置信区间约 ±4.4pp，单模型间 2-3pp 差异**统计上不可区分** ——
这解释了为什么单个"改进"看起来都无效，而**集成是唯一稳定增益**。

---

## 六、目录结构

```
he-yi/
├── backend/main.py              # FastAPI: /api/diagnose /api/chat /api/records
├── frontend/                    # 单页演示前端 + PWA(mobile.html, sw.js)
├── agent/                       # 农事智能体: 4 Tool + 编排 + RAG 知识库 + 防幻觉
├── model/
│   ├── train_hier.py            # 主训练器: 层次化双头 + MixStyle（昇腾 NPU）
│   ├── train_field_v3.py        # 单头基线训练器
│   ├── train_multitask.py       # 多任务辅助监督（27+13+94 三头）
│   ├── ssl_pretrain.py          # 田间域自监督预训练 (SimCLR DAP)
│   ├── extract_ssl_corpus.py    # 11 万田间图 → 192px 语料
│   ├── build_aux_index.py       # 回读 parquet 重建 94 类标签索引
│   ├── eval_hier.py             # 层次化评测（含 α 融合搜索 + 高危守门曲线）
│   ├── eval_ensemble.py         # 异构概率集成评测
│   ├── export_onnx.py           # 导出 ONNX（层次化/单头均支持）
│   ├── selftest_pipeline.py     # 合成微数据端到端自检（已实抓 2 个真 bug）
│   ├── infer.py                 # 四级自适应推理引擎（精度优先）
│   └── smoke_infer.py           # 演示链路冒烟测试
├── docs/                        # 设计方案 / 审计报告 / 结果汇总 / 部署指南
└── results/                     # 全部评测结果 JSON（可追溯）
```

---

## 七、可复现命令

```bash
# 端到端自检（几秒，改动代码后先跑这个）
python model/selftest_pipeline.py

# 层次化训练（昇腾 910B）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com     # timm 权重走镜像
python model/train_hier.py --pv-data <PV> --pd-root <pd_v3> --extra-data <pv_extra> \
  --backbone resnet101 --img-size 224 --epochs 20 --batch 64 --workers 48 \
  --crop-weight 0.3 --out model/checkpoints/best_hier.pth

# 评测 + 高危守门曲线
python model/eval_hier.py --ckpt model/checkpoints/best_hier.pth \
  --data <pd_v3>/test_clean --val <pd_v3>/val --tta 1 --sizes 192,224,288,352,416 \
  --out results/eval_hier.json

# 异构集成
python model/eval_ensemble.py --ckpt <ckpt1> <ckpt2> ... \
  --data <pd_v3>/test_clean --tta 1 --sizes 192,224,288,352,416 \
  --out results/ensemble.json

# 导出端侧模型
python model/export_onnx.py --ckpt model/checkpoints/best_v11.pth --backbone resnet101 --size 224
atc --model=model/onnx_v11/heyi_v3_hr.onnx --framework=5 \
  --output=heyi_v11_Ascend310B4 --soc_version=Ascend310B4 --input_shape="input:1,3,224,224"
```

---

## 八、演示要点（评委视角）

1. 晚疫病样本 → 秒级诊断 + **红色升级卡**（暴发预警 → 联系植保站）
2. 健康叶片对照 → 无误报
3. 农事问答智能体 → RAG 引用知识库作答、标注来源
4. 报告中每个药名/剂量可溯源（**防幻觉设计是答辩核心叙事**）
5. **置信度守门**：低置信样本自动转人工，高危漏诊率可压到 0
