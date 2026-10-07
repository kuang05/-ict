# 禾医 HeYi —— 农作物病虫害智能诊断与防控决策平台

> 第十一届华为ICT大赛创新赛 · 赛题二（基于华为云 AgentArts 智能体开发平台的 Agent 设计和应用）
> 拍照 → 诊断 → 开方 → 留痕：端-云-边三级推理 + 防幻觉农事智能体
> ℹ️ 本仓库为禾医的早期本地版；**赛题二完整参赛交付见 `new/` 目录**（AgentArts 平台构建/编排/部署 + 本地对照实现）。

## 30秒上手（本地演示）

```bash
pip install -r requirements.txt
python model/train_light.py        # 训练Demo基线模型(首次运行, 约1分钟)
python -m uvicorn backend.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000
# 演示照片在 data/demo_images/ 六个类别目录中各取一张
```

## 目录结构

```
he-yi/
├── backend/main.py            # FastAPI: /api/diagnose /api/chat /api/records
├── frontend/index.html        # 单页演示前端
├── model/
│   ├── train_mindspore.py     # 主训练栈: 昇思MindSpore ResNet50迁移(昇腾NPU可直跑)
│   ├── train_pytorch.py       # 对照: PyTorch ResNet18迁移
│   ├── train_light.py         # Demo基线: 掩膜色度特征+质心/kNN(纯numpy, 含测试集评估)
│   ├── infer.py               # 四级自适应推理引擎(torch/ms/light/heuristic)
│   ├── features.py            # 背景掩膜+色度学特征
│   └── labels.json            # PlantVillage 38类标签
├── agent/
│   ├── agent.py               # 农事智能体: 4个Tool+编排(诊断报告/RAG问答/高危升级)
│   ├── knowledge_base.py      # 知识库检索
│   └── llm_client.py          # OpenAI兼容客户端(华为云MaaS即插即用, 无Key自动降级)
├── data/
│   ├── knowledge/diseases.json  # 38类植保知识库(6类含处方级方案)
│   ├── demo_images/             # 6类演示与训练子集(约108张, jsDelivr下载)
│   └── download_dataset.py      # 全量54k数据集下载(断点续传)
└── docs/
    ├── 01-作品设计方案.md       # 参赛提交主文档
    ├── 02-华为云部署指南.md
    ├── 03-答辩PPT大纲.md
    └── 04-演示视频脚本.md
```

## 接入华为云大模型（可选，一行环境变量）

```bash
export HEYI_LLM_BASE_URL=https://api.modelarts-maas.com/v1
export HEYI_LLM_API_KEY=<MaaS控制台创建的Key>
export HEYI_LLM_MODEL=DeepSeek-V3
```
未配置时智能体自动降级为知识库模板应答，功能不受影响。

## 训练正式模型（已完成，真实指标）

```bash
python data/download_dataset.py            # 全量数据集(推荐ModelScope源, 827MB国内CDN秒级)
# PyTorch路径(已在AMD W7900/ROCm 7.2.4上实测)
python model/train_pytorch.py --data data/plantvillage --epochs 8 --batch 128 --workers 16
# MindSpore路径(昇思主栈; 昇腾环境设ASCEND_HOME_PATH即用NPU)
python model/train_mindspore.py --data data/plantvillage --epochs 8
```

**已交付的训练结果**（权重随仓库: `model/checkpoints/best.pth` + `meta.json`）：

| 指标 | 数值 |
|---|---|
| 验证准确率 | **99.31%**（随机分层5%验证集, 2,772张, 覆盖39类, seed=42可复现） |
| 训练曲线 | ep1 98.92% → ep6 99.31%(best) → ep8 99.24% |
| 硬件/耗时 | AMD W7900 (ROCm 7.2.4 + PyTorch 2.10), batch 128, 每轮27~31s, 8轮共约4.5分钟 |
| 端到端回归 | 平台全链路6类演示集108张 **100%** 诊断正确 |

## 演示要点（评委视角）

1. 晚疫病样本 → 秒级诊断 + **红色升级卡**（暴发预警→联系植保站）
2. 健康叶片对照 → 无误报
3. 农事问答智能体 → RAG引用知识库作答、标注来源
4. 报告中每个药名/剂量可溯源（防幻觉设计是答辩核心叙事）

> 演示照片请用 `data/demo_images/演示样本清单.json` 中已验证识别正确的样本（每类一张）。
