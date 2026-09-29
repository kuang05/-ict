# 禾医 HeYi —— 农作物病虫害智能诊断与防控决策平台

> 第十一届华为ICT大赛创新赛 · 赛题1（基于华为云产品打造AI创新应用）
> 拍照 → 诊断 → 开方 → 留痕：端-云-边三级推理 + 防幻觉农事智能体

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

## 训练正式模型（98%+指标路径）

```bash
python data/download_dataset.py            # 全量54,306张(约800MB, 支持断点续传)
# MindSpore路径(推荐, 体现昇思栈; 昇腾环境设ASCEND_HOME_PATH即用NPU)
python model/train_mindspore.py --data data/plantvillage --epochs 8
# 或 PyTorch路径
python model/train_pytorch.py --data data/plantvillage --epochs 8
# 训练产物放入 model/checkpoints(_ms)/ 后, /api/health 的 engine 自动切换
```

## 演示要点（评委视角）

1. 晚疫病样本 → 秒级诊断 + **红色升级卡**（暴发预警→联系植保站）
2. 健康叶片对照 → 无误报
3. 农事问答智能体 → RAG引用知识库作答、标注来源
4. 报告中每个药名/剂量可溯源（防幻觉设计是答辩核心叙事）

> 演示照片请用 `data/demo_images/演示样本清单.json` 中已验证识别正确的样本（每类一张）。
