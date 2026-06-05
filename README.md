# MAGMA

MAGMA 是面向 OpenClaw 多 Agent 系统的跨会话、跨 Agent 记忆架构，全称为 **Multi-Graph Adaptive Memory Architecture**。

它负责把对话事件、实体、关系、向量、召回记录和反馈写入本地 SQLite + FAISS 记忆层，并通过 `magma-recall` OpenClaw 插件，在 Agent 构建提示词前自动注入相关记忆。

![MAGMA 多图谱智能体记忆架构](docs/assets/magma-hero.jpg)

## 图解

### 1. 架构全景：三层解耦的记忆大脑

![MAGMA 架构全景](docs/assets/magma-architecture-overview.jpg)

### 2. 意图感知路由：语义、关键词、时间信号融合

![MAGMA 意图感知路由](docs/assets/magma-intent-router.jpg)

### 3. 基准效果：复杂推理场景下的稳定召回

![MAGMA 基准效果](docs/assets/magma-benchmark.jpg)

## 当前运行态

- Embedding 模型：本地 `Qwen3-Embedding-0.6B`
- Embedding 维度：1024
- 慢路径 LLM 后端：DeepSeek V3 via OpenRouter
- 主 API：`http://127.0.0.1:8904`
- MCP 服务：指向 8904 主 API 的薄代理

LLM 后端和 embedding 模型是两套东西，不能混用。DeepSeek V3 用于慢路径关系抽取、因果推断和记忆巩固；`Qwen3-Embedding-0.6B` 用于本地语义向量召回。历史上的 MiniLM-L6-v2 / 384 维向量和 bge-small-zh-v1.5 / 512 维向量不是当前运行态。

## 可用性边界

这个仓库包含 MAGMA 的代码、OpenClaw 插件和运维脚本，但不包含任何本地运行数据。首次启动会创建一个空数据库；要看到真实召回效果，需要通过 API、MCP 或 OpenClaw 插件持续写入记忆。

开箱即用：

- `magma-recall` 插件自动捕获对话并注入召回
- MCP 服务器提供 20 个工具（12 核心 + 8 扩展）
- `magma_doctor` 健康检查（红黄绿）
- `magma_l1_distill_llm.py` L1 语义提炼

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API 服务器
python -m uvicorn magma.api.server:create_app --factory --host 127.0.0.1 --port 8904

# 健康检查
python -m magma.ops status
```

## 目录结构

```
magma/
├── api/            # FastAPI 服务器 + MCP 服务器
├── graph/          # SQLite 存储 + FAISS 索引
├── search.py       # 语义 + 关键词混合搜索
├── fact_extractor.py    # LLM 原子事实提取
├── context_synthesis.py # 上下文合成
├── encoder.py      # Embedding 编码器
└── recall_eval.py  # 召回质量评估
scripts/
├── seed_operational_anchors.py  # 运维锚点注入
├── magma_l1_distill_llm.py     # L1 语义提炼
└── magma_governance.py          # 轻量治理
```

## 技术细节

- **存储**：SQLite（节点/边/事实）+ FAISS（向量索引）
- **搜索**：语义向量 + 关键词 + 时间衰减 + 因果图谱
- **提炼**：L0 原始对话 → L1 事实/决策/状态（规则 + LLM）
- **召回**：意图路由 → Beam Search 图遍历 → 上下文合成

## 许可

私有仓库，仅供内部使用。
