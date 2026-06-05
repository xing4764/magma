# MAGMA

**Multi-Graph Adaptive Memory Architecture** — 面向 OpenClaw 多 Agent 系统的跨会话、跨 Agent 记忆架构。

MAGMA 把对话事件、实体、关系、向量、召回记录和反馈写入本地 SQLite + FAISS 记忆层，并通过 `magma-recall` 插件在 Agent 构建提示词前自动注入相关记忆。

![MAGMA 架构](docs/assets/magma-hero.jpg)

## 特性

- **多图谱记忆**：实体图、因果图、时间线三层解耦
- **语义 + 关键词混合搜索**：FAISS 向量索引 + BM25 关键词 + 时间衰减
- **意图感知路由**：自动识别查询意图，融合语义/关键词/时间信号
- **L0 → L1 提炼**：原始对话自动提炼为事实/决策/状态节点
- **MCP 工具集**：20 个工具（12 核心 + 8 扩展），可直接对接 AI Agent
- **OpenClaw 插件**：`magma-recall` 自动捕获对话并注入召回

## 快速开始

### 前置条件

- Python 3.10+
- [OpenClaw](https://github.com/openclaw/openclaw)（可选，用于插件集成）

### 安装

```bash
git clone https://github.com/xing4764/magma.git
cd magma
pip install -r requirements.txt
```

### 启动 API 服务器

```bash
python -m uvicorn magma.api.server:create_app --factory --host 127.0.0.1 --port 8904
```

### 健康检查

```bash
python -m magma.ops status
```

### 配置环境变量

复制 `.env.example` 为 `.env`，按需修改：

```bash
cp .env.example .env
```

关键变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MAGMA_API_PORT` | API 端口 | 8904 |
| `MAGMA_EMBEDDING_MODEL` | Embedding 模型路径 | `BAAI/bge-small-zh-v1.5` |
| `OPENROUTER_API_KEY` | OpenRouter API Key（慢路径 LLM） | - |

### OpenClaw 插件集成

在 `openclaw.json` 中添加 MCP 服务器配置：

```json
{
  "mcp": {
    "servers": {
      "magma-memory": {
        "command": "python",
        "args": ["-m", "magma.api.mcp_server"],
        "cwd": "/path/to/magma",
        "env": {
          "MAGMA_API_BASE": "http://127.0.0.1:8904"
        }
      }
    }
  }
}
```

## 目录结构

```
magma/
├── api/
│   ├── server.py          # FastAPI 服务器
│   └── mcp_server.py      # MCP 工具服务器
├── graph/
│   ├── sqlite_store.py    # SQLite 节点/边存储
│   └── faiss_index.py     # FAISS 向量索引
├── search.py              # 语义 + 关键词混合搜索
├── fact_extractor.py      # LLM 原子事实提取
├── context_synthesis.py   # 上下文合成（拓扑排序 + 溯源）
├── encoder.py             # Embedding 编码器
├── recall_eval.py         # 召回质量评估
└── recall_feedback.py     # 召回反馈闭环
scripts/
├── magma_l1_distill_llm.py      # L1 语义提炼（LLM）
├── magma_l1_distill.py          # L1 规则提炼
├── seed_operational_anchors.py  # 运维锚点注入
├── magma_governance.py          # 轻量治理（去重/标旧）
├── magma_ops.py                 # 运维工具（status/repair）
└── magma_recall_eval.py         # 召回评估脚本
```

## 架构

### 三层解耦

```
L0（原始层）  ── 对话事件、实体、关系直接写入
    ↓ 提炼
L1（事实层）  ── 原子事实、决策、当前状态、运维锚点
    ↓ 巩固
L2（精炼层）  ── 高价值记忆自动提升权重
```

### 搜索管线

```
查询 → 意图路由 → [语义召回 | 关键词召回 | 时间召回] → RRF 融合 → 因果图扩展 → 上下文合成
```

## MCP 工具

| 工具 | 说明 |
|------|------|
| `magma_query` | 自然语言搜索记忆 |
| `magma_add_node` | 添加节点 |
| `magma_add_edge` | 添加关系边 |
| `magma_get_node` | 获取节点详情 |
| `magma_update_node` | 更新节点属性 |
| `magma_delete_node` | 软删除节点 |
| `magma_memory_edit` | 编辑记忆内容 |
| `magma_memory_forget` | 标记记忆失效 |
| `magma_feedback` | 召回反馈 |
| `magma_consolidate` | 手动清理/合并 |
| `magma_doctor` | 健康检查 |
| `magma_stats` | 统计信息 |

## 贡献

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交变更：`git commit -m "feat: your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 开发规范

- Python 代码遵循 PEP 8
- 新功能需附带测试
- 提交前运行 `python -m magma.ops status` 确认无破坏

## 许可证

MIT License
