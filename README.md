# MAGMA - Multi-Graph Adaptive Memory Architecture

> 为 AI Agent 提供跨会话、跨 Agent 的知识图谱记忆系统

## 简介

MAGMA 是一套面向 AI Agent 的持久化记忆架构，通过四正交图（Event / Entity / Relation / Concept）实现知识的自动召回、自动抓取和跨 Agent 共享。设计目标是在多 Agent 协作场景下，让每个 Agent 都能"记住"历史对话、决策和关键事件，同时保持记忆的一致性和治理能力。

## 核心能力

| 能力 | 说明 |
|------|------|
| **自动召回** | Agent 启动时自动加载相关上下文记忆 |
| **自动抓取** | 对话结束后自动提取关键信息写入记忆 |
| **跨 Agent 共享** | 多 Agent 通过统一图存储共享知识 |
| **反馈闭环** | 记忆召回结果支持正负反馈，持续优化排序 |
| **意图路由** | 根据查询意图选择召回策略（精确 / 模糊 / 因果） |
| **轻量治理** | 内置重复检测、过期清理、重要度评分 |

## 架构

```
┌─────────────────────────────────────────────────┐
│                   OpenClaw Host                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  Agent A   │  │  Agent B   │  │  Agent C   │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │          │
│        └──────────┬───┴──────────────┘          │
│                   │                             │
│        ┌──────────▼──────────┐                  │
│        │   MAGMA Recall Plugin│  ← OpenClaw 插件 │
│        └──────────┬──────────┘                  │
│                   │                             │
│        ┌──────────▼──────────┐                  │
│        │    MAGMA API Server  │  ← FastAPI + MCP │
│        │    (localhost:8902)  │                  │
│        └──────────┬──────────┘                  │
│                   │                             │
│     ┌─────────────┼─────────────┐              │
│     │             │             │              │
│  ┌──▼──┐   ┌─────▼─────┐  ┌───▼───┐         │
│  │SQLite│   │FAISS Index │  │Encoder│         │
│  │Graph │   │(bge-small) │  │(BGE)  │         │
│  └──────┘   └───────────┘  └───────┘         │
└─────────────────────────────────────────────────┘
```

四正交图模型：
- **Event Graph** — 对话事件、操作记录
- **Entity Graph** — 人名、项目、工具等实体锚点
- **Relation Graph** — 实体间的显式/隐式关系
- **Concept Graph** — 主题、标签、领域知识

## 技术栈

- **Python 3.10+**
- **SQLite** — 图存储（nodes / edges / recall_events / recall_feedback）
- **FAISS** — 向量索引，支持语义检索
- **BAAI/bge-small-zh-v1.5** — 512 维中文嵌入模型
- **FastAPI** — REST API 服务
- **MCP (Model Context Protocol)** — Agent 原生调用协议
- **OpenClaw Plugin SDK** — 插件集成

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
python magma/api/server.py
# 默认监听 http://127.0.0.1:8902
```

### 3. 配置 OpenClaw 插件

将 `openclaw-plugin-magma-recall/` 目录注册到 OpenClaw 插件系统：

```json
{
  "plugins": {
    "magma-recall": {
      "path": "./openclaw-plugin-magma-recall",
      "enabled": true
    }
  }
}
```

### 4. 验证

```bash
# 检查 API 健康状态
python scripts/magma_ops.py status

# 运行诊断
python scripts/magma_doctor.py

# CLI 查询记忆
python scripts/magma_cli.py search "你的查询"
```

## 目录结构

```
magma/
├── magma/                          # 核心 Python 包
│   ├── api/                        # FastAPI 服务 + MCP Server
│   ├── graph/                      # SQLite 图存储
│   └── vector/                     # FAISS 向量索引 + 编码器
├── scripts/                        # 运维脚本
│   ├── magma_cli.py                # CLI 工具（查询、导入、导出）
│   ├── magma_doctor.py             # 健康诊断（红/黄/绿）
│   ├── magma_ops.py                # 状态检查 + 安全修复
│   ├── magma_governance.py         # 记忆治理（去重、过期清理）
│   ├── magma_recall_eval.py        # 召回质量评估
│   ├── backfill_entities.py        # 实体锚点回填
│   ├── seed_operational_anchors.py # 运维锚点种子数据
│   └── migrate_source_agent.py     # 数据库迁移脚本
├── openclaw-plugin-magma-recall/   # OpenClaw 插件（Agent 自动召回/抓取）
├── data/                           # 运行时数据（不提交到 Git）
│   ├── magma.db                    # SQLite 数据库
│   └── faiss.index                 # FAISS 向量索引
├── requirements.txt
├── RUNBOOK.md                      # 运维手册
├── HANDOFF_OPENCLAW.md             # OpenClaw 集成说明
└── 起源.md                         # 项目设计文档
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MAGMA_DB_PATH` | SQLite 数据库完整路径 | `<project>/data/magma.db` |
| `MAGMA_DATA_DIR` | 数据目录 | `<project>/data/` |

## 运维手册

详见 [RUNBOOK.md](./RUNBOOK.md)

## 许可证

MIT License
