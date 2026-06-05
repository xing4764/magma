# MAGMA

**Multi-Graph Adaptive Memory Architecture**，面向 OpenClaw 多 Agent 系统的本地长期记忆架构。

MAGMA 将对话事件、实体、关系、向量、召回记录和反馈写入本地 `SQLite + FAISS` 记忆层，并通过 `magma-recall` OpenClaw 插件在 Agent 构建提示词前自动注入相关记忆。它的目标不是替代模型上下文，而是把跨会话、跨 Agent、可追溯的长期事实变成默认可用的认知层。

![MAGMA 架构](docs/assets/magma-hero.jpg)

## 当前状态

MAGMA 已转为公开仓库形态，代码可以独立部署和二次开发。仓库不包含运行数据、数据库、日志、模型权重和私有密钥；所有敏感配置都通过环境变量或本地 `.env` 提供。

公开仓库地址：

https://github.com/xing4764/magma

## 核心能力

- **多层记忆结构**：L0 原始对话层、L1 事实/决策层、L2 高价值巩固层。
- **多图分离**：语义图、时间图、情景/关系图分开维护，查询时按意图组合。
- **混合检索**：向量语义召回、中文关键词召回、时间衰减和生命周期过滤共同参与排序。
- **自动召回**：OpenClaw `before_prompt_build` 阶段注入相关记忆。
- **自动抓取**：OpenClaw `agent_end` 阶段自动写入对话 L0 记忆。
- **召回反馈闭环**：记录 recall batch、弱正反馈和 importance 变化。
- **治理与运维**：doctor、ops、governance、recall eval 等脚本支持状态诊断、软治理和质量评估。
- **MCP 工具接口**：提供查询、增删改、反馈、统计和治理相关工具，方便 Agent 直接调用。

## 适用场景

- OpenClaw 多 Agent 长期记忆。
- 跨会话项目上下文追踪。
- 运营、技术、助理等 Agent 的共享知识层。
- 需要本地部署、可控数据边界和可审计召回记录的 AI 工作流。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/xing4764/magma.git
cd magma
```

### 2. 安装依赖

建议使用 Python 3.10+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量

复制模板：

```bash
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

常用变量：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `MAGMA_API_HOST` | API 监听地址 | `127.0.0.1` |
| `MAGMA_API_PORT` | API 端口 | `8902` |
| `MAGMA_API_BASE` | MCP/插件访问 API 的地址 | `http://127.0.0.1:8902` |
| `MAGMA_DB_PATH` | SQLite 数据库路径 | `./data/magma.db` |
| `MAGMA_EMBEDDING_MODEL` | Embedding 模型名称或本地路径 | `BAAI/bge-small-zh-v1.5` |
| `OPENROUTER_API_KEY` | 慢路径 LLM API Key，可选 | 空 |

注意：不要提交 `.env`、数据库、日志、模型目录或任何真实 API Key。

### 4. 启动 API

```bash
python -m uvicorn magma.api.server:create_app --factory --host 127.0.0.1 --port 8902
```

如果你希望使用 Qwen3 Embedding 等本地模型，可把 `MAGMA_EMBEDDING_MODEL` 指向本地模型目录，例如：

```bash
set MAGMA_EMBEDDING_MODEL=C:\openclaw-magma\models\Qwen\Qwen3-Embedding-0___6B
```

### 5. 健康检查

```bash
python scripts/magma_doctor.py --quick
python scripts/magma_ops.py status
```

## OpenClaw 集成

MAGMA 有两种接入方式，可以同时使用。

### 方式一：magma-recall 插件

在 `openclaw.json` 中启用插件：

```json
{
  "plugins": {
    "allow": ["magma-recall"],
    "entries": {
      "magma-recall": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "apiBaseUrl": "http://127.0.0.1:8902",
          "topK": 6,
          "timeoutMs": 30000,
          "scoreThreshold": 0.35,
          "capture": {
            "enabled": true,
            "ttlDays": 180,
            "maxChars": 4000
          }
        }
      }
    },
    "load": {
      "paths": [
        "C:\\openclaw-magma\\openclaw-plugin-magma-recall"
      ]
    }
  }
}
```

插件能力：

- `before_prompt_build`：自动召回并注入相关记忆。
- `before_message_write`：剥离注入内容，保持会话历史干净。
- `agent_end`：自动抓取本轮对话并写入 L0。

### 方式二：MCP 工具

在 `openclaw.json` 中添加 MCP server：

```json
{
  "mcp": {
    "servers": {
      "magma-memory": {
        "command": "python",
        "args": ["-m", "magma.api.mcp_server"],
        "cwd": "C:\\openclaw-magma",
        "env": {
          "MAGMA_API_BASE": "http://127.0.0.1:8902"
        },
        "timeout": 60000
      }
    }
  }
}
```

MCP server 默认作为 API 的薄代理，不直接冷加载 embedding 模型，也不绕过 FastAPI 治理逻辑。

## 常用命令

```bash
# API 健康检查
python scripts/magma_doctor.py --quick

# 完整诊断
python scripts/magma_doctor.py --json

# 运维状态
python scripts/magma_ops.py status

# 安全修复建议
python scripts/magma_ops.py repair

# 召回质量评估
python scripts/magma_recall_eval.py

# 治理 dry-run
python scripts/magma_governance.py --dry-run

# 软治理
python scripts/magma_governance.py --apply
```

## 目录结构

```text
magma/
  api/
    server.py          # FastAPI 服务
    mcp_server.py      # MCP 工具服务
  graph/
    sqlite_store.py    # SQLite 节点/边/事件存储
    faiss_index.py     # FAISS 向量索引
  search.py            # 混合检索
  context_synthesis.py # 召回结果合成与排序
  encoder.py           # Embedding 编码器
  recall_feedback.py   # 召回反馈闭环

openclaw-plugin-magma-recall/
  index.js             # OpenClaw hook 插件

scripts/
  magma_doctor.py      # 状态诊断
  magma_ops.py         # 运维入口
  magma_governance.py  # 记忆治理
  magma_recall_eval.py # 召回评估
```

## 架构概览

### 写入链路

```text
OpenClaw Agent
  -> magma-recall agent_end hook
  -> MAGMA API
  -> SQLite 节点/边/事件
  -> Embedding 编码
  -> FAISS 索引
```

### 读取链路

```text
用户消息
  -> before_prompt_build
  -> 混合检索
  -> 生命周期/importance/来源加权
  -> 上下文合成
  -> 注入 Agent 提示词
```

### 短指令召回改进方向

短指令如“更新”“继续”“开始”语义不足，不能只依赖普通向量相似度。MAGMA 的目标策略是：

```text
短指令
  -> 最近会话锚点
  -> L1 decision/task_intent 节点优先
  -> L0 原文证据补充
  -> 最终召回
```

这能减少“记住了但召回不到”的问题。

## 数据与安全

仓库默认排除：

- `data/`
- `*.db`, `*.db-wal`, `*.db-shm`
- `models/`
- `logs/`
- `.env`
- `backups/`

公开前建议运行：

```bash
git status --short
git ls-files | findstr /R "\.env$ \.db$ data/ models/ logs/"
```

如果输出包含敏感文件，请先移出仓库或更新 `.gitignore`。

## 贡献指南

欢迎提交 Issue 和 Pull Request。建议流程：

1. Fork 仓库。
2. 创建分支：`git checkout -b feature/your-feature`
3. 修改代码并补充必要测试。
4. 运行基础检查：`python scripts/magma_doctor.py --quick`
5. 提交 PR，并说明变更动机、影响范围和验证结果。

更多细节见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT License
