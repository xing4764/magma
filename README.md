# MAGMA

**Multi-Graph Adaptive Memory Architecture**，面向 OpenClaw 多 Agent 系统的本地长期记忆架构。

MAGMA 将对话事件、实体、关系、向量、召回记录和反馈写入本地 `SQLite + FAISS` 记忆层，并通过 `magma-recall` OpenClaw 插件在 Agent 构建提示词前自动注入相关记忆。它的目标不是替代模型上下文，而是把跨会话、跨 Agent、可追溯的长期事实变成默认可用的认知层。

![MAGMA 架构](docs/assets/magma-hero.jpg)

## 当前状态

MAGMA 已转为公开仓库形态，代码可以独立部署和二次开发。仓库不包含运行数据、数据库、日志、模型权重和私有密钥；所有敏感配置都通过环境变量或本地 `.env` 提供。

公开仓库地址：

https://github.com/xing4764/magma

## 核心能力

### 基础架构
- **多层记忆结构**：L0 原始对话层、L1 事实/决策层、L2 高价值巩固层。
- **多图分离**：语义图、时间图、情景/关系图分开维护，查询时按意图组合。
- **混合检索**：向量语义召回、中文关键词召回、时间衰减和生命周期过滤共同参与排序。
- **自动召回**：OpenClaw `before_prompt_build` 阶段注入相关记忆。
- **自动抓取**：OpenClaw `agent_end` 阶段自动写入对话 L0 记忆。
- **召回反馈闭环**：记录 recall batch、弱正反馈、importance 变化和 FA_MISS 信号。
- **MCP 工具接口**：13 个工具（查询、增删改、反馈、验证、统计、治理），方便 Agent 直接调用。

### Harness-1 优化（2026-06-09 新增）

借鉴 [Harness-1](https://arxiv.org/abs/2606.02373) 论文的 State-Externalizing Harness 理念，新增 19 项优化：

**写入层**
- **内容去重**：MinHash + LSH（阈值 0.85），相似记忆自动合并而非新建。
- **质量自动标注**：L1 distill 时自动标注 importance_label（状态变更→useful，约束→useful，运维→reference）。

**召回层**
- **本地 Reranker**：cross-encoder 精排（`ms-marco-MiniLM-L-6-v2`），与 RRF 0.7/0.3 融合。默认关闭。
- **Bridge Entity 检测**：统计出现在 ≥2 个不同节点中的实体，作为多源交叉验证信号。
- **Two-Tier 标记**：L1 节点→summary（只返回摘要），L0 节点→full（返回全文），importance≥0.9 强制 full。
- **RRF Intent 权重**：根据查询意图动态调整（why→causal ×5.0，when→temporal ×4.0，entity→entity ×3.0）。
- **Graph 缓存优化**：intent-aware 缓存 key，causal intent 不缓存。

**策展层**
- **减法策展**：首次搜索自动 top-8 加入精选集，模型只需做"升级好的、删除差的"。
- **重要性标签**：4 级语义标签（critical / useful / reference / noise），节点级 + 检索级双层。
- **Sentence Compress**：BM25 句子打分取 top-K 关键句，summary tier 节点自动压缩。默认关闭。

**反馈层**
- **FA_MISS 信号**：标注"被召回但未使用的关键记忆"，保护机制防止 importance 衰减。
- **Verify 验证**：claim + node_ids → verdict（supported/unsupported/partial），LLM 推理，1h 缓存。

**安全层**
- **Token Budget 分级**：critical 任务不暴露预算（防止草率决策），normal 正常暴露，low 在 60% 提示收尾。
- **回溯检测**：同一 session 3 次查询结果重复率 >60% 时注入提示，不强制中断。
- **Feature Flag**：8 个 `MAGMA_FEATURE_*` 环境变量开关，可独立启用/禁用各功能。

## 适用场景

- OpenClaw 多 Agent 长期记忆（跨会话、跨 Agent 的持久化知识层）。
- 需要智能策展的检索场景（减法筛选、重要性标签、多源交叉验证）。
- 运营、技术、助理等 Agent 的共享知识层（Bridge Entity 自动检测交叉验证）。
- 高风险决策场景（Verify 声明验证、Token Budget 分级管控、回溯检测）。
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
| `MAGMA_API_PORT` | API 端口 | `8904` |
| `MAGMA_API_BASE` | MCP/插件访问 API 的地址 | `http://127.0.0.1:8904` |
| `MAGMA_DB_PATH` | SQLite 数据库路径 | `./data/magma.db` |
| `MAGMA_EMBEDDING_MODEL` | Embedding 模型名称或本地路径 | `Qwen/Qwen3-Embedding-4B` |
| `OPENROUTER_API_KEY` | 慢路径 LLM API Key（Verify 接口依赖） | 空 |

Feature Flag 开关（可选）：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `MAGMA_FEATURE_BEAM_SEARCH` | Beam Search | `1`（开启） |
| `MAGMA_FEATURE_RECENCY_BOOST` | 时间衰减加权 | `1` |
| `MAGMA_FEATURE_RRF_FUSION` | RRF 多路融合 | `1` |
| `MAGMA_FEATURE_GRAPH_WALK` | 图谱遍历 | `1` |
| `MAGMA_FEATURE_INTENT_ROUTING` | 意图路由 | `1` |
| `MAGMA_FEATURE_LOCAL_RERANKER` | 本地 Reranker 精排 | `0`（关闭） |
| `MAGMA_FEATURE_CONTENT_DEDUP` | MinHash 内容去重 | `1` |
| `MAGMA_FEATURE_SUBTRACTIVE_CURATION` | 减法策展 | `1` |
| `MAGMA_FEATURE_SENTENCE_COMPRESS` | 句子压缩 | `0`（关闭） |

注意：不要提交 `.env`、数据库、日志、模型目录或任何真实 API Key。

### 4. 启动 API

```bash
python -m uvicorn magma.api.server:create_app --factory --host 127.0.0.1 --port 8904
```

如果你希望使用 Qwen3 Embedding 等本地模型，可把 `MAGMA_EMBEDDING_MODEL` 指向本地模型目录，例如：

```bash
set MAGMA_EMBEDDING_MODEL=C:\openclaw-magma\models\Qwen\Qwen3-Embedding-4B
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
          "apiBaseUrl": "http://127.0.0.1:8904",
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
          "MAGMA_API_BASE": "http://127.0.0.1:8904"
        },
        "timeout": 60000
      }
    }
  }
}
```

MCP server 默认作为 API 的薄代理，不直接冷加载 embedding 模型，也不绕过 FastAPI 治理逻辑。

### MCP 工具列表（13 个）

| 工具 | 说明 |
|------|------|
| `magma_query` | 语义+关键词混合检索，支持 priority/tier_override 参数 |
| `magma_add_node` | 添加记忆节点 |
| `magma_update_node` | 更新节点（支持 importance_label） |
| `magma_delete_node` | 软删除节点 |
| `magma_get_node` | 获取单个节点详情 |
| `magma_list_nodes` | 列出节点（支持按 label 过滤） |
| `magma_search_by_entity` | 按实体精确查找 |
| `magma_feedback` | 召回反馈（支持 used/recalled/missed） |
| `magma_verify` | 声明验证（claim → verdict） |
| `magma_stats` | 知识图谱统计 |
| `magma_doctor` | 健康检查 |
| `magma_mark_important` | 标记重要 |
| `magma_mark_wrong` | 标记错误 |

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

# 召回质量评估（基础）
python scripts/magma_recall_eval.py

# 新能力评测（Reranker/去重/Bridge/Budget/回溯/Verify）
python scripts/magma_p2_eval.py

# 治理 dry-run
python scripts/magma_governance.py --dry-run

# 软治理
python scripts/magma_governance.py --apply
```

## 目录结构

```text
magma/
  api/
    server.py            # FastAPI 服务
    mcp_server.py        # MCP 工具服务（13 个工具）
  graph/
    sqlite_store.py      # SQLite 节点/边/事件存储
    faiss_index.py       # FAISS 向量索引
  search.py              # 混合检索 + RRF + Bridge Entity + Two-Tier + 回溯检测
  context_synthesis.py   # 召回结果合成、Token Budget、Sentence Compress
  encoder.py             # Embedding 编码器（Qwen3-Embedding-4B）
  recall_feedback.py     # 召回反馈闭环
  capture_policy.py      # 写入策略 + MinHash 去重
  l1_distiller.py        # L1 蒸馏 + 质量自动标注
  reranker.py            # 本地 Reranker（cross-encoder，默认关闭）
  sentence_compress.py   # BM25 句子压缩（默认关闭）

openclaw-plugin-magma-recall/
  index.js               # OpenClaw hook 插件

scripts/
  magma_doctor.py        # 状态诊断
  magma_ops.py           # 运维入口
  magma_governance.py    # 记忆治理
  magma_recall_eval.py   # 召回评估（23/24, 95.8%）
  magma_p2_eval.py       # 新能力评测（12/14, 85.7%）
  classify_unknown_nodes.py  # 节点层级分类
  cleanup_short_l0.py    # 超短 L0 清理
  purge_expired_nodes.py # 过期节点清理
```

## 架构概览

### 写入链路

```text
OpenClaw Agent
  -> magma-recall agent_end hook
  -> MAGMA API
  -> MinHash 内容去重（相似度 > 0.85 合并）
  -> SQLite 节点/边/事件
  -> Embedding 编码（Qwen3-Embedding-4B, 2560 维）
  -> FAISS 索引
  -> L1 distill 时自动标注 importance_label
```

### 读取链路

```text
用户消息
  -> before_prompt_build
  -> 意图路由（why/when/entity/general）
  -> 混合检索（向量 + 关键词 + 图谱遍历 + Beam Search）
  -> RRF intent-driven 融合
  -> 本地 Reranker 精排（可选）
  -> Bridge Entity 检测 + Two-Tier 标记
  -> 生命周期/importance/来源/时间衰减加权
  -> 减法策展（active_context top-8）
  -> Sentence Compress 压缩（可选）
  -> Token Budget 分级管控
  -> 回溯检测
  -> 上下文合成
  -> 注入 Agent 提示词
```

### 反馈闭环

```text
Agent 使用记忆
  -> magma_feedback（used / recalled / missed）
  -> FA_MISS 信号保护（missed 节点 importance 不衰减）
  -> 重复查询 → 回溯检测提示
  -> 重要性动态调整
```

## Feature Flag

所有新功能通过环境变量控制，可独立开关：

```bash
# 开启 Reranker（需先下载模型）
set MAGMA_FEATURE_LOCAL_RERANKER=1

# 开启 Sentence Compress
set MAGMA_FEATURE_SENTENCE_COMPRESS=1

# 关闭减法策展
set MAGMA_FEATURE_SUBTRACTIVE_CURATION=0

# 关闭内容去重
set MAGMA_FEATURE_CONTENT_DEDUP=0
```

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
5. 运行召回评估：`python scripts/magma_recall_eval.py`
6. 提交 PR，并说明变更动机、影响范围和验证结果。

更多细节见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT License
