# MAGMA - Multi-Graph Adaptive Memory Architecture

为 AI Agent 提供跨会话、跨 Agent 的知识图谱记忆系统。

## 核心能力

- **自动召回**：Agent 对话时自动注入相关记忆上下文
- **自动抓取**：对话内容自动写入知识图谱
- **跨 Agent 共享**：多个 Agent 之间的记忆互通
- **反馈闭环**：召回质量自动评估和优化
- **意图路由**：根据查询意图选择最优召回策略
- **轻量治理**：自动识别重复、过期、低质量记忆
- **健康监控**：红黄绿诊断 + 自助排障脚本

## 架构

四正交图架构（Event / Entity / Relation / Concept）：
- **Event**：对话事件、操作记录
- **Entity**：人物、项目、商品、概念
- **Relation**：实体之间的关系（depends_on、created_by 等）
- **Concept**：主题、领域、技能

## 技术栈

- Python 3.13+
- SQLite + FAISS（向量检索）
- FastAPI（API 服务）
- sentence-transformers（embedding）
- OpenClaw Plugin SDK

## 目录结构

```
magma/                          # 核心 Python 包
├── api/                        # API 服务 + MCP 服务器
├── core/                       # 图谱核心逻辑
├── encoder/                    # 文本编码器
├── search/                     # 搜索引擎
└── store/                      # 存储层

scripts/                        # 运维脚本
├── magma_doctor.py             # 健康诊断（红黄绿）
├── magma_ops.py                # 状态检查 + 安全自修
├── magma_governance.py         # 记忆治理（去重/降权/清理）
├── magma_recall_eval.py        # 召回质量评估
├── seed_operational_anchors.py # 运维锚点注入
└── migrate_source_agent.py     # 数据库迁移

openclaw-plugin-magma-recall/   # OpenClaw 插件
```

## 部署

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python -m magma.store.init

# 启动 API 服务
python -m magma.api.server

# 配置 OpenClaw MCP（在 openclaw.json 中）
# 参考 RUNBOOK.md
```

## 运维

```bash
# 健康检查
python scripts/magma_doctor.py --json

# 状态查看
python scripts/magma_ops.py status

# 安全自修
python scripts/magma_ops.py repair

# 记忆治理（dry-run）
python scripts/magma_governance.py --dry-run --json

# 召回质量测试
python scripts/magma_recall_eval.py
```

详细运维文档见 [RUNBOOK.md](RUNBOOK.md)。

## 许可证

MIT
