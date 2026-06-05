# Contributing to MAGMA

感谢你愿意参与 MAGMA。这个项目关注的是本地长期记忆、OpenClaw 多 Agent 集成和可审计召回质量。

## 开发环境

```bash
git clone https://github.com/xing4764/magma.git
cd magma
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

## 提交前检查

至少运行：

```bash
python scripts/magma_doctor.py --quick
```

如果改动涉及召回、排序、生命周期或反馈闭环，也建议运行：

```bash
python scripts/magma_recall_eval.py
python scripts/magma_governance.py --dry-run
```

## 代码原则

- 保持改动聚焦，避免无关重构。
- 不提交本地数据、数据库、日志、模型权重和 API Key。
- 新增配置优先走环境变量或 `.env.example` 模板。
- 对召回排序、记忆治理、数据迁移等高影响逻辑，尽量提供可复现验证方式。

## PR 说明建议

请在 PR 中说明：

- 解决了什么问题。
- 改动了哪些模块。
- 是否影响数据库 schema、索引或 OpenClaw 插件配置。
- 你运行过哪些验证命令。

## 安全

如果你发现凭证泄漏、越权访问、数据泄露等安全问题，请不要直接公开细节。先通过私下渠道联系维护者，确认修复后再公开披露。
