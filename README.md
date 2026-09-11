# Local Tutor

完全本地运行的初高中 AI 家教系统。单用户 / 少量用户，`llama-server` 是唯一主推理后端。

- **权威计划**：[`ENGINEERING_PLAN.md`](ENGINEERING_PLAN.md)（`temp.md` 定稿）。其中 **§0–§23、§27 为权威**；§24–§26 是写给 Coding Agent 的示例任务，**仅供参考，非权威**。
- **阶段顺序与 DoD**：见 `work.md`（在工作区 `~/models/`，不属于本仓库）。

## 当前状态

**Phase 0 — 工程基线与仓库骨架**：仓库结构、`pyproject.toml`、`config.yaml` + 环境变量覆盖、
logging 骨架（`request_id` + §18 metadata）、`GET /health` 占位、质量门。
**尚无任何 LLM 调用与业务逻辑**（Phase 0 明确不做）。

## 环境

- Python **3.12**，由 `uv` 管理
- `uv` 0.12+（本机位于 `~/.local/bin/uv`）

## 安装

```bash
uv sync          # 创建 .venv 并安装运行依赖 + dev 依赖
```

## 运行

```bash
uv run uvicorn tutor.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
# {"backend":"ok","llama":"unknown","model":"qwen3.5-9b","context_size":null}
```

## 质量门

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

三条命令必须在骨架下无错误通过（§23.10）。

## 配置（§19）

唯一配置文件是仓库根目录的 `config.yaml`，任何值都可被环境变量覆盖：

```bash
LOCAL_TUTOR__LLM__MODEL=qwen3.6-27b uv run uvicorn tutor.main:app
```

- 覆盖规则：`LOCAL_TUTOR__<SECTION>__<KEY>`，值按 YAML 标量解析（`true` / `65536` / 字符串）。
- 配置文件路径可用 `TUTOR_CONFIG_PATH` 指定。
- **禁止在 Python 代码中硬编码模型路径或绝对路径**（§1.5）；模型路径一律经配置/环境变量注入。
- 覆盖示例见 `.env.example`。

## 日志（§18）

- 每个请求生成 `request_id`，响应头回显 `X-Request-Id`。
- 每请求一行 JSON metadata：`request_id`、`student_id`、`conversation_id`、`model`、
  `prompt_tokens`、`completion_tokens`、`latency`、`tools_used`、`verification_status`。
- **普通 log 不记录学生消息内容**；对话内容进数据库。

## 仓库结构（§4）

```text
local-tutor/
├── README.md
├── ENGINEERING_PLAN.md
├── pyproject.toml
├── config.yaml
├── .env.example
├── backend/tutor/
│   ├── main.py          # app factory + request_id middleware
│   ├── config.py        # config.yaml + env overrides (§19)
│   ├── logging_config.py# §18 metadata logging skeleton
│   ├── api/             # chat.py(P1/P6) students.py(P7) health.py(P0)
│   ├── llm/             # client.py(P1) models.py(P1) prompts.py(P2) context.py(P2)
│   ├── tutor/           # engine.py(P2) policy.py(P2) classifier.py(P2) verifier.py(P4) response.py(P2)
│   ├── tools/           # registry.py calculator.py algebra.py (P3) units.py (reserved)
│   ├── memory/          # student.py(P7) conversation.py(P8) summarizer.py(P8)
│   ├── retrieval/       # ingest.py chunk.py search.py models.py (P9)
│   └── db/              # models.py session.py (P7)
├── frontend/            # P11
├── prompts/             # P2
├── eval/                # datasets/ runner.py graders.py reports/ (P5)
├── scripts/             # start_llama.sh benchmark_model.sh (P1) smoke_test.sh (P5)
└── tests/
```

括号内的 `P<n>` 表示该文件由哪个阶段实现。当前未被实现的文件是**显式占位**：
只有一行 docstring 说明归属阶段，**不含任何逻辑**。
