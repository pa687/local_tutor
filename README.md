# Local Tutor

完全本地运行的初高中 AI 家教系统。单用户 / 少量用户，`llama-server` 是唯一主推理后端。

- **权威计划**：[`ENGINEERING_PLAN.md`](ENGINEERING_PLAN.md)（`temp.md` 定稿）。其中 **§0–§23、§27 为权威**；§24–§26 是写给 Coding Agent 的示例任务，**仅供参考，非权威**。
- **阶段顺序与 DoD**：见本地文件 `work.md`（**不入库**）。
- **本地专用文档（故意不提交）**：`work.md`（阶段顺序 / DoD / 裁决记录）、`temp.md`（计划原件）、`PROJECT_MEMORY.md`（项目记忆：约定、环境、踩坑、进度）。

## 当前状态

**Phase 2 — 最小 Tutor Engine + 完整 Tutor Policy Layer**（已完成）：

- `/api/chat` 不再直接调用 `LlamaClient`，也不再拼 prompt；统一走
  `TutorEngine → Classifier → TutorPolicy → ContextBuilder → LlamaClient`（§6、§12）；
- 三个模式 `tutor` / `explain` / `check` 都返回结构化响应（`subject`、`estimated_grade`、
  `topic`、`confidence`、`tools_used`、`warnings`）；
- prompt 全部从 `prompts/*.md` 读取（§13），改 prompt 不需要改 Python；
- `policy.py` 为完整实现：五条 Tutor 规则 + 年级约束，均**不**写在 system prompt 里；
- 逐条验收与真实模型记录见 [`docs/phase2_verification.md`](docs/phase2_verification.md)。

**仍无工具 / 数据库 / RAG**：`tools_used` 恒为空，`verification_status` 日志字段仍为 null（Phase 3/4 填）。

## 环境

- Python **3.12**，由 `uv` 管理
- `uv` 0.12+（本机位于 `~/.local/bin/uv`）

## 安装

```bash
uv sync          # 创建 .venv 并安装运行依赖 + dev 依赖
```

## 运行

先启动推理后端（需要 `LLAMA_SERVER_BIN` / `LLAMA_MODEL_PATH`，见 `.env.example`）：

```bash
cp .env.example .env      # 填入本机路径（.env 不入库）
scripts/start_llama.sh    # 默认 ctx=65536 kv=q8_0 parallel=1 fa=on ngl=999
```

再启动后端：

```bash
uv run uvicorn tutor.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
# {"backend":"ok","llama":"ok","model":"qwen3.5-9b","context_size":65536}
```

llama-server 不在时 `/health` 返回 `llama: "down"`（HTTP 仍 200），`/api/chat` 返回 503 —— 后端本身不会崩。

### 基准测试

```bash
BENCH_CONTEXTS="65536 98304 131072" scripts/benchmark_model.sh   # 结果追加到 docs/llama_benchmarks.md
```

## API

### `POST /api/chat`（SSE）

```json
{"student_id": "student-001", "conversation_id": "abc", "mode": "tutor", "grade": 9, "message": "为什么这里要配方？"}
```

- `mode` ∈ `tutor` / `explain` / `check`，缺省取 `tutor.default_mode`；其他值返回 **422**。
- `grade` 可选（1–12）；不传时用分类器估计的年级（Phase 7 接入真实学生档案）。

响应 `text/event-stream`，事件序列：

```text
event: start   data: {"request_id", "model", "mode", "subject", "estimated_grade",
                      "topic", "confidence", "tools_used", "warnings"}
event: token   data: {"text": "…"}
event: done    data: {"request_id", "prompt_tokens", "completion_tokens", "latency_ms",
                      "mode", "subject", "estimated_grade", "topic", "confidence",
                      "tools_used", "warnings"}
event: error   data: {"error", "detail"}      # 流已开始后模型侧失败时
```

`start` 与 `done` 的 metadata 形状相同：`start` 在首个 token 前给出（`confidence` 为分类阶段的基线值），
`done` 在答案写完后给出（`confidence` 已计入策略检查，`warnings` 可能新增）。
**`answer` 不在 `done` 里重复出现** —— 客户端已经从 `token` 事件拿到全文。

首批 token 之前就失败的情况用 HTTP 状态码表达（无需解析流）：

| 情况 | 状态码 | 异常 |
|---|---|---|
| llama-server 未启动 / crash / OOM 被杀 | 503 | `LlamaUnavailableError` |
| 超过 `llm.timeout` 无响应 | 504 | `LlamaTimeoutError` |
| llama-server 返回错误状态或异常 payload | 502 | `LlamaResponseError` |
| 请求体缺字段 / message 为空 / mode 非法 / grade 越界 | 422 | pydantic 校验、`InvalidTutorModeError` |

### `GET /health`

`{"backend": "ok", "llama": "ok"|"down", "model": "<config.llm.model>", "context_size": <llama-server 的 n_ctx 或 null>}`

## 质量门

```bash
uv run pytest                     # 单元测试（默认跳过 integration）
uv run pytest -m integration      # 需要 llama-server 在跑 + LLAMA_INTEGRATION=1
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

四条命令必须在提交前无错误通过（§23.10）。真实后端验证：

```bash
scripts/start_llama.sh &
LLAMA_INTEGRATION=1 uv run pytest -m integration -v
```

## 配置（§19）

唯一配置文件是仓库根目录的 `config.yaml`，任何值都可被环境变量覆盖：

```bash
LOCAL_TUTOR__LLM__MODEL=qwen3.6-27b uv run uvicorn tutor.main:app
```

- 覆盖规则：`LOCAL_TUTOR__<SECTION>__<KEY>`，值按 YAML 标量解析（`true` / `65536` / 字符串）。
- 配置文件路径可用 `TUTOR_CONFIG_PATH` 指定；prompt 目录可用 `TUTOR_PROMPTS_DIR` 指定（默认 `<repo>/prompts`）。
- **禁止在 Python 代码中硬编码模型路径或绝对路径**（§1.5）；模型路径一律经配置/环境变量注入。
- 推理参数同样属配置：`llm.enable_thinking` 控制混合思考模型是否先思考再作答
  （默认 `false`：实测关闭后 TTFT 0.19 s，开启后 23.21 s，见 `docs/phase2_verification.md` §3.2；
  **分类调用永远关闭思考**）。
- 覆盖示例见 `.env.example`。

## Prompt 文件（§13）

```text
prompts/tutor_system.md      # 短而硬的 system prompt（§13 的原则）
prompts/solve.md             # 各模式共用的作答要求
prompts/classify.md          # 题目结构化（subject / grade / topic / uncertain）
prompts/verify.md            # Phase 4 接线
prompts/summarize_student.md # Phase 8 接线
```

- 每次访问都从磁盘读取：改 prompt 立即生效，不需要重启、不需要改代码。
- 模式相关规则（五条 Tutor 规则、年级约束）**不在** prompt 里，由 `backend/tutor/tutor/policy.py`
  生成并作为独立层注入；`tests/test_context.py::TestPolicyStaysOutOfPrompts` 守住这条线。
- Qwen3.5 的 chat template 要求 system message 必须在最前，因此三层（system / 作答要求 / policy）
  会拼成**一条** system message 发送，逻辑分层不变。

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
│   ├── api/             # chat.py(P2/P6) students.py(P7) health.py(P0)
│   ├── llm/             # client.py(P1) models.py(P1) prompts.py(P2) context.py(P2)
│   ├── tutor/           # engine.py(P2) policy.py(P2) classifier.py(P2) verifier.py(P4) response.py(P2)
│   ├── tools/           # registry.py calculator.py algebra.py (P3) units.py (reserved)
│   ├── memory/          # student.py(P7) conversation.py(P8) summarizer.py(P8)
│   ├── retrieval/       # ingest.py chunk.py search.py models.py (P9)
│   └── db/              # models.py session.py (P7)
├── frontend/            # P11
├── prompts/             # P2：tutor_system / solve / classify / verify / summarize_student
├── eval/                # datasets/ runner.py graders.py reports/ (P5)
├── docs/                # 实测记录：llama_benchmarks.md、phase1_verification.md、phase2_verification.md
├── scripts/             # start_llama.sh benchmark_model.sh (P1) bench_llama.py smoke_test.sh (P5)
└── tests/
```

括号内的 `P<n>` 表示该文件由哪个阶段实现。当前未被实现的文件是**显式占位**：
只有一行 docstring 说明归属阶段，**不含任何逻辑**。
