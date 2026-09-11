# Phase 2 验收记录 — 最小 Tutor Engine + 完整 Tutor Policy Layer

> 计划依据：`ENGINEERING_PLAN.md` §6、§12、§13、§10、§22
> 阶段清单：`work.md` Phase 2
> 执行日期：2026-09-11
> 环境：RTX 5070 Ti 16 GB / i9-14900HX / Ubuntu 26.04；`llama-server` b9010-1904，
> 模型 `Qwen_Qwen3.5-9B-Q4_K_M.gguf`（alias `qwen3.5-9b`），ctx 65536 / kv q8_0 / fa on / ngl 999

---

## 1. 结论

Phase 2 的调用链已落地并跑通真实模型：

```text
ChatRequest → TutorEngine → Classifier ┐
                         → TutorPolicy  ├→ ContextBuilder → LlamaClient → TutorResponse
                         └──────────────┘
```

* `/api/chat` 不再接触 `LlamaClient`，也不再拼任何 prompt；prompt 由 `prompts/*.md` 提供，policy 规则由 `policy.py` 生成。
* 三个模式（`tutor` / `explain` / `check`）都返回结构化 `TutorResponse`。
* 验收场景 A / B / C / E 用真实模型跑过（见 §4）。

---

## 2. DoD 逐条核对

| DoD（work.md Phase 2） | 状态 | 证据 |
|---|---|---|
| 三个模式都能端到端返回结构化 `TutorResponse` | ✅ | `tests/test_engine.py::TestThreeModes`（三模式参数化）、`tests/test_integration_llama.py::test_every_mode_answers_against_a_live_model`（真实模型） |
| 任意 endpoint 代码中不出现直接 `LlamaClient` 调用 | ✅ | `tests/test_api_boundaries.py`：AST 扫描 `api/*.py`，禁止 `.stream_chat` / `.complete` 调用与 `app.state.llama` 引用 |
| 修改 prompt 文件无需改 Python 代码 | ✅ | `PromptLibrary` 每次访问都读盘；`tests/test_prompts.py::test_edited_prompt_is_reread_without_a_restart`、`tests/test_context.py::test_prompt_edits_reach_the_context_without_code_changes` |
| 存在至少一个打真实 `llama-server` 的 integration 测试 | ✅ | `LLAMA_INTEGRATION=1 uv run pytest -m integration` → **9 passed**（17.1 s） |
| `policy.py` 为完整实现：五条规则 + 年级约束均有测试 | ✅ | `tests/test_policy.py`（55 项，按规则分 class：`TestRule1StudentAttempt`…`TestRule5`） |
| 这五条规则不出现在 system prompt 中 | ✅ | `tests/test_context.py::TestPolicyStaysOutOfPrompts`：逐条指令 + 6 个特征短语断言不出现在 `tutor_system.md` / `solve.md` |
| prompt 一律从 `prompts/*.md` 加载，不在 Python 中内嵌长字符串 | ✅ | `tests/test_skeleton.py::test_no_long_prompt_literals_in_python`（AST，>400 字符的字面量，docstring 豁免） |
| `confidence` 用 `high/medium/low` 枚举 | ✅ | `tutor/response.py::Confidence`；`tutor/engine.py::_baseline_confidence` / `_final_confidence` |

质量门（提交前，§23.10）：

```text
uv run pytest              → 241 passed, 9 skipped
uv run pytest -m integration → 9 passed        (LLAMA_INTEGRATION=1，真实模型)
uv run ruff check .        → All checks passed!
uv run ruff format --check . → 63 files already formatted
uv run mypy                → Success: no issues found in 51 source files
```

---

## 3. 真实环境发现（两个必须记录的后端约束）

### 3.1 Qwen3.5 的 chat template 只允许 system message 出现在最前

第一次真实集成运行 7 项全灭，`llama-server` 返回 HTTP 500：

```text
Jinja Exception: System message must be at the beginning.
```

模板（GGUF 内嵌，`/props` 可见，第 85 行）在 system message 不是首条时直接
`raise_exception`。因此「system 策略 / 作答要求 / policy 指令分三条消息发送」在这个模型上
不成立。

**处理**：`ContextBuilder` 把三层**拼接成单条 system message**（顺序仍为
`tutor_system.md` → `solve.md` → policy 块），逻辑分层不变、可测试性不变。
证据：`tests/test_context.py::test_the_system_layers_share_one_message`。

### 3.2 混合思考模型默认「先想 20 多秒，再吐第一个字」

`Qwen3.5-9B` 默认开启思考：推理内容进 `reasoning_content`，`content` 长时间为空
（实测 `max_tokens=200/700` 都只产出推理、`content` 为空字符串）。

实测同一道九年级二次方程（`thinking` 由 `chat_template_kwargs.enable_thinking` 控制）：

| 设置 | TTFT | 总耗时 | 输出长度 |
|---|---|---|---|
| `enable_thinking=false` | **0.19 s** | 1.06 s | 158 字 |
| `enable_thinking=true` | 23.21 s | 24.54 s | 251 字 |

两条回答质量相当（都是「找两个数，积为 6、和为 -5」的引导），但后者意味着学生盯着空白等 23 秒。
§17 又明确禁止展示 chain-of-thought，所以不能靠转发推理内容掩盖。

**处理**：§1.5 要求推理参数必须来自配置，因此新增 `llm.enable_thinking`（默认 `false`），
引擎经 `TutorEngine.answer_options` 读取；**分类调用永远关闭思考**（结构化抽取不需要思考，
开启后会把 token 预算烧在推理上、拿不到 JSON）。证据：
`tests/test_engine.py::test_answer_thinking_follows_the_configuration`。

对 hard 题目可以把该项改回 `true`（`LOCAL_TUTOR__LLM__ENABLE_THINKING=true`）。

### 3.3 分类调用实测

```text
[0.50s] '{"subject": "math", "grade": 9, "topic": "quadratic_equations", "uncertain": false}'
```

关闭思考后 0.5 s 稳定返回合法 JSON。

---

## 4. 验收场景（真实模型，grade=9，2026-09-11）

经 `TutorEngine.stream()` 直接驱动真实 `llama-server`：

| 场景 | 输入 | 模式 | 耗时 | 结果 |
|---|---|---|---|---|
| **A** §22 | 解方程 x²-5x+6=0 | tutor | 2.0 s | 只给第一步提示（「找两个数，积为 6、和为 -5」），**没有**直接给答案 ✅ |
| **B** §22 | 我算出来 x = 2，3，对吗？ | check | 2.5 s | 要求先看过程，不重新长篇解题 ✅ |
| **C** §22 | 为什么移项之后符号变了？ | tutor | 7.3 s | 用「天平/等式两边同时加减」按九年级讲，没有只说「移项要变号」✅ |
| **E** §22 | 我的过程：(x-2)(x-3)，所以 x=2，3 | check | 3.7 s | 逐步核对学生过程并判定正确，未重生成标准答案 ✅ |

Case B 的 `confidence=medium`：分类器判定题目信息不完整（`uncertain=true`），置信度按设计下调。

HTTP 层真实烟测（`uvicorn` + `curl`）：

```text
GET /health
{"backend":"ok","llama":"ok","model":"qwen3.5-9b","context_size":65536}

POST /api/chat → 200 text/event-stream
event: start
data: {"request_id":"707da929-…","model":"qwen3.5-9b","mode":"tutor","subject":"math",
       "estimated_grade":9,"topic":"quadratic_equations","confidence":"high",
       "tools_used":[],"warnings":[]}
event: token data: {"text":"你好"} …
event: done  data: {…,"latency_ms":…}

POST /api/chat {"mode":"socratic"} → 422 {"detail":"unknown tutor mode 'socratic'; allowed: tutor, explain, check"}
```

---

## 5. 与计划的偏差（§23.11：先报告，不偷偷改架构）

| 项 | 内容 | 理由 |
|---|---|---|
| `StudentRef` / `ConversationRef` | §6 签名是 `respond(student, conversation, message)`，而 `StudentProfile`（Phase 7）与 `Conversation`（Phase 8）尚不存在。本阶段用两个**只读值对象**顶位，签名形状保持不变 | 避免在 Phase 2 偷做学生档案 / 记忆库；Phase 7/8 只需扩展值对象，不必改调用链 |
| 新增 `prompts/classify.md` | §4 只列了 4 个 prompt 文件 | 题目结构化（§1）需要自己的 prompt；同时按 §4 补齐 `verify.md` / `summarize_student.md` 内容（Phase 4 / Phase 8 接线） |
| 新增 `llm.enable_thinking` | §19 的配置示例没有这个键 | §1.5 禁止硬编码推理参数，见 §3.2 |
| `ChatRequest` 增加可选 `grade` | §16 的请求体只有 4 个字段 | work.md Phase 2 第 7 条：年级「本阶段从请求 / 配置取，Phase 7 接入真实学生档案」 |
| `GenerationOptions` 增加 `enable_thinking` | Phase 1 模块 | 同上，透传为 `chat_template_kwargs` |
| `unknown mode` 由 422 拒绝 | 原来 `mode` 是自由字符串 | 未知模式无法静默降级；错误信息带 allowed 列表 |

---

## 6. 本阶段明确不做

`tools/`（Phase 3）、`verifier.py`（Phase 4）、eval harness（Phase 5）、多模态（Phase 6）、
学生档案与数据库（Phase 7）、对话记忆与摘要（Phase 8）、RAG（Phase 9）、UI（Phase 11）。

因此 `TutorResponse.tools_used` 恒为空，`warnings` 只承载「结构化失败 / 年级未知 / 超出年级的方法」，
`verification_status` 日志字段仍为 null（Phase 4 填）。`§22 Case D`（模糊图片）属 Phase 6。

## 7. 遗留 / 下一阶段输入

1. `tutor_system.md` 目前未承载任何模式规则，Phase 10 增加 `exam` / `practice` 时应在 `policy.py` 扩展而非写进 prompt。
2. 思考模式（`llm.enable_thinking=true`）下的 TTFT 已实测为 20 s+；Phase 11 的 UI 需要一个「正在思考」的占位状态，或 Phase 12 用同一数据集比较思考开关的**正确率**收益。
3. `find_out_of_scope_methods` 是启发式扫描（否定语境已处理），Phase 4 会被 SymPy 验证取代为更强的判据。
4. 分类器的科目词典（`_SUBJECT_ALIASES`）按实际出现的写法增补即可，无架构影响。
