# Local Tutor — 工程实施计划

## 0. 项目目标

构建一个完全本地运行的初高中 AI 家教系统。

目标运行环境：

* Ubuntu
* RTX 5070 Ti 16GB
* Intel i9-14900HX
* 64GB RAM
* llama.cpp 作为唯一主推理后端
* 主模型初始选择 Qwen3.5-9B GGUF
* 单用户/少量用户，不考虑高并发
* 优先保证：

  1. 解题可靠性
  2. 多模态题目理解
  3. 长上下文
  4. 可审计的工具调用
  5. 学生长期学习档案
* 不优先：

  * 超低首 token 延迟
  * 多租户
  * 云部署
  * Kubernetes
  * 炫技型前端

最终产品应当更像「私人家教」，而不是套壳 ChatGPT。

---

# 1. 产品行为原则

系统默认工作方式不是：

> 用户发题 → LLM 直接吐答案

而是：

```text
题目输入
    ↓
题目结构化
    ↓
识别年级 / 科目 / 知识点
    ↓
LLM 独立推理
    ↓
可验证部分交给工具校验
    ↓
检测答案与推理是否矛盾
    ↓
根据学生水平生成解释
    ↓
必要时追问 / 提示
    ↓
记录学生薄弱点
```

家教模式优先“教”，而不是“代写”。

系统支持至少三种回答模式：

```text
Tutor
不给完整答案优先，引导学生一步一步做。

Explain
完整讲解一道题，包括知识点、推导和答案。

Check
学生提供自己的过程，系统只检查哪里出错。
```

以后再加入：

```text
Exam
不给提示，只判断答案和评分。

Practice
根据学生薄弱点自动出同类题。

Socratic
原则上只通过提问推动学生思考。
```

---

# 2. MVP 技术架构

保持单体应用。

```text
Browser
   │
   ▼
Frontend
   │
   ▼
FastAPI
   │
   ├── Conversation Engine
   │
   ├── Tutor Policy
   │
   ├── Tool Dispatcher
   │       ├── Calculator
   │       ├── SymPy
   │       └── Unit checker
   │
   ├── Student Profile
   │
   ├── SQLite
   │
   └── Retrieval
   │
   ▼
llama-server
   │
   ▼
Qwen3.5-9B
```

llama.cpp 独立进程运行。

应用通过 OpenAI-compatible HTTP API 调用 llama-server。

不要：

* 把 llama.cpp 嵌入 Python
* 自己维护 CUDA binding
* 一开始引入 vLLM
* 引入 Redis
* 引入 PostgreSQL
* 拆微服务

---

# 3. 推荐技术栈

Backend：

```text
Python 3.12
FastAPI
Pydantic v2
httpx
SQLAlchemy / SQLModel
SQLite
SymPy
pytest
ruff
mypy
uv
```

Frontend：

优先简单。

建议：

```text
React
Vite
TypeScript
```

如果前端开发成为阻碍，可以第一阶段只写极简 HTML/JS。

Streaming 使用 SSE。

不要第一版就引入 WebSocket，除非确实有需求。

---

# 4. Repo 结构

```text
local-tutor/
├── README.md
├── ENGINEERING_PLAN.md
├── pyproject.toml
├── .env.example
│
├── backend/
│   └── tutor/
│       ├── main.py
│       │
│       ├── api/
│       │   ├── chat.py
│       │   ├── students.py
│       │   └── health.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   ├── models.py
│       │   ├── prompts.py
│       │   └── context.py
│       │
│       ├── tutor/
│       │   ├── engine.py
│       │   ├── policy.py
│       │   ├── classifier.py
│       │   ├── verifier.py
│       │   └── response.py
│       │
│       ├── tools/
│       │   ├── registry.py
│       │   ├── calculator.py
│       │   ├── algebra.py
│       │   └── units.py
│       │
│       ├── memory/
│       │   ├── student.py
│       │   ├── conversation.py
│       │   └── summarizer.py
│       │
│       ├── retrieval/
│       │   ├── ingest.py
│       │   ├── chunk.py
│       │   ├── search.py
│       │   └── models.py
│       │
│       └── db/
│           ├── models.py
│           └── session.py
│
├── frontend/
│
├── prompts/
│   ├── tutor_system.md
│   ├── solve.md
│   ├── verify.md
│   └── summarize_student.md
│
├── eval/
│   ├── datasets/
│   ├── runner.py
│   ├── graders.py
│   └── reports/
│
├── scripts/
│   ├── start_llama.sh
│   ├── benchmark_model.sh
│   └── smoke_test.sh
│
└── tests/
```

---

# 5. Phase 1 — llama.cpp 基础设施

目标：

保证本机推理稳定，然后再写应用。

初始模型：

```text
Qwen3.5-9B
Q5_K_M
```

初始配置不要直接挑战 128K。

先：

```text
context = 65536
KV = q8_0
parallel = 1
Flash Attention = on
GPU layers = all
```

稳定之后依次测试：

```text
64K
96K
128K
```

记录：

```text
VRAM idle
VRAM after model load
VRAM after 8K prompt
VRAM after 32K prompt
VRAM after target context

prompt tok/s
generation tok/s
time-to-first-token
```

必须编写：

```text
scripts/start_llama.sh
scripts/benchmark_model.sh
```

所有模型路径和推理参数必须从环境变量或配置文件读取。

禁止写死绝对路径。

### Phase 1 Definition of Done

下面全部满足才算完成：

```text
GET /health 能确认 llama-server 在线。

能进行普通文本聊天。

能 streaming 输出 token。

llama-server 重启后应用能自动恢复。

LLM HTTP 超时有明确异常。

OOM / llama-server crash 不导致 backend crash。

模型参数能够从 config 修改。
```

---

# 6. Phase 2 — 最小 Tutor Engine

不要让 API endpoint 直接拼 prompt。

建立明确调用链：

```text
ChatRequest
    ↓
TutorEngine
    ↓
ContextBuilder
    ↓
LLMClient
    ↓
TutorResponse
```

核心接口大致：

```python
class TutorEngine:
    async def respond(
        self,
        student: StudentProfile,
        conversation: Conversation,
        message: UserMessage,
    ) -> TutorResponse: ...
```

TutorResponse 至少包含：

```text
answer
subject
estimated_grade
topic
confidence
tools_used
warnings
```

confidence 不要求模型输出伪精确概率。

使用枚举：

```text
high
medium
low
```

---

# 7. Phase 3 — 数学可靠性闭环

这是整个项目最重要的阶段。

不要允许模型随意执行 Python。

第一版只提供结构化数学工具。

工具：

```text
calculator
solve_equation
simplify_expression
factor_expression
differentiate
integrate
evaluate_expression
check_equivalence
```

底层使用 SymPy。

例如：

```json
{
  "tool": "solve_equation",
  "arguments": {
    "equation": "x**2 - 5*x + 6",
    "variable": "x"
  }
}
```

执行结果：

```json
{
  "solutions": ["2", "3"]
}
```

必须严格验证 tool arguments。

不要：

```text
eval()
exec()
shell=True
直接执行模型生成 Python
```

数学回答流程：

```text
LLM 解题
    ↓
抽取可验证 claim
    ↓
SymPy 校验
    ↓
若冲突：
        让模型重新检查
    ↓
再次验证
    ↓
最终回答
```

最大 self-correction 次数：

```text
2
```

避免死循环。

如果仍然无法验证：

```text
confidence = low
warnings += "结果未能通过自动验证"
```

---

# 8. Phase 4 — 多模态题目输入

支持：

```text
JPEG
PNG
WEBP
```

不要第一版自己 OCR。

直接把图片交给 Qwen VLM。

但增加一个中间步骤。

对于题目图片，第一次模型调用只允许完成：

```text
Transcription
```

输出结构：

```json
{
  "question_text": "...",
  "detected_subject": "math",
  "diagram_description": "...",
  "values": [],
  "uncertain_regions": []
}
```

第二次调用才解题。

这是刻意设计的。

避免：

```text
视觉误读
    +
推理过程
```

同时发生，导致无法判断究竟是哪一步出了错。

如果 `uncertain_regions` 非空：

UI 必须显示：

```text
“我不完全确定图片中的这些内容，请确认。”
```

---

# 9. Phase 5 — 学生档案

SQLite 即可。

StudentProfile：

```text
id
name
grade
preferred_language
response_style
created_at
updated_at
```

KnowledgeState：

```text
student_id
subject
topic
mastery
confidence
mistake_count
last_seen
notes
```

例如：

```text
Math
└── Quadratic equations
    mastery = 0.62

Math
└── Fraction simplification
    mastery = 0.31
    common mistake:
        forgets denominator restrictions
```

注意：

不要让 LLM 每轮任意重写整个学生 profile。

采用：

```text
event → proposed update → validation → database
```

例如模型只能提出：

```json
{
  "topic": "quadratic_equations",
  "event": "correct_without_hint"
}
```

mastery 更新逻辑由代码决定。

---

# 10. Phase 6 — Conversation Memory

不能因为支持 128K 就无限塞聊天记录。

上下文分四层：

```text
System policy
Student profile
Relevant long-term memory
Recent conversation
```

长期对话采用摘要。

建议：

```text
最近 20～40 turns:
    原文

较老对话:
    structured summary
```

summary 存：

```text
topics covered
mistakes
concepts mastered
open questions
teacher observations
```

不要存一大段文学摘要。

使用结构化 JSON。

---

# 11. Phase 7 — 教材 RAG

这是后做，不是先做。

第一版 ingestion 支持：

```text
PDF
Markdown
TXT
```

教材 chunk 不使用机械固定 token 切割。

优先根据：

```text
chapter
section
heading
exercise
example
```

拆分。

metadata：

```text
book
grade
subject
chapter
section
page
```

搜索流程：

```text
query
 ↓
metadata filter
 ↓
vector retrieval
 ↓
optional reranker
 ↓
top documents
 ↓
LLM
```

RAG 内容必须和模型自身知识明显区分。

Prompt 中标记：

```text
REFERENCE MATERIAL
```

并要求：

```text
如果教材与自身知识发生冲突，指出冲突。
```

---

# 12. Phase 8 — Tutor Policy

建立单独的 policy layer。

不要把这些规则散落在 system prompt。

TutorMode：

```python
class TutorMode(Enum):
    TUTOR = "tutor"
    EXPLAIN = "explain"
    CHECK = "check"
    EXAM = "exam"
    PRACTICE = "practice"
```

Tutor 模式规则：

```text
1. 判断学生是否已经尝试。
2. 如果没有，优先给第一步提示。
3. 如果学生再次请求，再逐步增加提示。
4. 除非学生明确要求完整解答，否则不要马上展示完整答案。
5. 永远不要使用明显超出学生年级的技巧，除非说明。
```

例如：

初中生：

```text
禁止默认使用微积分
禁止偷偷使用矩阵
禁止使用复杂数技巧
```

高中低年级：

```text
使用知识范围必须受 grade 限制。
```

---

# 13. Prompt 设计原则

Prompt 文件全部放 `/prompts`。

禁止在 Python 代码里塞几千字字符串。

system prompt 保持短而硬。

主要原则：

```text
你是一名辅导老师，而不是答案生成器。

准确性高于流畅性。

若题目条件不确定，必须说明。

任何能够通过数学工具验证的最终结果都应优先验证。

不要声称调用过没有调用过的工具。

不要隐藏工具检测出的冲突。

根据学生年级选择解法。

不知道就明确说不知道。
```

禁止：

```text
“You are the smartest teacher...”
“Always answer perfectly...”
```

这些废话只浪费 context。

---

# 14. Eval Harness

这部分必须在项目早期完成。

建立：

```text
eval/datasets/
```

第一套自己收集至少：

```text
20 初中数学
20 高中数学
10 初中物理
10 高中物理
10 化学
10 英语
10 图片题
```

总计约 90 题。

每题保存：

```json
{
  "id": "...",
  "subject": "math",
  "grade": 9,
  "question": "...",
  "reference_answer": "...",
  "tags": ["quadratic"],
  "requires_image": false
}
```

至少统计：

```text
final answer correctness
tool verification success
hallucination
vision transcription error
unnecessary advanced method
latency
prompt tokens
completion tokens
```

不要只评价：

```text
“答案看起来不错”
```

---

# 15. 重点做 A/B Test

后续下载其它模型时，不靠感觉评价。

保持完全相同数据集比较：

```text
Qwen3.5-9B Q4
Qwen3.5-9B Q5
Qwen3.5-9B Q6

未来：
Qwen3.8-27B IQ3
```

比较：

```text
正确率
工具校验后的正确率
幻觉率
视觉识别
平均延迟
VRAM
tok/s
```

最终才能回答：

> 27B IQ3 是否真的值得牺牲速度和显存？

---

# 16. API 草案

## POST /api/chat

Request：

```json
{
  "student_id": "student-001",
  "conversation_id": "abc",
  "mode": "tutor",
  "message": "为什么这里要配方？"
}
```

Response 使用 SSE。

---

## POST /api/chat/image

multipart：

```text
student_id
conversation_id
mode
message
image
```

---

## GET /api/students/{id}

返回学生信息。

---

## GET /api/students/{id}/knowledge

返回当前知识掌握图谱。

---

## GET /health

返回：

```json
{
  "backend": "ok",
  "llama": "ok",
  "model": "...",
  "context_size": 65536
}
```

---

# 17. UI 第一版

只需要：

```text
左侧：
    conversation list

中间：
    chat

输入区域：
    textarea
    image upload
    Tutor / Explain / Check selector

右侧：
    当前学生
    年级
    当前知识点
    本轮工具调用
```

必须能展开：

```text
Tool trace
```

例如：

```text
✓ solve_equation
x² - 5x + 6 = 0
→ x = 2, 3
```

不需要展示 chain-of-thought。

只展示真实发生过的工具调用和验证结果。

---

# 18. Logging

所有请求生成 request_id。

日志记录：

```text
request_id
student_id
conversation_id
model
prompt_tokens
completion_tokens
latency
tools_used
verification status
```

不要默认记录完整学生消息到普通 log。

对话存数据库。

log 只记录 metadata。

---

# 19. Configuration

统一配置：

```yaml
llm:
  base_url: http://127.0.0.1:8080
  model: qwen3.5-9b
  timeout: 180

tutor:
  default_mode: tutor
  max_verification_attempts: 2

memory:
  recent_turns: 30

retrieval:
  enabled: false
  top_k: 6
```

允许环境变量覆盖。

---

# 20. 明确不做的东西

MVP 禁止实现：

```text
用户注册
OAuth
云同步
移动 App
multi-agent
agent swarm
MCP 大杂烩
Docker Swarm
Kubernetes
Redis
PostgreSQL
Kafka
GraphQL
自动上网搜索
自主 shell
无限 Python 执行
情绪识别
数字人
3D avatar
```

任何 Coding Agent 试图加入这些东西，应当拒绝。

---

# 21. 实施顺序

严格按以下顺序：

```text
1 llama.cpp integration
2 streaming chat
3 TutorEngine abstraction
4 SymPy tool system
5 verification loop
6 evaluation harness
7 multimodal transcription
8 student profiles
9 conversation memory
10 textbook RAG
11 practice generation
12 UI polish
```

RAG 排第 10 是故意的。

不要因为“AI 应用必须 RAG”就先做 RAG。

---

# 22. 第一阶段验收场景

必须通过以下真实场景。

### Case A

学生：

> 解 x² - 5x + 6 = 0

系统：

```text
能正确求解。
结果由 SymPy 验证。
```

### Case B

学生：

> 我算出来 x=2，3，对吗？

系统：

```text
验证答案，而不是重新长篇解题。
```

### Case C

学生：

> 为什么移项之后符号变了？

系统：

```text
根据年级解释等式两边执行相同操作。
不能只说“移项要变号”。
```

### Case D

学生上传模糊试卷照片。

系统：

```text
先展示识别的题目。
不确定数字明确标识。
不允许直接猜后继续计算。
```

### Case E

学生的答案错了。

系统：

```text
指出最早出现错误的步骤。
而不是重新生成标准答案。
```

---

# 23. Coding Agent 工作规则

DeepSeek/Copilot 在修改仓库时必须遵守：

1. 一次只完成一个 issue。
2. 修改前先读取相关代码。
3. 不主动重构无关模块。
4. 新功能必须添加测试。
5. API interface 修改必须同步更新测试和 README。
6. 禁止未经要求添加 dependency。
7. 禁止使用 `eval` / `exec` 执行模型生成内容。
8. 禁止静默 catch exception。
9. 不允许用 mock 掩盖 integration failure。
10. 每次完成任务后执行：

```text
pytest
ruff check
mypy
```

11. 如果需求和当前架构冲突，先指出冲突，不要偷偷改架构。
12. 保持模块简单；出现“FactoryManagerProvider”之类命名时重新考虑设计。

---

# 24. 给 Coding Agent 的第一个任务

执行以下任务，不做其它功能：

> 创建 Local Tutor 项目的 backend skeleton。
>
> 使用 Python 3.12、FastAPI、Pydantic v2、httpx 和 uv。
>
> 实现一个 `LlamaClient`，连接本地 llama.cpp OpenAI-compatible API。
>
> 实现：
>
> * `/health`
> * `/api/chat`
> * SSE streaming
> * config management
> * llama-server unavailable/error handling
>
> 创建相应单元测试。
>
> 暂时不要实现：
>
> * tools
> * database
> * authentication
> * RAG
> * student profiles
> * multimodal
>
> 代码必须符合本工程计划定义的 repo structure。
>
> 完成后输出：
>
> 1. 修改文件列表
> 2. 核心设计说明
> 3. 测试结果
> 4. 启动方式
> 5. 尚未解决的问题
>
> 不要继续进入下一阶段。

---

# 25. 第二个任务

第一阶段通过人工验收后：

> 实现 TutorEngine abstraction。
>
> Endpoint 不得直接调用 LlamaClient。
>
> 调用结构必须变成：
>
> `API → TutorEngine → LlamaClient`
>
> 实现 TutorMode：
>
> * tutor
> * explain
> * check
>
> Prompt 从 `/prompts` 加载。
>
> 添加覆盖三个模式的测试。
>
> 不实现 tools 或 RAG。

---

# 26. 第三个任务

> 实现受限数学 Tool Registry。
>
> 第一批工具：
>
> * calculator
> * solve_equation
> * simplify_expression
> * factor_expression
> * check_equivalence
>
> 使用 SymPy。
>
> 参数由 Pydantic 验证。
>
> 禁止 eval、exec、subprocess 和任意 Python execution。
>
> 为正常输入、恶意输入、非法 expression 和 timeout 添加测试。
>
> 完成后不要继续下一阶段。

---

# 27. 项目成功标准

这个项目成功与否，不看：

```text
UI 多漂亮
模型 benchmark 多高
Agent 有多少个
代码有多少行
```

而看：

```text
学生拿一道真实作业题过来，
系统正确理解了题目，
没看清的地方敢说没看清，
该计算的地方会自动验证，
能找到学生真正错的那一步，
解释符合他的年级，
第二天还记得他昨天哪里不会。
```

如果做到这些，9B 就已经不是“本地模型玩具”。

它是一个真正能用的私人家教系统。
