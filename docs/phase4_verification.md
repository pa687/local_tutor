# Phase 4 验收记录 — 数学可靠性闭环

> 计划依据：`ENGINEERING_PLAN.md` §7（权威）、§22 场景 A/B/E、§17、§18
> 阶段清单：`work.md` Phase 4
> 执行日期：2026-09-12
> 环境：Python 3.12.14 / sympy 1.14.0；`llama-server` b9010-1904 + `Qwen3.5-9B-Q4_K_M`（`llm.enable_thinking=false`）

---

## 1. 结论

§7 的可靠性闭环已落地：**答案生成后被工具校验，冲突则让模型重检，重检结果再次校验**，
校验结论、自修正次数、工具调用轨迹都进入 API 响应与日志。

```text
CHECK 模式：学生作答 ──(先校验)──┐
                                ├→ 模型解答 ──(后校验)──→ 冲突? ──是──→ 让模型重检（≤2 次）──→ 再校验
题目方程 ────────────────────┘                              └──否──→ 定稿
```

判据是 **SymPy，不是第二个模型**：自我评判的 LLM 有可能认同自己的错误，而 §7 要求「结果由 SymPy 验证」。

## 2. DoD 逐条核对

| DoD（work.md Phase 4） | 状态 | 证据 |
|---|---|---|
| 场景 A：解 `x²-5x+6=0` 答案正确且经 SymPy 验证 | ✅ | `tests/test_integration_llama.py::test_case_a_answer_is_verified_by_sympy`；实测 `status=verified`，`detail="x = 2、x = 3 代入原方程均成立"` |
| 场景 B：学生给出 `x=2,3` → 验证学生答案而非重新长篇解题 | ✅ | `::test_case_b_student_answer_is_verified_by_the_tools`；实测 `student_status=verified` |
| 场景 E：学生答错 → 指出最早出错步骤 | ✅ | `::test_case_e_wrong_student_work_is_caught_by_the_tools`；实测 `student_status=conflict`，`detail="x = -3 代入后方程两边的差为 30"`，模型据此指出因式分解那一步错了 |
| 冲突可见、warning 可传递到 API 响应 | ✅ | `CONFLICT_WARNING` / `UNVERIFIABLE_WARNING` 进入 `TutorResponse.warnings`；`done.verification` 事件带 `status`/`attempts`/`detail`/`checks`/`tools` |
| self-correction 最大次数 = `tutor.max_verification_attempts` | ✅ | `tests/test_engine.py::TestVerificationLoop::test_a_persistent_conflict_stops_after_the_configured_attempts` |
| 不得隐藏冲突、不得声称调用过未调用的工具 | ✅ | 冲突一律产生 warning + `confidence=low`；`tools` 只包含真实发生的调用（`ToolResult.as_event()`），无调用即空数组 |

质量门：

```text
uv run pytest                → 570 passed, 12 skipped
uv run pytest -m integration → 12 passed（真实模型，47 s）
uv run ruff check . / ruff format --check . / uv run mypy → 全部通过
```

## 3. 真实模型验收（2026-09-12）

### 3.1 §22 场景

```text
### A quadratic [explain]  解方程 x^2 - 5*x + 6 = 0
verification: status=verified attempts=0
  detail: x = 2、x = 3 代入原方程均成立
  warnings: []

### harder: fractions [explain]  解方程 6*x^2 - 5*x - 6 = 0
verification: status=verified attempts=0
  detail: x = 3/2、x = -2/3 代入原方程均成立

### harder: irrational [explain]  解方程 x^2 - 2*x - 1 = 0
verification: status=verified attempts=0
  detail: x = 1 + sqrt(2)、x = 1 - sqrt(2) 代入原方程均成立

### E wrong student [check]
Q: 题目是 x^2 - 5*x + 6 = 0，我的过程：(x - 2)(x + 3) = 0，所以 x = 2 或 x = -3
verification: status=verified attempts=0
  student: conflict  x = -3 代入后方程两边的差为 30
  answer: 「你的第一步（因式分解）错误：你写的是 (x-2)(x+3) = 0 …… 展开是 x^2 + x - 6，这显然不等于题目中的 x^2 - 5x + 6」
```

### 3.2 自修正闭环（真实触发过一次）

对 `6*x^2 - 5*x - 6 = 0`，模型首答把解写成 `x = 3、x = -2`（错）：

```text
event: revision #1  detail: x = 3 代入后方程两边的差为 33；x = -2 代入后方程两边的差为 28
event: revision #2  detail: （同上，重检仍未一次改对）
done: status=conflict attempts=2 confidence=low
      warnings: ["工具校验发现答案与题目冲突：x = 3 代入后方程两边的差为 33；x = -2 代入后方程两边的差为 28（已尝试 2 次自我修正，仍未通过）"]
```

模型最终承认错误并给出正确根（`x = 3/2, -2/3`，再次校验通过）。**冲突没有被隐藏**，且
自修正次数严格受 `tutor.max_verification_attempts` 限制。

### 3.3 SSE 契约（新增 `revision` 事件）

```text
event: start     {"request_id","model","mode","subject","estimated_grade","topic","confidence","tools_used","warnings"}
event: token     {"text":"…"}                       # 首答
event: revision  {"attempt":1,"detail":"x = 4 代入后方程两边的差为 …"}   # 客户端应丢弃已缓冲文本
event: token     {"text":"…"}                       # 修正后的完整解答
event: done      {…,"tools_used":["evaluate_expression"],
                  "verification":{"status":"verified","attempts":1,"detail":"…",
                                  "claims":[…],"checks":[…],"tools":[…],
                                  "student_status":null,"student_detail":null}}
```

`verification.tools` 是**真实发生过的**调用（含失败的），可直接用于 §17 的 tool trace 展开。

## 4. 设计要点

1. **claim 抽取是确定性的、保守的**（`tutor/verifier.py`）：
   - 只校验「已知方程的唯一未知量」的闭式解（`x = 2 或 x = 3`、`x₁ = 3/2`、`x = 1 ± √2`、`\sqrt{2}`、`x²`、`−`、`×` 都能读）；
   - 讨论式/试探式语句不算 claim（`当 x = 2 时`、`x = 2 代入后`、`试 x = 1，x = 0`）；
   - 否定语句不算 claim（`x = 3 不是解`）；
   - **结论段落决定 verdict**：最后一段「给出答案」的段落才是判据，更早的语句仍进入审计轨迹——
     否则模型自述「我一开始算成了 x = 3」会被当成它在主张 `x = 3`，进而产生幻觉式冲突。
2. **四种状态语义**（对应 §7 的「仍然无法验证」）：
   `verified` / `conflict` / `unverifiable`（工具无法判定 → `confidence=low` + warning）/ `nothing_to_verify`
   （无 claim 或无参照方程 → **不降级**，否则概念性讲解会被无辜扣分）。
3. **先校验学生、再校验模型**：CHECK 模式先对学生给出的解做工具校验，结论作为独立 context 层
   注入（`ContextBuilder(verification_block=…)`），使 §22 B/E 的「指出最早出错步骤」有工具依据。
4. **§18 日志字段落地**：`verification_status` 不再恒为 null，`tools_used` 记录真实工具名。
5. **token 计数按请求累计**（多次 pass 求和），因为日志关心的是这次请求的整体开销。

## 5. 过程中发现并修正的真实问题（都是跑真实模型才暴露的）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 学生的错解被「自己的因式分解」验证通过：`(x-2)(x+3)=0` 自洽，代入 `x=-3` 成立 → 误判 verified | 参照方程改为**取最靠前的合法条件方程**（题目在过程之前） |
| 2 | **恒等式**被当作参照方程：`(x-2)(x-3) = x^2-5x+6` 对任意 x 成立 → 错误解也「成立」 | 参照方程必须**非恒等**（`simplify(lhs-rhs) != 0`） |
| 3 | 修正后的答案里**引用了刚被否定的值**（`x = 3 不是解`）→ 反复报冲突、白跑两次重检 | 否定语句不计 claim；**结论段落**决定 verdict |
| 4 | 多行结论只判最后一行（`x₁ = 1` / `x₂ = 10/11`）→ 漏判 | 结论以**段落**为单位，整段都算判据 |
| 5 | 只给 `x^2`、`x₁`、`√2`、`1 ± √2`、`5x`、`−3`、`×` 时完全抽不到 claim（中文数学书写习惯） | 归一化 Unicode 上下标 / 根号 / 隐含乘号 / LaTeX `\frac`、`\sqrt`、`^{}`、`_{}` |
| 6 | 列表式试探（`试 x = 1，x = 0，x = 2`）只有第一项被 cue 抑制，后两项漏进来 | cue 检查覆盖「赋值之前的整句前缀」 |

## 6. 与计划的偏差（§23.11：先报告）

| 项 | 内容 | 理由 |
|---|---|---|
| claim 抽取为确定性实现，非 LLM 调用 | §7 只要求「抽取可验证 claim」，未指定手段 | 判据必须是 SymPy；用 LLM 抽取会给校验器自身引入幻觉与额外延迟。代价是覆盖面（见 §7） |
| `prompts/verify.md` 改为「冲突修正」提示 | Phase 2 时它是「judge」提示 | 判定不再由模型做，该文件改作 §7「让模型重新检查」的指令 |
| `nothing_to_verify` 不降级 confidence | §7 只说「仍然无法验证 → low」 | 「试过但没法判」才降级；「本来就没有可校验内容」不应惩罚概念性讲解 |
| 非数学科目不做 SymPy 校验 | §7 面向数学 | 未经校验的科目保持 Phase 2 语义，避免误判（如物理公式里的多未知量） |
| `TutorEngine.__init__` 增加 `verifier` 注入点 | 便于测试注入失败/慢速工具 | 与既有 `policy`/`classifier`/`context` 注入风格一致 |

## 7. 已知限制

1. **覆盖面**：只校验「单未知量 + 闭式解」。应用题、物理量、含参讨论、方程组都归为 `nothing_to_verify`（不降级、不误报）。
   Phase 5 的 eval harness 应统计这一比例，再决定是否扩展 claim 词汇。
2. **同段叙述**：若模型在同一段里既叙述错误又给结论（无空行分隔），仍可能被误判为冲突——
   方向是保守的（报冲突 + low + 至多 2 次重检），不会把错的答案说成 verified。
3. **重检提示依赖模型配合**：实测 9B 模型有时两次重检都没改对，此时如实给出 `conflict` + warning，
   不做无限重试（§7 明确禁止死循环）。
4. **工具超时只能约束等待**（Phase 3 遗留）：SymPy 无法取消。
