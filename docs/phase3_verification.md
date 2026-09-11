# Phase 3 验收记录 — 受限数学工具系统

> 计划依据：`ENGINEERING_PLAN.md` §7（权威）、§26（仅参考）、§23.7、§17
> 阶段清单：`work.md` Phase 3
> 执行日期：2026-09-11
> 环境：Python 3.12.14 / sympy 1.14.0 / pytest 8.x

---

## 1. 结论

§7 的**完整 8 个工具**已落地，全部经 `tools/registry.py` 统一注册与调度，参数由 Pydantic 严格校验，
输入必须穿过 `tools/sandbox.py` 的沙箱；每次调用都有结构化返回与可审计 trace。

```text
ToolCall(§7 信封) → ToolRegistry.call_request → Pydantic 校验 → 沙箱解析 → SymPy → ToolResult / ToolTrace
```

**本阶段不接线到 `TutorEngine`**：§7 的「LLM 解题 → 抽取 claim → SymPy 校验 → 冲突重检」是 Phase 4
（`tutor/verifier.py`）的工作，因此 `TutorResponse.tools_used` 目前仍为空。

---

## 2. DoD 逐条核对

| DoD（work.md Phase 3） | 状态 | 证据 |
|---|---|---|
| §7 列出的 8 个工具全部可用 | ✅ | `tests/test_tools_registry.py::TestEveryToolWorks`（8 项参数化）+ 每个工具自己的测试类 |
| 参数校验与错误处理风格一致 | ✅ | 所有参数模型继承同一个 `extra="forbid"` 基类；错误统一为 `ToolErrorKind`（`unknown_tool` / `invalid_arguments` / `rejected` / `unsupported` / `timeout` / `internal`），`TestToolContractConsistency` 逐工具核对 |
| 恶意输入被拒绝且不产生代码执行 | ✅ | `tests/test_tools_sandbox.py`（40+ 恶意输入）；`tests/test_skeleton.py::test_tool_modules_import_nothing_dangerous` 限定 tools 包只能 import 数学/标准库；§23.7 的 eval/exec/subprocess 全仓 AST 扫描继续通过 |
| 工具调用可被审计（有 trace） | ✅ | `ToolResult.as_event()` / `.summary()`、`ToolTrace.tools_used` / `.failures` / `.as_events()`（`TestToolTrace`、`TestStructuredCallEnvelope`） |
| 测试覆盖：正常 / 恶意 / 非法 expression / timeout | ✅ | `test_tools_calculator.py`、`test_tools_sandbox.py`、`test_tools_algebra.py`、`test_tools_registry.py::TestTimeout`（用刻意慢的 handler，不靠运气） |

质量门：

```text
uv run pytest                 → 481 passed, 9 skipped
uv run ruff check .           → All checks passed!
uv run ruff format --check .  → 59 files already formatted
uv run mypy                   → Success: no issues found in 56 source files
```

---

## 3. 八个工具与真实输出

真实运行（`build_default_registry()` + §7 信封调用）：

```text
catalog: 8 tools -> calculator, evaluate_expression, solve_equation, simplify_expression,
                    factor_expression, differentiate, integrate, check_equivalence

✓ calculator(expression=(3+5)*7/2)                 → result=28, decimal=28
✓ calculator(expression=1/3)                       → result=1/3, decimal=0.3333333333
✓ evaluate_expression(x^2 - 5*x, x=3)              → expression=x**2 - 5*x, result=-6
✓ solve_equation(x^2 - 5*x + 6 = 0)                → solutions=['2', '3'], count=2
✓ factor_expression(x^2 - 5*x + 6)                 → factored=(x - 3)*(x - 2)
✓ simplify_expression((x^2 - 1)/(x - 1))           → simplified=x + 1
✓ check_equivalence((x-2)*(x-3), x^2 - 5*x + 6)    → equivalent=True, difference=0
✓ differentiate(x^3 - 5*x, order=1)                → derivative=3*x**2 - 5
✓ integrate(x, 0..1)                               → value=1/2, decimal=0.5
✓ integrate(x^2)                                   → antiderivative=x**3/3, constant=C
```

设计约定：

* 数值结果**精确值 + 十进制**双写（`1/3` 与 `0.3333333333`），不把 `0.333333` 当成答案；
* 方程可以写 `x^2 - 5*x + 6` 或 `x^2 - 5*x + 6 = 0`（§7 的示例是前者，学生与模型两种写法都会出现）；
* SymPy 做不了就报 `unsupported`，**绝不**伪装成「无解」：`solve` 返回空列表时额外给
  `note=no solution found`，`integrate` 拿不到原函数时明确报错；
* 复数解（`x^2 + 1 = 0` → `['-I', 'I']`）如实返回，是否合适由 policy 层按年级决定。

## 4. 恶意 / 退化输入（实测）

```text
✗ calculator(__import__('os').system('id'))   → rejected: only calls to known maths functions are allowed
✗ solve_equation(x**)                          → rejected: not a valid maths expression: invalid syntax
✗ simplify_expression(9^9^9)                   → rejected: power too large: the result would have about 369693100 digits
✗ solve_equation(x = 1, variable="__class__")  → rejected: invalid variable name '__class__'
✗ calculator(1/0)                              → unsupported: the result is not a finite number: zoo
✓ solve_equation(x + 1 = x + 2)                → solutions=[], count=0, note="no solution found"
✗ integrate(exp(sin(x)))                       → unsupported: sympy could not find an antiderivative
✗ unknown_tool()                               → unknown_tool: unknown tool 'unknown_tool'; available: …
```

沙箱的四道闸门（`tools/sandbox.py`，全部先于 SymPy）：

1. 长度（500 字符）与 AST 节点数（200）上限；
2. AST 白名单：没有属性访问、下标、lambda、推导式、f-string、字符串/bytes 字面量、关键字参数、`*args`；
   函数调用只允许白名单里的数学函数（`factorial` 因无界输出被排除）；
3. 标识符策略：符号名是 ≤8 字符的 ASCII 标识符，`_` 开头与函数名一律拒绝；
4. 字面量幂的量级保护：`9**9**9` 在进入 SymPy 之前就被拒。

> SymPy 官方文档明确警告 `sympify` / `parse_expr` 内部会求值输入，这正是第 2 步必须先行、
> 且绝不让原始字符串直接进入解析器的原因。

## 5. 本阶段发现并修正的真实问题

### 5.1 `^` 在 SymPy 1.14 里不是乘方，而是异或（静默算错）

`standard_transformations` **不包含** `convert_xor`（本机实测），因此：

```text
parse_expr("x^2")  → TypeError: unsupported operand type(s) for ^
parse_expr("2^64") → 66        # 异或，不是 2 的 64 次方
parse_expr("9^9^9")→ 9
```

模型与学生都习惯写 `x^2`，静默算错是最糟的结果。处理：**在 AST 检查之前**把 `^` 规范化为 `**`
（`sandbox._CARET → _POWER`），这样既保证语义正确，又让量级保护看到真正的 `Pow` 节点。
`|`、`&`、`<<`、`>>`、`~` 仍然一律拒绝。

### 5.2 `float(sympy.Float("1e500"))` 返回 `inf` 而不是抛异常

原实现的兜底（`except OverflowError`）不会触发，会向模型返回 `decimal="inf"`。
现在显式检查 `math.isfinite`，非有限时回退到 `sympy.sstr`（`1.000000000e+500`）。

### 5.3 SymPy 没有类型信息

`import-untyped` 会让 `mypy --strict` 失败。§23.6 禁止为类型检查新增依赖，因此在
`pyproject.toml` 加了 `[[tool.mypy.overrides]] module = ["sympy", "sympy.*"]` +
`ignore_missing_imports`，并在边界上用显式 `str()` 收敛 `Any`。

## 6. 与计划的偏差（§23.11：先报告）

| 项 | 内容 | 理由 |
|---|---|---|
| 新增 `tools/sandbox.py` | §4 的 tools 目录只列 `registry/calculator/algebra/units` | 表达式沙箱是唯一的安全入口，塞进 `registry.py` 会同时承担调度与解析两件事；`retrieval/models.py` 已有「§4 之外新增模块」的先例 |
| `ToolCall` 信封模型 | §7 给出了信封格式但没指定实现位置 | 「结构化调用格式」属于工具契约本身；把模型自由文本转成信封仍留给 Phase 4 |
| `pyproject.toml` 增加 mypy override | 构建配置变更 | 见 §5.3，不引入任何新依赖 |
| `DEFAULT_TOOL_TIMEOUT = 5.0` 为模块常量 | §19 配置结构里没有超时项 | 它是保护阀不是推理参数；`ToolRegistry(default_timeout=...)` 已留出配置注入点，Phase 4 需要时可接 `config.yaml` |
| `units.py` 仍为占位 | §7 的 8 个工具里没有单位换算 | 按 work.md「不做无关模块」，留给后续阶段 |

## 7. 已知限制

1. **超时只能约束「等待」，不能杀死计算。** SymPy 没有取消机制，超时的调用会继续在自己的
   工作线程里跑（线程池上限 4）。表达式尺寸与幂量级两道前置闸门让这种情况很少发生；
   若 Phase 4 需要硬性终止，升级路径是进程池（`concurrent.futures.ProcessPoolExecutor`）。
2. **工具还没接进请求路径**，因此 Phase 3 不改变 `/api/chat` 的任何行为（`tools_used` 仍为 `[]`）。
3. `check_equivalence` 用 `simplify(a - b) == 0` 判定；对无法化简的超越等式可能给出
   `equivalent=False`（差异非零表达式）。Phase 4 若需要更强判定再评估 `equals()`。
4. 工具错误信息为英文短句（进日志与 Phase 4 的自检 prompt），面向学生的话术由 policy/response 层负责。
