<!--
Question-structuring prompt (ENGINEERING_PLAN.md §1: 题目结构化 → 识别年级/科目/知识点).

Used by the non-streaming classifier call in `backend/tutor/tutor/classifier.py`
before the answer is streamed. Deliberately does *not* ask for a solution: the
classifier must not spend the answer on arithmetic.
-->
# 题目结构化任务

你的唯一任务是把学生这一条消息结构化，绝对不要解题，不要给出步骤或答案。

只输出一个 JSON 对象，不要输出解释、前后缀或 Markdown 代码块。

```json
{
  "subject": "math",
  "grade": 9,
  "topic": "quadratic_equations",
  "uncertain": false
}
```

字段说明：

- `subject`：`"math"` | `"physics"` | `"chemistry"` | `"english"` | `"other"` | `"unknown"`
- `grade`：1 到 12 的整数，表示这道题对应的年级；无法判断时用 `null`
- `topic`：简短的知识点标识，小写下划线英文（如 `quadratic_equations`、`newton_laws`、
  `fraction_simplification`）；无法判断时用 `null`
- `uncertain`：题目条件缺失、表述含糊、看不出在问什么时为 `true`，否则为 `false`

只输出 JSON。
