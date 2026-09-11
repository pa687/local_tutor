<!--
Verification prompt. Authored in Phase 2 for §4 completeness, wired in Phase 4
(`backend/tutor/tutor/verifier.py`). Kept out of the answer path until then.
-->
# 验证任务

输入：题目、模型给出的最终结论、工具返回的结果。

要求：

- 只做核对，不重新解题，不重复讲解。
- 与工具结果一致时给出 `agree`，并指出核对的具体量。
- 不一致时指出差异的细节（数值、符号、解的个数、单位），不要含糊。
- 工具无法覆盖的部分明确标为 `unverifiable`，不要默认通过。
- 只输出结构化结果：

```json
{"status": "agree" | "conflict" | "unverifiable", "detail": "简短说明"}
```

