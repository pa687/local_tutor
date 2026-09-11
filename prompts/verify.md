<!--
Re-check prompt. ENGINEERING_PLAN.md §7: "若冲突：让模型重新检查". It is sent when SymPy
contradicts the model's own answer, so the model gets one bounded chance to find its
mistake instead of the system silently shipping a wrong answer.

The concrete conflict is appended by the engine at call time; nothing about it is
hardcoded here.
-->
# 冲突修正任务

系统用 SymPy 逐点校验了你上一条解答，并发现了下面列出的冲突。这些结论是把你的解代回原方程算出来的，
所以它是事实，不是意见。

请按下面顺序处理：

1. 先找出**真正**出错的那一步（可能是列方程、变形、求根或最后取值中的任意一步）；
2. 给出**修正后的完整解答**，修正后的结果必须能通过同样的代入校验；
3. 如果冲突说明原题被理解错了（例如漏掉条件、看错符号），先说明题目应该怎么理解，再重做；
4. 与冲突无关且已经正确的部分可以简要带过，不要重复长篇推导；
5. 不要道歉，不要解释内部流程，也不要提到「SymPy」「工具」「校验」「系统」这些实现细节——
   用辅导老师的口吻说明解法哪里出了问题。

如果重新检查后确认原答案错了，直接给出正确答案；如果坚持原答案是对的，必须指出代入校验所用的
方程或代入方式哪里不成立。

