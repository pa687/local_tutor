<!--
Long-conversation summarisation prompt. Authored in Phase 2 for §4 completeness,
wired in Phase 8 (`backend/tutor/memory/summarizer.py`). §10 forbids literary prose:
the summary must be structured JSON.
-->
# 对话摘要任务

把下面这段较老的对话压缩成结构化 JSON，不要写成段落式摘要。

```json
{
  "topics_covered": [],
  "mistakes": [],
  "concepts_mastered": [],
  "open_questions": [],
  "teacher_observations": []
}
```

要求：

- 每条不超过 20 字，只保留对后续教学有用的信息。
- 没有内容就用空数组，不要补空话。
- 不记录学生的隐私信息，只记录学习相关的事实。
- 只输出 JSON。
