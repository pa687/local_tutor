# Phase 1 验收记录 — llama.cpp 基础设施 + 流式聊天

日期：2026-09-11　环境：RTX 5070 Ti 16 GB / Ubuntu 26.04 / `~/llama.cpp` CUDA 构建

## DoD 逐条核对（计划 §5）

| DoD 条目 | 结果 | 证据 |
|---|---|---|
| `GET /health` 能确认 llama-server 在线 | ✅ | 在线：`{"backend":"ok","llama":"ok","model":"qwen3.5-9b","context_size":65536}`；离线：`llama":"down"`，HTTP 仍 200 |
| 能进行普通文本聊天 | ✅ | `POST /api/chat` 走 `/v1/chat/completions`，集成测试 `test_plain_chat_streams_real_tokens` |
| 能 streaming 输出 token | ✅ | SSE 事件序列 `start → token… → done`；`test_answer_arrives_as_more_than_one_token_event` 断言 >1 个 token 事件 |
| llama-server 重启后应用能自动恢复 | ✅ | 手工演练（下方记录）：后端进程 PID 不变，`down→ok`、503→200 |
| LLM HTTP 超时有明确异常 | ✅ | `LlamaTimeoutError` → HTTP 504（`test_timeout_returns_504`）；单元测试 `test_read_timeout_raises_timeout` |
| OOM / llama-server crash 不导致 backend crash | ✅ | 连接失败 → `LlamaUnavailableError` → HTTP 503（`test_unreachable_llama_returns_503_not_a_crash`）；流中断 → `event: error`（`test_failure_mid_stream_becomes_an_error_event`） |
| 模型参数能从 config 修改 | ✅ | `config.yaml` 的 `llm.base_url/model/timeout` 经 `AppConfig` 注入 `LlamaClient`；`test_health_reflects_injected_config` |

## 重启恢复演练（2026-09-11 22:53–22:54）

```text
1. curl /health                     → {"backend":"ok","llama":"ok","model":"qwen3.5-9b","context_size":65536}
2. curl -N -X POST /api/chat        → event: start / event: token {"text":"ready"} / event: done
3. pkill -f 'build/bin/llama-server'（模拟 crash/OOM 被杀）
4. curl /health                     → {"backend":"ok","llama":"down",...,"context_size":null}
5. curl -X POST /api/chat           → HTTP 503
   {"detail":"llama-server unavailable: llama-server unreachable: All connection attempts failed"}
6. curl /health                     → HTTP 200（后端没有崩）
7. 重新 scripts/start_llama.sh
8. curl /health                     → {"backend":"ok","llama":"ok",...,"context_size":65536}
9. curl -N -X POST /api/chat        → event: start / token "re" / token "covered" / event: done
   uvicorn PID 210754 全程未变 → “无需重启应用即自动恢复”
```

## 实测数据

见 [`llama_benchmarks.md`](llama_benchmarks.md)（64K / 96K / 128K 的显存、提示与生成吞吐、TTFT）。

## 已知限制 / 遗留

1. **Phase 1 的 `/api/chat` 直接把学生消息发给模型**，没有 policy 层、没有 prompt 加载 —— 这是 Phase 2（§6、§12）的职责，代码里已标注 `TODO(Phase 2)`。
2. `prompt_tokens` / `completion_tokens` 依赖 llama-server 的 `stream_options.include_usage`；若上游移除该支持，代码会回退到 llama.cpp 的 `timings` 字段，两者都缺失时字段为 `null`。
3. 工作区只有 `Q5_K_M` 计划的 Q4_K_M 量化（计划 §5 建议首批用 `Q5_K_M`）；量化对比留到 Phase 12。
4. 未启用 `--mmproj`（多模态在 Phase 6）。
