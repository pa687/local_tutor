# llama-server benchmarks — Local Tutor (ENGINEERING_PLAN.md §5)

生成方式：`BENCH_CONTEXTS="65536 98304 131072" BENCH_PROMPT_SIZES="8192 32768 target" scripts/benchmark_model.sh`
（每次运行都会重新启动/停止自己的 llama-server，并追加到本文件；`target` = `-c` 减 256 token 的模板余量。）

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 5070 Ti 16 GB（空闲占用 ~1.1 GB） |
| 模型 | `Qwen_Qwen3.5-9B-Q4_K_M.gguf`（alias `qwen3.5-9b`） |
| 后端 | `~/llama.cpp/build/bin/llama-server`，CUDA `120a-real`，`-ngl 999` |
| 参数 | `-c {65536,98304,131072}`、`-ctk/-ctv q8_0`、`-np 1`、`-fa on` |
| 测量 | `scripts/bench_llama.py`：`cache_prompt=false`，`n_predict=1` 量 prompt，`n_predict=128` 量生成与 TTFT |

## 结论（2026-09-11）

- **64K 是 Phase 1 默认**：显存 8.0 GB（含 1.1 GB 基线），生成 115.8 tok/s，TTFT 0.64 s，65290 token 的提示 18 s 处理完。
- **96K / 128K 也能跑**：显存 8.7 GB / 9.4 GB，生成速度几乎不变（116.4 / 118.1 tok/s），代价是 TTFT 与 prompt eval 时间线性增长（64K→128K 时 prompt 吞吐 3706→2764 tok/s，128K 提示需 47 s）。
- 显存随上下文近似线性：每 32K 上下文约 +0.7 GB（q8_0 KV）。16 GB 卡上有充足余量做多模态（Phase 6）。
- 提示吞吐 2.8k–4.8k tok/s，远高于生成速度，说明瓶颈在解码而不是预填充。

## 指标明细

## context 65536

| metric | value |
|---|---|
| VRAM idle (before start) | 1173 MiB |
| VRAM after model load | 8009 MiB |
| prompt 8192 (8202 tokens, 2 s wall) | 4810.0 tok/s |
| VRAM after prompt 8192 | 8151 MiB |
| prompt 32768 (32778 tokens, 7 s wall) | 4491.0 tok/s |
| VRAM after prompt 32768 | 8089 MiB |
| prompt target (-256) (65290 tokens, 18 s wall) | 3706.4 tok/s |
| VRAM after prompt target (-256) | 8084 MiB |
| generation (128 tokens) | 115.8 tok/s |
| time-to-first-token | 636 ms |
| VRAM after generation | 8084 MiB |
| llama-server reports | ctx=65536 model=qwen3.5-9b |

Largest measured prompt: 65290 tokens at 3706.4 tok/s.


## context 98304

| metric | value |
|---|---|
| VRAM idle (before start) | 1130 MiB |
| VRAM after model load | 8680 MiB |
| prompt 8192 (8202 tokens, 2 s wall) | 4835.7 tok/s |
| VRAM after prompt 8192 | 8788 MiB |
| prompt 32768 (32778 tokens, 7 s wall) | 4493.9 tok/s |
| VRAM after prompt 32768 | 8807 MiB |
| prompt target (-256) (98058 tokens, 31 s wall) | 3149.2 tok/s |
| VRAM after prompt target (-256) | 8802 MiB |
| generation (128 tokens) | 116.4 tok/s |
| time-to-first-token | 880 ms |
| VRAM after generation | 8802 MiB |
| llama-server reports | ctx=98304 model=qwen3.5-9b |

Largest measured prompt: 98058 tokens at 3149.2 tok/s.


## context 131072

| metric | value |
|---|---|
| VRAM idle (before start) | 1144 MiB |
| VRAM after model load | 9398 MiB |
| prompt 8192 (8202 tokens, 2 s wall) | 4822.4 tok/s |
| VRAM after prompt 8192 | 9506 MiB |
| prompt 32768 (32778 tokens, 7 s wall) | 4487.4 tok/s |
| VRAM after prompt 32768 | 9506 MiB |
| prompt target (-256) (130826 tokens, 47 s wall) | 2763.6 tok/s |
| VRAM after prompt target (-256) | 9632 MiB |
| generation (128 tokens) | 118.1 tok/s |
| time-to-first-token | 1146 ms |
| VRAM after generation | 9592 MiB |
| llama-server reports | ctx=131072 model=qwen3.5-9b |

Largest measured prompt: 130826 tokens at 2763.6 tok/s.

