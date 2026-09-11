#!/usr/bin/env bash
# VRAM + throughput benchmark for llama-server (ENGINEERING_PLAN.md §5).
#
# For every context size in BENCH_CONTEXTS this script records idle VRAM, starts
# llama-server through scripts/start_llama.sh, measures VRAM after load and after
# prompts of increasing size, measures prompt tok/s, generation tok/s and
# time-to-first-token, then stops that exact server process and appends a report to
# docs/llama_benchmarks.md.
#
# Usage:
#   BENCH_CONTEXTS="65536" BENCH_PROMPT_SIZES="8192 32768 target" scripts/benchmark_model.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f .env ]] && { set -a; source ./.env; set +a; }

BENCH_CONTEXTS="${BENCH_CONTEXTS:-65536}"
BENCH_PROMPT_SIZES="${BENCH_PROMPT_SIZES:-8192 32768 target}"
BENCH_GEN_TOKENS="${BENCH_GEN_TOKENS:-128}"
BENCH_REPORT="${BENCH_REPORT:-docs/llama_benchmarks.md}"
BENCH_BASE_URL="${BENCH_BASE_URL:-http://${LLAMA_HOST:-127.0.0.1}:${LLAMA_PORT:-8080}}"

PYTHON_BIN="${BENCH_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

mkdir -p "$(dirname "$BENCH_REPORT")" logs
if [[ ! -f "$BENCH_REPORT" ]]; then
  printf '# llama-server benchmarks — Local Tutor (ENGINEERING_PLAN.md §5)\n' >"$BENCH_REPORT"
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for ctx in $BENCH_CONTEXTS; do
  printf '=== context %s ===\n' "$ctx"

  if "$PYTHON_BIN" scripts/bench_llama.py --is-up --base-url "$BENCH_BASE_URL" >/dev/null; then
    printf '[benchmark] ERROR: %s already answers a request; stop that server first\n' \
      "$BENCH_BASE_URL" >&2
    exit 1
  fi

  idle_vram="$("$PYTHON_BIN" scripts/bench_llama.py --vram || echo n/a)"
  printf '[benchmark] idle VRAM: %s MiB\n' "$idle_vram"

  pid_file="logs/llama-$ctx.pid"
  log_file="logs/llama-$ctx.log"
  rm -f "$pid_file"
  LLAMA_CONTEXT_SIZE="$ctx" LLAMA_LOG_FILE="$log_file" LLAMA_PID_FILE="$pid_file" \
    scripts/start_llama.sh >/dev/null 2>&1 &
  launcher_pid=$!

  "$PYTHON_BIN" scripts/bench_llama.py --wait --base-url "$BENCH_BASE_URL" --timeout 900
  server_pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
    printf '[benchmark] ERROR: llama-server did not stay up; see %s\n' "$log_file" >&2
    exit 1
  fi
  printf '[benchmark] llama-server pid %s\n' "$server_pid"

  if ! "$PYTHON_BIN" scripts/bench_llama.py \
    --measure \
    --base-url "$BENCH_BASE_URL" \
    --context-size "$ctx" \
    --prompt-sizes $BENCH_PROMPT_SIZES \
    --gen-tokens "$BENCH_GEN_TOKENS" \
    --idle-vram "$idle_vram" | tee -a "$BENCH_REPORT"; then
    printf '[benchmark] WARNING: measurements failed for context %s (see %s)\n' \
      "$ctx" "$log_file" >&2
    printf '\n## context %s\n\nFAILED to measure — see `logs/llama-%s.log`\n' "$ctx" "$ctx" \
      >>"$BENCH_REPORT"
  fi

  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$launcher_pid" 2>/dev/null || true
  server_pid=""
  "$PYTHON_BIN" scripts/bench_llama.py --wait-down --base-url "$BENCH_BASE_URL" --timeout 120
done

printf '[benchmark] report: %s\n' "$BENCH_REPORT"
