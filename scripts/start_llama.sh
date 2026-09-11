#!/usr/bin/env bash
# Launch llama-server for Local Tutor (ENGINEERING_PLAN.md §5).
#
# Every path and every inference parameter comes from the environment (see
# .env.example); nothing is hardcoded here. The defaults below are inference
# parameters only — the server binary and the model file must be provided.
set -euo pipefail

die() { printf '[start_llama] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[start_llama] %s\n' "$*" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091  # machine-local file, never committed
  source ./.env
  set +a
fi

: "${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN to .../llama.cpp/build/bin/llama-server}"
: "${LLAMA_MODEL_PATH:?set LLAMA_MODEL_PATH to the model .gguf}"

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-65536}"
LLAMA_KV_TYPE="${LLAMA_KV_TYPE:-q8_0}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-all}"
LLAMA_ALIAS="${LLAMA_ALIAS:-$(basename "${LLAMA_MODEL_PATH%.gguf}")}"
LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-logs/llama-server.log}"

[[ -x "$LLAMA_SERVER_BIN" ]] || die "not executable: $LLAMA_SERVER_BIN"
[[ -f "$LLAMA_MODEL_PATH" ]] || die "model file not found: $LLAMA_MODEL_PATH"

BIN_DIR="$(cd "$(dirname "$LLAMA_SERVER_BIN")" && pwd)"
# Current upstream layout: a thin launcher plus shared libraries in the same
# directory. Copying the launcher alone yields a binary that cannot start.
if ! find "$BIN_DIR" -maxdepth 1 \( -name 'libllama*.so' -o -name 'libggml*.so' \) \
  -print -quit | grep -q .; then
  die "no libllama*/libggml* shared libraries next to $LLAMA_SERVER_BIN; keep the whole build/bin tree together"
fi

if [[ "$LLAMA_GPU_LAYERS" == "all" ]]; then
  gpu_layers=999
else
  gpu_layers="$LLAMA_GPU_LAYERS"
fi

args=(
  -m "$LLAMA_MODEL_PATH"
  -a "$LLAMA_ALIAS"
  --host "$LLAMA_HOST"
  --port "$LLAMA_PORT"
  -c "$LLAMA_CONTEXT_SIZE"
  -ctk "$LLAMA_KV_TYPE"
  -ctv "$LLAMA_KV_TYPE"
  -np "$LLAMA_PARALLEL"
  -fa "$LLAMA_FLASH_ATTN"
  -ngl "$gpu_layers"
)
if [[ -n "${LLAMA_MMPROJ_PATH:-}" ]]; then
  args+=(--mmproj "$LLAMA_MMPROJ_PATH")
fi

mkdir -p "$(dirname "$LLAMA_LOG_FILE")"
log "binary : $LLAMA_SERVER_BIN"
log "model  : $LLAMA_MODEL_PATH (alias: $LLAMA_ALIAS)"
log "params : ctx=$LLAMA_CONTEXT_SIZE kv=$LLAMA_KV_TYPE parallel=$LLAMA_PARALLEL fa=$LLAMA_FLASH_ATTN ngl=$gpu_layers"
log "listen : http://$LLAMA_HOST:$LLAMA_PORT  log: $LLAMA_LOG_FILE"

LLAMA_PID_FILE="${LLAMA_PID_FILE:-}"
if [[ -n "$LLAMA_PID_FILE" ]]; then
  # Detached mode: record the real server PID so callers (benchmark_model.sh) can
  # stop exactly this process instead of guessing at process groups.
  "$LLAMA_SERVER_BIN" "${args[@]}" >"$LLAMA_LOG_FILE" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" >"$LLAMA_PID_FILE"
  log "pid    : $server_pid (pid file: $LLAMA_PID_FILE)"
  wait "$server_pid"
else
  "$LLAMA_SERVER_BIN" "${args[@]}" 2>&1 | tee -a "$LLAMA_LOG_FILE"
fi
