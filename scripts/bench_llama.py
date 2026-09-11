#!/usr/bin/env python3
"""Benchmark helper for llama-server (ENGINEERING_PLAN.md §5).

Modes (one per invocation, so the shell wrapper stays readable):

* ``--vram``       print GPU memory currently used, in MiB
* ``--wait``       block until llama-server answers ``/health``
* ``--wait-down``  block until llama-server stops answering
* ``--measure``    record VRAM / prompt tok-s / generation tok-s / TTFT, print markdown

It is driven by ``scripts/benchmark_model.sh``, which starts and stops the server.
Stdlib only, so it also runs outside the project virtualenv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

FILLER_WORD = "a"
HEALTH_POLL_SECONDS = 2.0


def gpu_used_mib() -> int | None:
    """Current GPU memory usage, or ``None`` when nvidia-smi is unavailable."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(completed.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def get_json(url: str, timeout: float) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None


def post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"llama-server answered HTTP {exc.code}: {detail}") from exc


def is_up(base_url: str) -> bool:
    return get_json(f"{base_url}/health", timeout=3.0) is not None


def wait_for_health(base_url: str, timeout: float, *, up: bool) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        healthy = get_json(f"{base_url}/health", timeout=5.0) is not None
        if healthy is up:
            return 0
        time.sleep(HEALTH_POLL_SECONDS)
    print(f"timed out waiting for llama-server to be {'up' if up else 'down'}", file=sys.stderr)
    return 1


def filler_prompt(target_tokens: int) -> str:
    """A prompt of roughly ``target_tokens`` tokens (the actual count is reported)."""
    return " ".join([FILLER_WORD] * target_tokens)


def prompt_benchmark(base_url: str, target_tokens: int, timeout: float) -> dict[str, object]:
    """Send a prompt-only request and return llama-server's own timings."""
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": filler_prompt(target_tokens)}],
        # 1, not 0: llama-server coerces 0 to 1 anyway, and this keeps prompt timing
        # independent of how long the answer would be.
        "n_predict": 1,
        "cache_prompt": False,
        "temperature": 0,
        "stream": False,
    }
    return post_json(f"{base_url}/v1/chat/completions", payload, timeout)


def generation_benchmark(
    base_url: str, tokens: int, timeout: float
) -> tuple[dict[str, object], float]:
    """Generate ``tokens`` new tokens (cache disabled) and time the first one."""
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": "Counting: one, two"}],
        "n_predict": tokens,
        "cache_prompt": False,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(  # noqa: S310
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    first_byte_at: float | None = None
    usage: dict[str, object] = {}
    timings: dict[str, object] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        for raw_line in response:
            if first_byte_at is None:
                first_byte_at = time.perf_counter()
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if first_token_at is None and delta.get("content"):
                first_token_at = time.perf_counter()
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
    seen_at = first_token_at or first_byte_at
    ttft_ms = (seen_at - started) * 1000 if seen_at else float("nan")
    result = {"usage": usage, "timings": timings}
    return result, ttft_ms


def fmt(value: object, digits: int = 1) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "n/a"


def measure(args: argparse.Namespace) -> int:
    props = get_json(f"{args.base_url}/props", timeout=30.0) or {}
    after_load = gpu_used_mib()
    rows: list[tuple[str, str]] = [
        ("VRAM idle (before start)", f"{args.idle_vram} MiB"),
        ("VRAM after model load", f"{after_load if after_load is not None else 'n/a'} MiB"),
    ]

    actual_prompt_tokens = 0
    largest_prompt_speed: float | None = None
    for size in args.prompt_sizes:
        if size == "target":
            # Chat-template and special tokens need room, so target the window minus
            # a safety margin instead of exactly n_ctx.
            target = max(1, args.context_size - args.template_margin)
            label = f"target (-{args.template_margin})"
        else:
            target = int(size)
            label = size
        if target > args.context_size:
            rows.append((f"prompt {label}", "skipped (larger than -c)"))
            continue
        started = time.perf_counter()
        result = prompt_benchmark(args.base_url, target, args.request_timeout)
        timings = result.get("timings") or {}
        usage = result.get("usage") or {}
        actual = int(usage.get("prompt_tokens") or 0)
        speed = timings.get("prompt_per_second")
        rows.append(
            (
                f"prompt {label} ({actual} tokens, {fmt(time.perf_counter() - started, 0)} s wall)",
                f"{fmt(speed)} tok/s",
            )
        )
        rows.append((f"VRAM after prompt {label}", f"{gpu_used_mib()} MiB"))
        if actual >= actual_prompt_tokens:
            actual_prompt_tokens = actual
            if isinstance(speed, int | float):
                largest_prompt_speed = float(speed)

    result, ttft_ms = generation_benchmark(args.base_url, args.gen_tokens, args.request_timeout)
    gen_timings = result.get("timings") or {}
    rows.append(
        (
            f"generation ({args.gen_tokens} tokens)",
            f"{fmt(gen_timings.get('predicted_per_second'))} tok/s",
        )
    )
    rows.append(("time-to-first-token", f"{fmt(ttft_ms, 0)} ms"))
    rows.append(("VRAM after generation", f"{gpu_used_mib()} MiB"))

    server_ctx = None
    settings = props.get("default_generation_settings") if isinstance(props, dict) else None
    if isinstance(settings, dict):
        server_ctx = settings.get("n_ctx")
    rows.append(("llama-server reports", f"ctx={server_ctx} model={props.get('model_alias')}"))

    print(f"\n## context {args.context_size}\n")
    print("| metric | value |")
    print("|---|---|")
    for name, value in rows:
        print(f"| {name} | {value} |")
    print()
    print(
        f"Largest measured prompt: {actual_prompt_tokens} tokens at "
        f"{fmt(largest_prompt_speed) if largest_prompt_speed is not None else 'n/a'} tok/s.\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--vram", action="store_true", help="print current GPU memory used")
    mode.add_argument("--is-up", action="store_true", help="exit 0 when llama-server answers")
    mode.add_argument("--wait", action="store_true", help="wait until llama-server is healthy")
    mode.add_argument("--wait-down", action="store_true", help="wait until llama-server is gone")
    mode.add_argument("--measure", action="store_true", help="run the measurement battery")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=300.0, help="health wait timeout")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--context-size", type=int, default=65536)
    parser.add_argument("--prompt-sizes", nargs="*", default=["8192", "target"])
    parser.add_argument(
        "--template-margin",
        type=int,
        default=256,
        help="tokens kept free for the chat template when measuring the target prompt",
    )
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--idle-vram", default="n/a")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.vram:
        used = gpu_used_mib()
        print(used if used is not None else "n/a")
        return 0
    if args.is_up:
        return 0 if is_up(args.base_url) else 1
    if args.wait:
        return wait_for_health(args.base_url, args.timeout, up=True)
    if args.wait_down:
        return wait_for_health(args.base_url, args.timeout, up=False)
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
