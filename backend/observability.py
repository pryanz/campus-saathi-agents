"""Small, redacted metrics sink for outbound Groq requests."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_WRITE_LOCK = threading.Lock()


def _usage_value(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    return value if isinstance(value, int) else None


def _usage_dict(usage: Any) -> dict[str, int | None]:
    return {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def _metrics_path() -> Path:
    return Path(os.getenv("GROQ_METRICS_LOG", "logs/groq_calls.jsonl"))


def _write_event(event: dict[str, Any]) -> None:
    path = _metrics_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, separators=(",", ":")) + "\n")
    except OSError:
        # Observability must never take down an otherwise successful request.
        pass


def groq_call(
    call_type: str,
    request: Callable[[], Any],
    *,
    model: str,
) -> Any:
    """Execute and record one Groq call without recording prompt contents."""
    started = time.perf_counter()
    event: dict[str, Any] = {
        "event": "groq_call",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "call_type": call_type,
        "model": model,
    }
    try:
        response = request()
    except Exception as exc:
        event.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        _write_event(event)
        raise

    latency_seconds = time.perf_counter() - started
    usage = _usage_dict(getattr(response, "usage", None))
    completion_tokens = usage["completion_tokens"]
    event.update(
        {
            "ok": True,
            "latency_ms": round(latency_seconds * 1000, 2),
            "usage": usage,
            "output_tokens_per_second": (
                round(completion_tokens / latency_seconds, 3)
                if completion_tokens is not None and latency_seconds > 0
                else None
            ),
        }
    )
    _write_event(event)
    return response
