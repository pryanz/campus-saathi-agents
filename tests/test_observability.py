import json
from types import SimpleNamespace

import pytest

from backend.observability import groq_call


def test_groq_call_records_latency_and_usage(tmp_path, monkeypatch):
    log_path = tmp_path / "groq.jsonl"
    monkeypatch.setenv("GROQ_METRICS_LOG", str(log_path))
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20)
    )

    assert groq_call("test", lambda: response, model="model") is response

    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert event["ok"] is True
    assert event["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }
    assert event["latency_ms"] >= 0
    assert event["output_tokens_per_second"] > 0


def test_groq_call_records_error_and_reraises(tmp_path, monkeypatch):
    log_path = tmp_path / "groq.jsonl"
    monkeypatch.setenv("GROQ_METRICS_LOG", str(log_path))

    with pytest.raises(RuntimeError, match="boom"):
        groq_call("test", lambda: (_ for _ in ()).throw(RuntimeError("boom")), model="model")

    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert event["ok"] is False
    assert event["error_type"] == "RuntimeError"
    assert event["latency_ms"] >= 0
