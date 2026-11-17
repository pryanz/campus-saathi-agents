#!/usr/bin/env python3
"""Run the checked 80-question grounded-answer evaluation set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "tests" / "eval" / "questions.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.answering import answer_knowledge_query
from backend.knowledge import KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Write per-question results as JSON")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    kb = KnowledgeBase()
    rows = []

    for question in questions:
        started = time.perf_counter()
        try:
            result = answer_knowledge_query(question["query"], kb=kb)
            error = None
            response = result.model_dump()
        except Exception as exc:  # record failures without losing the rest of the run
            error = type(exc).__name__
            response = None

        rows.append(
            {
                "id": question["id"],
                "category": question["category"],
                "expected_record_ids": question["expected_record_ids"],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "error": error,
                "response": response,
            }
        )

    unsupported = [row for row in rows if not row["expected_record_ids"] and not row["error"]]
    supported = [row for row in rows if row["expected_record_ids"] and not row["error"]]
    false_positives = [row for row in unsupported if row["response"]["found"]]
    grounded = [
        row for row in supported
        if row["response"]["found"] and row["response"]["used_record_ids"]
    ]
    metrics = {
        "total": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "unsupported": len(unsupported),
        "unsupported_false_positive_rate": (
            len(false_positives) / len(unsupported) if unsupported else None
        ),
        "unsupported_safe_rate": (
            1 - len(false_positives) / len(unsupported) if unsupported else None
        ),
        "supported_found_rate": len(grounded) / len(supported) if supported else None,
        "answer_type_counts": dict(
            Counter(row["response"]["answer_type"] for row in rows if row["response"])
        ),
        "mean_latency_seconds": statistics.mean(row["elapsed_seconds"] for row in rows),
        "p50_latency_seconds": statistics.median(row["elapsed_seconds"] for row in rows),
    }

    output = {"metrics": metrics, "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
