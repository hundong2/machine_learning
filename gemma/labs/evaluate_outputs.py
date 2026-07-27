"""Small deterministic evaluation for tutorial prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_DEMO = Path(__file__).resolve().parent / "data" / "demo_predictions.jsonl"


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[.!?。！？]+$", "", text)


def sentence_count(text: str) -> int:
    pieces = [piece.strip() for piece in re.split(r"[.!?。！？]\s*|\n+", text)]
    return sum(bool(piece) for piece in pieces)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        for field in ("id", "reference", "prediction"):
            if not isinstance(value.get(field), str):
                raise ValueError(f"{path}:{line_number}: {field} must be a string")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if not args.demo and args.input is None:
        parser.error("provide --input PATH or --demo")
    path = DEFAULT_DEMO if args.demo else args.input
    assert path is not None

    try:
        rows = load_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    exact = 0
    required_hits = 0
    required_total = 0
    forbidden_rows = 0
    length_ok = 0
    details = []

    for row in rows:
        prediction = row["prediction"]
        exact_match = normalize(prediction) == normalize(row["reference"])
        exact += int(exact_match)

        required = row.get("required_keywords", [])
        required_found = [
            keyword for keyword in required if normalize(keyword) in normalize(prediction)
        ]
        required_hits += len(required_found)
        required_total += len(required)

        forbidden = row.get("forbidden_keywords", [])
        forbidden_found = [
            keyword for keyword in forbidden if normalize(keyword) in normalize(prediction)
        ]
        forbidden_rows += int(bool(forbidden_found))

        max_sentences = int(row.get("max_sentences", 3))
        count = sentence_count(prediction)
        row_length_ok = 1 <= count <= max_sentences
        length_ok += int(row_length_ok)
        details.append(
            {
                "id": row["id"],
                "exact": exact_match,
                "required": f"{len(required_found)}/{len(required)}",
                "forbidden": forbidden_found,
                "sentences": count,
                "length_ok": row_length_ok,
            }
        )

    count = len(rows)
    metrics = {
        "samples": count,
        "exact_match": exact / count,
        "required_keyword_recall": (
            required_hits / required_total if required_total else 1.0
        ),
        "forbidden_hit_rate": forbidden_rows / count,
        "length_compliance": length_ok / count,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nper-sample:")
    for detail in details:
        print(json.dumps(detail, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

