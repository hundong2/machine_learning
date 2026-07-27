"""Validate the small Gemma tutorial JSONL datasets with the standard library."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
ROLE_ORDER = {"system", "user", "assistant"}
SECRET_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)01[016789][-\s]?\d{3,4}[-\s]?\d{4}(?!\d)"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"{path}: file does not exist"]

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            errors.append(f"{path}:{line_number}: blank line")
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number}: row must be an object")
            continue
        value["_source_line"] = line_number
        rows.append(value)
    return rows, errors


def validate_row(row: dict[str, Any], path: Path) -> list[str]:
    line = row.get("_source_line", "?")
    prefix = f"{path}:{line}"
    errors: list[str] = []

    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        errors.append(f"{prefix}: id must be a non-empty string")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return errors + [f"{prefix}: messages must contain at least two turns"]

    previous_role: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{prefix}: messages[{index}] must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_ORDER:
            errors.append(f"{prefix}: messages[{index}] has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{prefix}: messages[{index}].content must be non-empty")
            continue
        if role == previous_role and role != "system":
            errors.append(f"{prefix}: consecutive {role!r} turns at index {index}")
        previous_role = role
        for pattern_name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(
                    f"{prefix}: possible {pattern_name} in messages[{index}]"
                )

    if messages[0].get("role") not in {"system", "user"}:
        errors.append(f"{prefix}: first role must be system or user")
    if messages[-1].get("role") != "assistant":
        errors.append(f"{prefix}: last role must be assistant")

    tags = row.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append(f"{prefix}: tags must be a list of strings")
    return errors


def first_user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[DEFAULT_DATA_DIR / "train.jsonl", DEFAULT_DATA_DIR / "eval.jsonl"],
        help="JSONL files to validate",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    all_rows: list[tuple[Path, dict[str, Any]]] = []
    for path in args.files:
        rows, load_errors = load_jsonl(path)
        all_errors.extend(load_errors)
        for row in rows:
            all_errors.extend(validate_row(row, path))
            all_rows.append((path, row))
        print(f"{path}: {len(rows)} rows")

    id_locations: dict[str, list[str]] = {}
    prompt_locations: dict[str, list[str]] = {}
    lengths: list[int] = []
    tag_counts: Counter[str] = Counter()

    for path, row in all_rows:
        location = f"{path}:{row.get('_source_line', '?')}"
        row_id = row.get("id")
        if isinstance(row_id, str):
            id_locations.setdefault(row_id, []).append(location)
        prompt = normalize(first_user_text(row))
        if prompt:
            prompt_locations.setdefault(prompt, []).append(location)
        for message in row.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                lengths.append(len(content))
        tag_counts.update(row.get("tags", []))

    for row_id, locations in id_locations.items():
        if len(locations) > 1:
            all_errors.append(f"duplicate id {row_id!r}: {', '.join(locations)}")
    for prompt, locations in prompt_locations.items():
        if len(locations) > 1:
            all_errors.append(
                f"duplicate normalized user prompt {prompt!r}: {', '.join(locations)}"
            )

    if lengths:
        ordered = sorted(lengths)
        print(
            "content characters: "
            f"min={ordered[0]}, median={ordered[len(ordered) // 2]}, max={ordered[-1]}"
        )
    print(f"tags: {dict(tag_counts.most_common())}")

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} error(s)", file=sys.stderr)
        for error in all_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"OK: {len(all_rows)} rows, no structural errors or exact prompt leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

