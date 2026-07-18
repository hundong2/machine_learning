#!/usr/bin/env python3
"""Validate daily-study Markdown and GitHub-compatible math before commit."""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token


FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
ENV_RE = re.compile(r"\\(begin|end)\{([^{}]+)\}")
EXTERNAL_SCHEMES = {"http", "https", "mailto", "data", "tel"}


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slash_count += 1
        index -= 1
    return slash_count % 2 == 1


def find_repo_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return parent
    return path.parent


def protect_display_math(text: str) -> str:
    """Wrap display math in an HTML block so vanilla Markdown cannot parse its lines."""
    output: list[str] = []
    fence: tuple[str, int] | None = None
    math_kind: str | None = None

    for line in text.splitlines(keepends=True):
        stripped_line = line.rstrip("\r\n")
        newline = line[len(stripped_line) :]
        stripped = stripped_line.strip()

        if fence is not None:
            output.append(line)
            match = FENCE_CLOSE_RE.match(stripped_line)
            if match:
                marker = match.group(1)
                if marker[0] == fence[0] and len(marker) >= fence[1]:
                    fence = None
            continue

        fence_match = FENCE_RE.match(stripped_line)
        if fence_match:
            marker = fence_match.group(1)
            fence = (marker[0], len(marker))
            output.append(line)
            continue

        if math_kind is None and stripped in {"$$", r"\["}:
            math_kind = stripped
            output.append("<div class=\"math-display\">\\[" + newline)
        elif math_kind is not None and stripped == ("$$" if math_kind == "$$" else r"\]"):
            output.append("\\]</div>" + newline)
            math_kind = None
        else:
            output.append(line)
    return "".join(output)


def parse_markdown(text: str) -> tuple[MarkdownIt, list[Token], str]:
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    protected = protect_display_math(text)
    tokens = parser.parse(protected)
    rendered = parser.render(protected)
    return parser, tokens, rendered


def check_fences(path: Path, lines: list[str]) -> list[Issue]:
    issues: list[Issue] = []
    opened: tuple[str, int, int] | None = None

    for number, line in enumerate(lines, start=1):
        without_newline = line.rstrip("\r\n")
        if opened is None:
            match = FENCE_RE.match(without_newline)
            if not match:
                continue
            marker, info = match.groups()
            opened = (marker[0], len(marker), number)
            if not info.strip():
                issues.append(Issue(path, number, "fenced code block has no language tag"))
            continue

        match = FENCE_CLOSE_RE.match(without_newline)
        if match:
            marker = match.group(1)
            if marker[0] == opened[0] and len(marker) >= opened[1]:
                opened = None

    if opened is not None:
        issues.append(Issue(path, opened[2], "unclosed fenced code block"))
    return issues


def code_line_ranges(tokens: list[Token]) -> set[int]:
    masked: set[int] = set()
    for token in tokens:
        if token.type not in {"fence", "code_block"} or token.map is None:
            continue
        start, end = token.map
        masked.update(range(start, end))
    return masked


def mask_code_lines(lines: list[str], masked_lines: set[int]) -> str:
    output: list[str] = []
    for index, line in enumerate(lines):
        if index in masked_lines:
            output.append("\n" if line.endswith("\n") else "")
        else:
            output.append(line)
    return "".join(output)


def mask_inline_code(path: Path, text: str) -> tuple[str, list[Issue]]:
    chars = list(text)
    issues: list[Issue] = []
    opened: tuple[int, int] | None = None
    index = 0

    while index < len(text):
        if text[index] != "`" or is_escaped(text, index):
            if opened is not None and text[index] != "\n":
                chars[index] = " "
            index += 1
            continue

        end = index
        while end < len(text) and text[end] == "`":
            end += 1
        run_length = end - index

        if opened is None:
            opened = (run_length, index)
        elif opened[0] == run_length:
            opened = None

        for position in range(index, end):
            chars[position] = " "
        index = end

    if opened is not None:
        issues.append(
            Issue(path, line_number(text, opened[1]), "unclosed inline-code backtick span")
        )
    return "".join(chars), issues


def check_heading_structure(
    path: Path,
    tokens: list[Token],
    lines: list[str],
    masked_lines: set[int],
) -> list[Issue]:
    issues: list[Issue] = []
    headings: list[tuple[int, int]] = []

    for token in tokens:
        if token.type != "heading_open" or token.map is None:
            continue
        headings.append((int(token.tag[1]), token.map[0] + 1))

    h1_lines = [line for level, line in headings if level == 1]
    if len(h1_lines) != 1:
        issues.append(
            Issue(path, h1_lines[0] if h1_lines else 1, f"expected exactly one H1, found {len(h1_lines)}")
        )

    previous_level = 0
    for level, number in headings:
        if previous_level and level > previous_level + 1:
            issues.append(
                Issue(path, number, f"heading level jumps from H{previous_level} to H{level}")
            )
        previous_level = level

    for index, line in enumerate(lines):
        if index in masked_lines or not HEADING_RE.match(line):
            continue
        number = index + 1
        if index > 0 and lines[index - 1].strip():
            issues.append(Issue(path, number, "heading must have a blank line before it"))
        if index + 1 < len(lines) and lines[index + 1].strip():
            issues.append(Issue(path, number, "heading must have a blank line after it"))
    return issues


def table_cell_count(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return len(re.split(r"(?<!\\)\|", stripped))


def check_tables(path: Path, lines: list[str], masked_lines: set[int]) -> list[Issue]:
    issues: list[Issue] = []
    for index, line in enumerate(lines):
        if index in masked_lines or not TABLE_SEPARATOR_RE.match(line.rstrip("\r\n")):
            continue
        expected = table_cell_count(line)
        if index == 0 or table_cell_count(lines[index - 1]) != expected:
            issues.append(Issue(path, index + 1, "table header and separator have different column counts"))
        row = index + 1
        while row < len(lines) and lines[row].strip() and "|" in lines[row]:
            if row in masked_lines:
                break
            actual = table_cell_count(lines[row])
            if actual != expected:
                issues.append(
                    Issue(path, row + 1, f"table row has {actual} columns; expected {expected}")
                )
            row += 1
    return issues


def validate_latex_fragment(path: Path, fragment: str, start_line: int) -> list[Issue]:
    issues: list[Issue] = []
    brace_stack: list[int] = []

    for index, char in enumerate(fragment):
        if char not in "{}" or is_escaped(fragment, index):
            continue
        if char == "{":
            brace_stack.append(index)
        elif brace_stack:
            brace_stack.pop()
        else:
            issues.append(
                Issue(path, start_line + fragment.count("\n", 0, index), "unmatched '}' in math")
            )
    for index in brace_stack:
        issues.append(
            Issue(path, start_line + fragment.count("\n", 0, index), "unmatched '{' in math")
        )

    env_stack: list[tuple[str, int]] = []
    for match in ENV_RE.finditer(fragment):
        kind, environment = match.groups()
        current_line = start_line + fragment.count("\n", 0, match.start())
        if kind == "begin":
            env_stack.append((environment, current_line))
        elif not env_stack:
            issues.append(Issue(path, current_line, f"\\end{{{environment}}} has no matching begin"))
        elif env_stack[-1][0] != environment:
            issues.append(
                Issue(
                    path,
                    current_line,
                    f"math environment closes {environment!r} while {env_stack[-1][0]!r} is open",
                )
            )
        else:
            env_stack.pop()
    for environment, current_line in env_stack:
        issues.append(Issue(path, current_line, f"unclosed math environment {environment!r}"))

    left_count = len(re.findall(r"(?<!\\)\\left\b", fragment))
    right_count = len(re.findall(r"(?<!\\)\\right\b", fragment))
    if left_count != right_count:
        issues.append(
            Issue(path, start_line, f"\\left/\\right count differs ({left_count}/{right_count})")
        )
    return issues


def check_math(path: Path, text: str) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    fragments: list[tuple[str, int]] = []
    lines = text.splitlines(keepends=True)
    block_kind: str | None = None
    block_start = 0
    block_content: list[str] = []

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if block_kind is not None:
            expected = "$$" if block_kind == "$$" else r"\]"
            if stripped == expected:
                fragments.append(("".join(block_content), block_start + 1))
                block_kind = None
                block_content = []
            else:
                block_content.append(line)
            continue

        if stripped == "$$":
            block_kind = "$$"
            block_start = number
            continue
        if stripped == r"\[":
            block_kind = r"\["
            block_start = number
            continue
        if "$$" in line:
            issues.append(
                Issue(path, number, "display-math '$$' delimiter must be on its own line")
            )

        inline_open: int | None = None
        index = 0
        while index < len(line):
            if line[index] == "$" and not is_escaped(line, index):
                if index + 1 < len(line) and line[index + 1] == "$":
                    index += 2
                    continue
                if inline_open is None:
                    inline_open = index
                else:
                    fragments.append((line[inline_open + 1 : index], number))
                    inline_open = None
            index += 1
        if inline_open is not None:
            issues.append(Issue(path, number, "unclosed inline '$' math delimiter"))

        paren_stack: list[int] = []
        for match in re.finditer(r"(?<!\\)\\([()])", line):
            if match.group(1) == "(":
                paren_stack.append(match.end())
            elif not paren_stack:
                issues.append(Issue(path, number, r"\) has no matching \("))
            else:
                start = paren_stack.pop()
                fragments.append((line[start : match.start()], number))
        if paren_stack:
            issues.append(Issue(path, number, r"unclosed \( math delimiter"))

    if block_kind is not None:
        closing = "$$" if block_kind == "$$" else r"\]"
        issues.append(Issue(path, block_start, f"unclosed display-math block; expected {closing}"))

    for fragment, start_line in fragments:
        if not fragment.strip():
            issues.append(Issue(path, start_line, "empty math expression"))
        issues.extend(validate_latex_fragment(path, fragment, start_line))
    return issues, len(fragments)


def walk_tokens(tokens: list[Token]):
    for token in tokens:
        yield token
        if token.children:
            yield from walk_tokens(token.children)


def check_local_links(path: Path, tokens: list[Token]) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    count = 0
    repo_root = find_repo_root(path)

    for token in walk_tokens(tokens):
        if token.type not in {"link_open", "image"}:
            continue
        target = token.attrGet("href") if token.type == "link_open" else token.attrGet("src")
        if not target:
            continue
        count += 1
        parsed = urlsplit(target)
        if parsed.scheme.lower() in EXTERNAL_SCHEMES or target.startswith("#"):
            continue
        decoded_path = unquote(parsed.path)
        if not decoded_path:
            continue
        candidate = (
            repo_root / decoded_path.lstrip("/")
            if decoded_path.startswith("/")
            else path.parent / decoded_path
        )
        if not candidate.exists():
            line = token.map[0] + 1 if token.map else 1
            issues.append(Issue(path, line, f"local link target does not exist: {decoded_path}"))
    return issues, count


def validate_file(path: Path, html_output: Path | None = None) -> tuple[list[Issue], dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    _, tokens, rendered = parse_markdown(text)
    masked_lines = code_line_ranges(tokens)

    issues: list[Issue] = []
    issues.extend(check_fences(path, lines))
    issues.extend(check_heading_structure(path, tokens, lines, masked_lines))
    issues.extend(check_tables(path, lines, masked_lines))

    without_blocks = mask_code_lines(lines, masked_lines)
    without_code, inline_issues = mask_inline_code(path, without_blocks)
    issues.extend(inline_issues)
    math_issues, math_count = check_math(path, without_code)
    issues.extend(math_issues)
    link_issues, link_count = check_local_links(path, tokens)
    issues.extend(link_issues)

    if "<h1" not in rendered:
        issues.append(Issue(path, 1, "Markdown render produced no H1"))

    if html_output is not None:
        title = html.escape(path.stem)
        document = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]}};</script>"
            "<script defer src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>"
            "</head><body>"
            f"{rendered}</body></html>"
        )
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(document, encoding="utf-8")

    metrics = {
        "lines": len(lines),
        "tokens": len(tokens),
        "math": math_count,
        "links": link_count,
    }
    return issues, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--html-output",
        type=Path,
        help="write a rendered HTML preview (only valid with one input file)",
    )
    args = parser.parse_args()

    if args.html_output is not None and len(args.files) != 1:
        parser.error("--html-output requires exactly one input file")

    all_issues: list[Issue] = []
    for index, path in enumerate(args.files):
        if not path.is_file():
            all_issues.append(Issue(path, 1, "file does not exist"))
            continue
        output = args.html_output if index == 0 else None
        issues, metrics = validate_file(path, output)
        all_issues.extend(issues)
        if not issues:
            print(
                f"PASS {path} "
                f"(lines={metrics['lines']}, tokens={metrics['tokens']}, "
                f"math={metrics['math']}, links={metrics['links']})"
            )

    if all_issues:
        for issue in sorted(all_issues, key=lambda item: (str(item.path), item.line, item.message)):
            print(issue.render(), file=sys.stderr)
        print(f"FAIL: {len(all_issues)} Markdown/math issue(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
