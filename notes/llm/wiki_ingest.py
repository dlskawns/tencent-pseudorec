#!/usr/bin/env python3
"""Auto-ingest hook for notes/llm/wiki/.

Fires on PostToolUse for Write|Edit|MultiEdit. Reads hook payload from stdin,
classifies the touched file, then upserts an Obsidian-friendly catalog page
under notes/llm/wiki/by_<section>/<entity>.md and appends an event to log.md.

Failures are non-fatal — exit 0 always (tool execution must not be blocked).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIKI_ROOT = PROJECT_ROOT / "notes" / "llm" / "wiki"
LOG_PATH = WIKI_ROOT / "log.md"
INDEX_PATH = WIKI_ROOT / "index.md"

KST = timezone(timedelta(hours=9))

SKIP_PREFIXES = (
    "notes/llm/",
    ".claude/",
    ".git/",
    ".obsidian/",
    ".venv-arm64/",
    ".venv/",
    "data/",
)
SKIP_NAMES = (".DS_Store",)
SKIP_SUFFIXES = (".pyc", ".bundle", ".parquet", ".lock")

SECTION_TYPE = {
    "by_hypothesis": ("hypothesis", "hypothesis"),
    "by_experiment": ("experiment", "experiment"),
    "by_eda": ("eda-out", "eda"),
    "by_paper": ("paper", "paper"),
    "by_governance": ("governance", "governance"),
}

INDEX_HEADERS = {
    "by_hypothesis": "## Hypotheses",
    "by_experiment": "## Experiments",
    "by_eda": "## EDA",
    "by_paper": "## Papers",
    "by_governance": "## Governance",
}


def now_kst_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


def to_relpath(file_path: str) -> str | None:
    if not file_path:
        return None
    p = Path(file_path)
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT)
    except (ValueError, OSError):
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            return None
    return str(rel)


def should_skip(rel: str) -> bool:
    if any(rel.startswith(pref) for pref in SKIP_PREFIXES):
        return True
    if any(rel.endswith(suf) for suf in SKIP_SUFFIXES):
        return True
    name = rel.rsplit("/", 1)[-1]
    if name in SKIP_NAMES:
        return True
    return False


def classify(rel: str):
    """Return (section, entity_id) or None."""
    m = re.match(r"hypotheses/(H\d{3}[_A-Za-z0-9]*)/", rel)
    if m:
        return ("by_hypothesis", m.group(1))
    if rel == "hypotheses/INDEX.md":
        return ("by_hypothesis", "INDEX")
    m = re.match(r"experiments/((?:H|E)\d{3}[_A-Za-z0-9]*)/", rel)
    if m:
        return ("by_experiment", m.group(1))
    if rel == "experiments/INDEX.md":
        return ("by_experiment", "INDEX")
    m = re.match(r"eda/out/([^/]+)\.json$", rel)
    if m:
        return ("by_eda", m.group(1))
    m = re.match(r"papers/([^/]+)/([^/]+)\.md$", rel)
    if m:
        return ("by_paper", f"{m.group(1)}__{m.group(2)}")
    m = re.match(r"notes/refs/([^/]+)\.md$", rel)
    if m:
        return ("by_governance", m.group(1))
    if rel == "CLAUDE.md":
        return ("by_governance", "CLAUDE")
    if rel == "progress.txt":
        return ("by_governance", "progress")
    return None


def page_template(section: str, entity_id: str, title: str, date: str) -> str:
    return f"""# {title}

*auto-generated catalog · 원본은 source 링크 참조*

## Sources

## Activity

## Notes

(인간/LLM 자유 기술 영역 — auto-ingest는 이 섹션을 건드리지 않음)
"""


def entity_title(section: str, entity_id: str) -> str:
    if section == "by_paper":
        return entity_id.replace("__", " / ")
    typ, _ = SECTION_TYPE.get(section, ("note", "note"))
    return f"{entity_id} ({typ})"


def discover_sources(section: str, entity_id: str) -> list[str]:
    """Return list of wikilink targets (without .md) under the entity's source dir."""
    candidates: list[Path] = []
    if section == "by_hypothesis" and entity_id != "INDEX":
        for d in PROJECT_ROOT.glob(f"hypotheses/{entity_id}*"):
            if d.is_dir():
                candidates.append(d)
                break
    elif section == "by_experiment" and entity_id != "INDEX":
        for d in PROJECT_ROOT.glob(f"experiments/{entity_id}*"):
            if d.is_dir():
                candidates.append(d)
                break
    elif section == "by_eda":
        f = PROJECT_ROOT / "eda" / "out" / f"{entity_id}.json"
        if f.is_file():
            return [f"eda/out/{entity_id}.json"]
        return []
    elif section == "by_paper":
        if "__" in entity_id:
            cat, name = entity_id.split("__", 1)
            f = PROJECT_ROOT / "papers" / cat / f"{name}.md"
            if f.is_file():
                return [f"papers/{cat}/{name}"]
        return []
    elif section == "by_governance":
        if entity_id == "CLAUDE":
            return ["CLAUDE"]
        if entity_id == "progress":
            return ["progress.txt"]
        if entity_id == "INDEX":
            return []
        f = PROJECT_ROOT / "notes" / "refs" / f"{entity_id}.md"
        if f.is_file():
            return [f"notes/refs/{entity_id}"]
        return []

    targets: list[str] = []
    for d in candidates:
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            if any(f.name.endswith(s) for s in (".pyc", ".lock")):
                continue
            rel = f.relative_to(PROJECT_ROOT)
            target = str(rel)
            if target.endswith(".md"):
                target = target[:-3]
            targets.append(target)
    return targets


def replace_section_block(text: str, header: str, new_body: str) -> str:
    """Replace text under `header` (## Foo) up to next `## ` or EOF. Header preserved."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    replaced = False
    while i < n:
        line = lines[i]
        if not replaced and line.strip() == header:
            out.append(line)
            i += 1
            # consume until next ## header or EOF
            while i < n and not lines[i].startswith("## "):
                i += 1
            out.append("\n")
            out.append(new_body if new_body.endswith("\n") else new_body + "\n")
            out.append("\n")
            replaced = True
            continue
        out.append(line)
        i += 1
    return "".join(out)


def append_to_section(text: str, header: str, line: str) -> str:
    """Append `line` to the end of the named ## section, normalizing whitespace.

    Section spans from the header line to the next `## ` header or EOF. Trailing
    blank lines inside the section are stripped, then `line` is appended on its
    own line, followed by a single blank separator before the next header.
    """
    pattern = re.compile(
        r"(^" + re.escape(header) + r"[ \t]*\n)(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return text
    head = m.group(1)
    body = m.group(2).rstrip("\n")
    bullet = line.rstrip("\n")
    if body.strip():
        new_body = body + "\n" + bullet + "\n\n"
    else:
        new_body = "\n" + bullet + "\n\n"
    return text[: m.start()] + head + new_body + text[m.end() :]


def upsert_page(section: str, entity_id: str, rel_touched: str, action: str, ts_str: str) -> None:
    page_path = WIKI_ROOT / section / f"{entity_id}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)

    date = ts_str[:10]
    if not page_path.exists():
        title = entity_title(section, entity_id)
        page_path.write_text(page_template(section, entity_id, title, date), encoding="utf-8")

    text = page_path.read_text(encoding="utf-8")

    sources = discover_sources(section, entity_id)
    if sources:
        body = "\n".join(f"- [[{t}]]" for t in sources)
        text = replace_section_block(text, "## Sources", body)

    activity_line = f"- {ts_str} — {action} `{rel_touched}`\n"
    text = append_to_section(text, "## Activity", activity_line)

    page_path.write_text(text, encoding="utf-8")


def append_log(rel_touched: str, action: str, ts_str: str, classified) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# LLM Wiki Log\n\n> chronological, append-only.\n\n", encoding="utf-8"
        )
    if classified:
        section, eid = classified
        link = f"[[{section}/{eid}]]"
    else:
        link = "(unmapped)"
    line = f"- {ts_str} | {action} | `{rel_touched}` | {link}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def update_index(classified) -> None:
    if not classified or not INDEX_PATH.exists():
        return
    section, eid = classified
    header = INDEX_HEADERS.get(section)
    if not header:
        return
    text = INDEX_PATH.read_text(encoding="utf-8")
    link = f"- [[{section}/{eid}]]"
    if link in text:
        return
    if header not in text:
        return
    pattern = re.compile(re.escape(header) + r"\n+")
    new_text, n = pattern.subn(f"{header}\n{link}\n\n", text, count=1)
    if n:
        INDEX_PATH.write_text(new_text, encoding="utf-8")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Write", "Edit", "MultiEdit"):
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or ""
    rel = to_relpath(file_path)
    if not rel or should_skip(rel):
        return 0

    ts_str = now_kst_str()
    action = {"Write": "write", "Edit": "edit", "MultiEdit": "edit*"}.get(tool_name, "touch")
    classified = classify(rel)

    try:
        append_log(rel, action, ts_str, classified)
        if classified:
            section, eid = classified
            upsert_page(section, eid, rel, action, ts_str)
            update_index(classified)
    except Exception as e:
        sys.stderr.write(f"wiki_ingest error: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
