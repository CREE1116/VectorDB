#!/usr/bin/env python3
"""Optional Codex/Claude Code UserPromptSubmit hook; read-only and silent on failure."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_database(cwd: Path) -> Path | None:
    override = os.environ.get("VECTORDB_DB_DIR")
    if override:
        path = Path(override).expanduser()
        return path if (path / "metadata.bin").is_file() else None
    for directory in (cwd, *cwd.parents):
        path = directory / ".vectordb"
        if (path / "metadata.bin").is_file():
            return path
    return None


def context_for_prompt(event: dict) -> str | None:
    prompt = event.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 12:
        return None
    binary = os.environ.get("VECTORDB_BIN") or shutil.which("vectordb")
    if not binary:
        return None
    cwd = Path(event.get("cwd") or os.getcwd()).resolve()
    db_dir = find_database(cwd)
    if db_dir is None:
        return None
    try:
        result = subprocess.run(
            [binary, "--db-dir", str(db_dir), "search", prompt[:300],
             "--format", "json", "--limit", "3", "--expand-graph", "0"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        )
        hits = json.loads(result.stdout).get("hits", [])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    leads = []
    for hit in hits[:3]:
        chunk = hit.get("chunk", {})
        path = chunk.get("file_path")
        line = chunk.get("start_line")
        if isinstance(path, str) and isinstance(line, int):
            safe_path = " ".join(path.split())[:180]
            leads.append(f"- {safe_path}:{line}")
    if not leads:
        return None
    return "Possible VectorDB leads (snapshot may be stale; verify current files):\n" + "\n".join(leads)


def main() -> None:
    try:
        event = json.load(sys.stdin)
        context = context_for_prompt(event)
        if context:
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit", "additionalContext": context
            }}))
    except (OSError, ValueError, TypeError):
        pass


if __name__ == "__main__":
    main()
