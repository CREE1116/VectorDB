#!/usr/bin/env python3
"""Install the repository skill and optional prompt hooks into another project."""

import argparse
import json
from pathlib import Path
import shutil
import shlex
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SKILL_SOURCE = SOURCE_ROOT / ".agents" / "skills" / "vectordb-search"
HOOK_SOURCE = SOURCE_ROOT / "integrations" / "prompt_hook.py"


def install_skill(project: Path, force: bool) -> None:
    destination = project / ".agents" / "skills" / "vectordb-search"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.resolve() != SKILL_SOURCE.resolve() and (destination / "SKILL.md").read_bytes() != (SKILL_SOURCE / "SKILL.md").read_bytes():
            if not force:
                raise RuntimeError(f"Different skill already exists at {destination}; use --force to replace it")
            shutil.rmtree(destination)
    else:
        shutil.copytree(SKILL_SOURCE, destination)

    claude = project / ".claude" / "skills" / "vectordb-search"
    claude.parent.mkdir(parents=True, exist_ok=True)
    if claude.is_symlink() and claude.resolve() == destination.resolve():
        return
    if claude.is_dir() and (claude / "SKILL.md").is_file() and (claude / "SKILL.md").read_bytes() == (SKILL_SOURCE / "SKILL.md").read_bytes():
        return
    if claude.exists() or claude.is_symlink():
        if not force:
            raise RuntimeError(f"Existing Claude skill at {claude}; use --force to replace it")
        if claude.is_symlink() or claude.is_file():
            claude.unlink()
        else:
            shutil.rmtree(claude)
    try:
        claude.symlink_to(Path("..") / ".." / ".agents" / "skills" / "vectordb-search", target_is_directory=True)
    except OSError:
        shutil.copytree(destination, claude)


def install_hook(project: Path, agent: str) -> None:
    hook_file = project / ".agents" / "vectordb" / "prompt_hook.py"
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HOOK_SOURCE, hook_file)
    if agent == "codex":
        config_path = project / ".codex" / "hooks.json"
        command = f"python3 {shlex.quote(str(hook_file))}"
    else:
        config_path = project / ".claude" / "settings.json"
        command = 'python3 "${CLAUDE_PROJECT_DIR}/.agents/vectordb/prompt_hook.py"'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    hooks = config.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise RuntimeError(f"Invalid hooks object in {config_path}")
    prompt_hooks = hooks.setdefault("UserPromptSubmit", [])
    if not isinstance(prompt_hooks, list):
        raise RuntimeError(f"Invalid UserPromptSubmit list in {config_path}")
    existing = False
    for group in prompt_hooks:
        if not isinstance(group, dict):
            continue
        for handler in group.get("hooks", []):
            if isinstance(handler, dict) and ".agents/vectordb/prompt_hook.py" in str(handler.get("command", "")):
                handler["command"] = command
                existing = True
    if not existing:
        prompt_hooks.append({"hooks": [{"type": "command", "command": command, "timeout": 8}]})
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Target project directory")
    parser.add_argument("--hooks", choices=["none", "codex", "claude", "both"], default="none")
    parser.add_argument("--force", action="store_true", help="Replace a different existing VectorDB skill")
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    if not project.is_dir():
        parser.error(f"Project directory does not exist: {project}")
    try:
        install_skill(project, args.force)
        for agent in ("codex", "claude"):
            if args.hooks in (agent, "both"):
                install_hook(project, agent)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Installed VectorDB skill in {project}")
    if args.hooks != "none":
        print(f"Installed optional {args.hooks} prompt hook(s); review and trust them in your agent before use")
    if shutil.which("vectordb") is None:
        print("Install the CLI first: cargo install --path crates/vectordb-cli --locked", file=sys.stderr)


if __name__ == "__main__":
    main()
