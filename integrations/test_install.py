"""Behavioral checks for the portable skill installer and read-only prompt hook."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AgentIntegrationTest(unittest.TestCase):
    def test_install_preserves_settings_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / "first"
            project.mkdir()
            (project / ".claude").mkdir()
            settings = project / ".claude" / "settings.json"
            settings.write_text(json.dumps({"permissions": {"allow": ["Read"]}}))
            command = ["python3", str(ROOT / "integrations" / "install.py"), str(project), "--hooks", "both"]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(command, check=True, capture_output=True)
            self.assertTrue((project / ".agents/skills/vectordb-search/SKILL.md").is_file())
            self.assertTrue((project / ".claude/skills/vectordb-search/SKILL.md").is_file())
            claude = json.loads(settings.read_text())
            codex = json.loads((project / ".codex/hooks.json").read_text())
            self.assertEqual(claude["permissions"]["allow"], ["Read"])
            self.assertEqual(len(claude["hooks"]["UserPromptSubmit"]), 1)
            self.assertEqual(len(codex["hooks"]["UserPromptSubmit"]), 1)
            moved = Path(temp) / "moved"
            project.rename(moved)
            subprocess.run(command[:2] + [str(moved), "--hooks", "both"], check=True, capture_output=True)
            codex = json.loads((moved / ".codex/hooks.json").read_text())
            commands = [handler["command"] for group in codex["hooks"]["UserPromptSubmit"] for handler in group["hooks"]]
            self.assertEqual(len(commands), 1)
            self.assertIn(str(moved), commands[0])

    @unittest.skipUnless(shutil.which("vectordb"), "vectordb CLI not installed")
    def test_hook_returns_actual_index_path(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "notes.md").write_text("authentication token refresh routine")
            subprocess.run(["vectordb", "index", "."], cwd=project, check=True, capture_output=True)
            event = {"cwd": str(project), "prompt": "find authentication token refresh routine"}
            hook = ROOT / "integrations" / "prompt_hook.py"
            result = subprocess.run(["python3", str(hook)], input=json.dumps(event), text=True,
                                    capture_output=True, check=True)
            context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
            self.assertIn("notes.md:", context)
            self.assertNotIn("authentication token refresh routine", context)


if __name__ == "__main__":
    unittest.main()
