"""Tool Experience Extractor — Extracts L1 facts from tool outputs.

Captures:
- Bash failures (exit ≠ 0): error patterns, fix attempts
- Key successes: compiled, passed, installed, built
- Environment quirks: not found, permission denied, path issues

Skips:
- Pure read/query commands: ls, cat, echo, find, read, search
- Routine successes without learning value
- Tool metadata summaries

Design decisions (10万条 scale):
- Only analyzes bash outputs (not read/find/search/lsp)
- Failure + high-value success only (not all commands)
- ~150 bash calls analyzed → ~50 L1 facts extracted
"""

import re
from typing import Any

# Commands to skip (pure queries, no experience value)
_SKIP_COMMANDS = {
    "ls", "cat", "echo", "find", "read", "search", "head", "tail",
    "wc", "grep", "rg", "ag", "ack", "more", "less", "pwd", "which",
    "whoami", "date", "env", "printenv", "true", "false", "test",
}

# Error patterns to extract
_ERROR_PATTERNS = [
    (r"error\[E\d+\]:\s*(.+?)(?:\n|$)", "rust_error"),
    (r"error:\s*(.+?)(?:\n|$)", "error"),
    (r"FAILED:\s*(.+?)(?:\n|$)", "build_failed"),
    (r"fatal:\s*(.+?)(?:\n|$)", "fatal"),
    (r"panic:\s*(.+?)(?:\n|$)", "panic"),
    (r"Exception:\s*(.+?)(?:\n|$)", "exception"),
    (r"Traceback \(most recent call last\):", "python_traceback"),
    (r"Permission denied", "permission"),
    (r"No such file or directory", "file_not_found"),
    (r"not found$", "not_found"),
    (r"command not found", "cmd_not_found"),
    (r"ModuleNotFoundError:\s*(.+?)(?:\n|$)", "module_missing"),
    (r"ImportError:\s*(.+?)(?:\n|$)", "import_error"),
    (r"OSError:\s*(.+?)(?:\n|$)", "os_error"),
    (r"WinError \d+", "win_error"),
]

# Success patterns to extract
_SUCCESS_PATTERNS = [
    (r"(\d+)\s+passed", "tests_passed"),
    (r"compiled\s+successfully", "compiled"),
    (r"Successfully installed\s+(.+?)(?:\n|$)", "installed"),
    (r"Installed\s+(.+?)(?:\n|$)", "installed"),
    (r"Build\s+(?:complete|successful|finished)", "build_success"),
    (r"ok\s*$", "ok"),
]

# Commands that indicate build/test (high value successes)
_BUILD_TEST_COMMANDS = {
    "cargo", "maturin", "pip", "npm", "yarn", "make", "cmake",
    "pytest", "jest", "go test", "python -m pytest",
}


class ToolExperienceExtractor:
    """Extracts experience facts from tool outputs."""

    def __init__(self):
        pass

    def should_analyze(self, command: str) -> bool:
        """Check if a bash command is worth analyzing.

        Skips pure query commands and trivial operations.
        """
        if not command:
            return False

        # Extract the first word (the command name)
        first_word = command.strip().split()[0].split("/")[-1]
        if first_word in _SKIP_COMMANDS:
            return False

        return True

    def extract_from_output(
        self,
        command: str,
        output: str,
        exit_code: int = 0,
    ) -> list[dict]:
        """Extract experience facts from a bash command output.

        Args:
            command: The bash command that was run.
            output: The combined stdout+stderr output.
            exit_code: The exit code (0 = success).

        Returns:
            List of fact dicts with 'label', 'importance', 'tags', 'source'.
        """
        if not self.should_analyze(command):
            return []

        facts = []

        if exit_code != 0:
            # Failure: extract error patterns
            facts.extend(self._extract_errors(command, output))
        else:
            # Success: only extract high-value successes
            facts.extend(self._extract_successes(command, output))

        return facts

    def _extract_errors(self, command: str, output: str) -> list[dict]:
        """Extract error experience facts."""
        facts = []
        cmd_short = command[:80]

        for pattern, error_type in _ERROR_PATTERNS:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                error_msg = match.group(0)[:150]
                label = f"命令 '{cmd_short}' 报错: {error_msg}"
                facts.append({
                    "label": label,
                    "importance": 0.7,
                    "tags": ["tool_experience", "error", error_type],
                    "source": "tool",
                })
                break  # One error per command is enough

        # Environment quirks
        if "not found" in output.lower() or "not recognized" in output.lower():
            facts.append({
                "label": f"命令 '{cmd_short}' 在当前环境不可用",
                "importance": 0.6,
                "tags": ["tool_experience", "environment"],
                "source": "tool",
            })

        return facts

    def _extract_successes(self, command: str, output: str) -> list[dict]:
        """Extract high-value success facts."""
        facts = []
        cmd_short = command[:80]

        # Only analyze build/test commands for successes
        is_build_test = any(bt in command.lower() for bt in _BUILD_TEST_COMMANDS)
        if not is_build_test:
            return []

        for pattern, success_type in _SUCCESS_PATTERNS:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                detail = match.group(0)[:100]
                label = f"命令 '{cmd_short}' 成功: {detail}"
                facts.append({
                    "label": label,
                    "importance": 0.5,
                    "tags": ["tool_experience", "success", success_type],
                    "source": "tool",
                })
                break  # One success per command is enough

        return facts

    def format_for_extract_deep(self, facts: list[dict]) -> str:
        """Format facts as a string for extract_deep input."""
        if not facts:
            return ""
        lines = [f["label"] for f in facts]
        return "\n".join(lines)
