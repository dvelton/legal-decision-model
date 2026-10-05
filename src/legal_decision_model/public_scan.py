"""Scan public-project files for likely internal or sensitive remnants."""

import re
from dataclasses import dataclass
from pathlib import Path

from legal_decision_model.constants import PROJECT_ROOT

EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "benchmark",
    "policy-model.work",
}
TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _patterns() -> dict[str, re.Pattern[str]]:
    internal_host = "github" + r"\.(?:corp|net|internal)"
    slack_archive = "slack" + r"\.com/archives/"
    home_path = "/" + "Users" + "/"
    private_repositories = (
        "github/"
        + "(?:"
        + "|".join(
            (
                "comm" + "legal",
                "product-and-" + "privacy-legal",
                "customer-" + "security-trust",
            )
        )
        + ")"
    )
    session_path = r"\." + "copilot/session-state"
    internal_markers = (
        "Development " + "Milestones",
        "Decisions " + r"\(Resolved\)",
        "Copilot-" + "Session:",
    )
    internal_tools = ("Work" + "IQ", "CEL" + "A")
    return {
        "internal host": re.compile(internal_host, re.IGNORECASE),
        "Slack archive URL": re.compile(slack_archive, re.IGNORECASE),
        "local user path": re.compile(re.escape(home_path) + r"[^/\s]+/"),
        "private repository reference": re.compile(private_repositories, re.IGNORECASE),
        "session artifact path": re.compile(session_path, re.IGNORECASE),
        "internal conversation marker": re.compile("|".join(internal_markers), re.IGNORECASE),
        "internal organization or tool name": re.compile(
            r"\b(?:" + "|".join(internal_tools) + r")\b", re.IGNORECASE
        ),
        "corporate email": re.compile(
            r"\b[A-Z0-9._%+-]+@" + r"(?:github|microsoft)\.com\b", re.IGNORECASE
        ),
        "credential-like token": re.compile(
            r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})"
        ),
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }


@dataclass(frozen=True)
class Finding:
    """One public-content scan match."""

    path: str
    line: int
    category: str
    excerpt: str


def candidate_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """List text-like project files outside local build and model directories."""
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        paths.append(path)
    return sorted(paths)


def scan_public_content(root: Path = PROJECT_ROOT) -> list[Finding]:
    """Return likely internal or sensitive strings that require review."""
    findings = []
    patterns = _patterns()
    for path in candidate_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for category, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(
                        Finding(
                            path=str(path.relative_to(root)),
                            line=line_number,
                            category=category,
                            excerpt=line.strip()[:160],
                        )
                    )
    return findings
