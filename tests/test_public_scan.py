from pathlib import Path

from legal_decision_model.public_scan import scan_public_content


def test_public_scan_finds_sensitive_patterns(tmp_path: Path) -> None:
    internal_host = "github" + ".corp"
    token = "ghp_" + "abcdefghijklmnopqrstuvwxyz1234"
    (tmp_path / "bad.txt").write_text(
        f"See {internal_host} and token {token}\n",
        encoding="utf-8",
    )
    findings = scan_public_content(tmp_path)
    assert {finding.category for finding in findings} == {
        "credential-like token",
        "internal host",
    }


def test_public_scan_ignores_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "local.txt").write_text("github" + ".corp\n", encoding="utf-8")
    assert scan_public_content(tmp_path) == []
