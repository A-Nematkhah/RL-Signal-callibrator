"""Every text-mode file operation must pass an explicit encoding (the Windows default cp1252 broke YAML loading)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    p
    for d in ("rlsignal", "tests", "experiments")
    for p in (ROOT / d).rglob("*.py")
    if "pilot_v2_reanalysis" not in p.parts
)


def _missing_encoding(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name not in ("open", "read_text", "write_text"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        if name == "open":
            mode = (
                node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
            )
            if isinstance(mode, ast.Constant) and "b" in str(mode.value):
                continue
        yield node.lineno


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_text_io_declares_encoding(path):
    lines = list(_missing_encoding(ast.parse(path.read_text(encoding="utf-8"))))
    assert not lines, f"{path.name}: open/read_text/write_text without encoding= at lines {lines}"
