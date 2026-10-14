"""`requires-python >= 3.10`: every module must parse with the Python 3.10 grammar (e.g. no PEP 701 f-strings)."""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for d in ("rlsignal", "tests", "experiments") for p in (ROOT / d).rglob("*.py")
               if "pilot_v2_reanalysis" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_module_parses_with_python_3_10_grammar(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="older interpreters already reject PEP 701 f-strings when parsing")
@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_pep701_fstrings(path):
    """CPython >= 3.12 ignores feature_version for PEP 701, so reused outer quotes / backslashes are checked here."""
    src = path.read_text(encoding="utf-8")
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.JoinedStr):
            continue
        seg = ast.get_source_segment(src, node) or ""
        body = seg.lstrip("rRbBfFuU")
        if body[:3] in ('"""', "'''"):
            continue
        quote = body[:1]
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                expr = ast.get_source_segment(src, v.value) or ""
                if quote in expr or "\\" in expr:
                    bad.append(f"line {node.lineno}: {expr}")
    assert not bad, bad
