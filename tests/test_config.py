"""Config loading: duplicate YAML keys must fail loudly (the last one used to win silently)."""

from pathlib import Path

import pytest

from rlsignal.config import DuplicateKeyError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_key_in_nested_mapping_is_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "name: x\nestimator:\n  uncertain_reference: block\n  prob: 0.8\n  uncertain_reference: ignore_with_flag\n",
        encoding="utf-8",
    )
    with pytest.raises(DuplicateKeyError, match="uncertain_reference") as e:
        load_config(p)
    assert "line 5" in str(e.value)


def test_duplicate_top_level_section_is_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("estimator: {prob: 0.8}\nestimator: {prob: 0.9}\n", encoding="utf-8")
    with pytest.raises(DuplicateKeyError, match="estimator"):
        load_config(p)


def test_utf8_bom_and_non_ascii_are_accepted(tmp_path):
    p = tmp_path / "bom.yaml"
    p.write_bytes("\ufeffname: caf\u00e9  # \u2014 comment\nlist: [1, 2]\n".encode("utf-8"))
    assert load_config(p) == {"name": "caf\u00e9", "list": [1, 2]}


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_have_no_duplicate_keys(path):
    assert isinstance(load_config(path), dict)
