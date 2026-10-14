"""Configuration loading, compatibility hashing and checkpoint-grid helpers."""

from __future__ import annotations
import hashlib
import json
import yaml

# Only these sections determine whether stored results are mutually compatible.
# The checkpoint grid, seed list, horizon and detection thresholds are
# deliberately excluded: extending a grid or re-analysing must be allowed.
COMPAT_KEYS = ("adapter", "rewards", "algo", "eval")
# Machine-specific adapter keys must never influence compatibility (moving the repo must not quarantine results).
MACHINE_KEYS = ("crowdnav_path",)


class DuplicateKeyError(ValueError):
    pass


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys in any mapping (PyYAML silently keeps the last one)."""

    def construct_mapping(self, node, deep=False):
        seen = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            line = key_node.start_mark.line + 1
            if key in seen:
                raise DuplicateKeyError(
                    f"{key_node.start_mark.name}: duplicate key {key!r} at line {line} (first defined at line {seen[key]})"
                )
            seen[key] = line
        return super().construct_mapping(node, deep=deep)


def load_config(path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.load(f, Loader=_StrictLoader)


def compat_hash(cfg: dict) -> str:
    sub = {k: cfg.get(k) for k in COMPAT_KEYS}
    if isinstance(sub.get("adapter"), dict):
        sub["adapter"] = {k: v for k, v in sub["adapter"].items() if k not in MACHINE_KEYS}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:12]


def snap_to_batch(target: int, batch: int) -> int:
    """PPO-style algorithms train in whole rollouts of n_envs*n_steps transitions,
    so a requested checkpoint is rounded UP to a multiple of the rollout size."""
    return max(batch, -(-int(target) // batch) * batch)


def checkpoint_grid(grid, batch: int) -> list:
    return sorted({snap_to_batch(c, batch) for c in grid if c > 0})
