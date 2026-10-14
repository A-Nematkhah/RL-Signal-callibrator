"""Parameters of the reference-based budget estimator. Every threshold lives here (and in the YAML `estimator`
section); nothing is hard-coded in the algorithms. Defaults are documented placeholders to be calibrated per
environment -- they are NOT claimed to be universally correct."""
from __future__ import annotations
from dataclasses import dataclass, field, fields


@dataclass
class Criterion:
    """One evaluation criterion. `direction` is mandatory so that a ranking is never built on an implicit sign."""
    name: str
    source: str = "metric"          # metric -> ep["metrics"][key] | component -> ep["reward_components"][key]
                                    # field  -> ep[key] (e.g. return, length)
    key: str | None = None
    direction: str = "higher"       # higher | lower  (is a larger value better or worse?)
    weight: float = 1.0             # only used by rule == "weighted"
    optional: bool = False          # an optional criterion missing from the data is skipped (with a note)

    def __post_init__(self):
        if self.direction not in ("higher", "lower"):
            raise ValueError(f"criterion {self.name!r}: direction must be 'higher' or 'lower', got {self.direction!r}")
        if self.source not in ("metric", "component", "field"):
            raise ValueError(f"criterion {self.name!r}: unknown source {self.source!r}")
        self.key = self.key or self.name

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "higher" else -1.0


def default_criteria():
    # Pre-registered default for crowd navigation: task success first, collisions as a second (lower-is-better)
    # objective. They are combined by Pareto dominance, never by silently adding incompatible units.
    return [Criterion("success", "metric", "success", "higher"),
            Criterion("collision", "metric", "collision", "lower")]


@dataclass
class EstimatorParams:
    # ---- ranking rule
    criteria: list = field(default_factory=default_criteria)
    rule: str = "pareto"                 # single | weighted | pareto
    pareto_eps: float = 0.0              # a criterion difference <= eps is not a dominance
    # ---- resampling design (seeds = independent training runs; episodes nested in policies)
    n_exp_seeds: int | None = None       # seeds in the simulated "experiment" group; None -> n_seeds // 2
    n_boot: int = 500
    resample_episodes: bool = True       # nested bootstrap of evaluation episodes inside each policy
    seed: int = 0
    # ---- agreement criteria at a budget
    tau_threshold: float = 0.6
    prob: float = 0.8                    # P(criterion holds) required at a budget
    min_candidates_for_tau: int = 3      # fewer items: tau-b is reported as not applicable
    resolve_prob: float = 0.9            # a reference pair is "resolved" if one order has this probability
    pair_prob: float = 0.8               # resolved pairs must be reproduced with at least this probability ...
    pair_fraction: float = 0.9           # ... for at least this fraction of resolved pairs
    top_set_prob: float | None = 0.8     # P(selected best candidate lies in the reference top set); None = off
    # ---- stopping rule
    persist_k: int = 2
    persist_ratio: float | None = 2.0    # criteria must hold up to ratio x onset step (step rule, grid independent)
    looks_correction: str = "none"       # none | bonferroni (probabilities tightened for the number of looks)
    min_reference_ratio: float = 3.0     # a budget is eligible only if reference_budget >= ratio * budget
    min_seeds: int = 6
    min_eval_episodes: int = 30
    safety_margin: float = 1.5
    # ---- reference policies
    reference_budget: int | None = None  # None -> largest evaluated budget
    require_stable_reference: bool = True    # legacy switch: False == uncertain_reference 'ignore_with_flag'
    uncertain_reference: str = "block"       # block | exclude_pairs | ignore_with_flag (what an uncertain reference does)
    ref_max_between_seed_sd: float = 0.15   # primary-criterion SD across reference seeds (metric units)
    ref_max_probe_js: float = 0.15          # mean cross-seed JS divergence of references (bits)
    ref_behavior_blocks: bool = False       # behavioural dispersion of references also blocks the ranking decision
    ref_trend_fraction: float = 0.5         # compare the reference with the budget closest to fraction x ref
    ref_max_trend: float = 0.05             # |change| of the primary criterion over that span that counts as drift
    # ---- behavioural convergence (diagnostic, separate from ranking)
    behavior_metrics: tuple = ("probe_js", "action_marginal_js", "feature_energy", "task_gap")
    behavior_threshold_mode: str = "floor_relative"   # absolute | floor_relative
    behavior_floor_factor: float = 1.0
    behavior_thresholds: dict = field(default_factory=lambda: {"probe_js": 0.05, "action_marginal_js": 0.02,
                                                                "task_gap": 0.10, "feature_energy": None})
    behavior_ci: float = 0.95
    behavior_min_seeds_ci: int = 6

    @classmethod
    def from_cfg(cls, d):
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names and k != "criteria"}
        if d.get("criteria") is not None:
            kw["criteria"] = [c if isinstance(c, Criterion) else Criterion(**c) for c in d["criteria"]]
        if "behavior_metrics" in kw:
            kw["behavior_metrics"] = tuple(kw["behavior_metrics"])
        if "behavior_thresholds" in kw:
            merged = dict(cls().behavior_thresholds)
            merged.update(kw["behavior_thresholds"])
            kw["behavior_thresholds"] = merged
        p = cls(**kw)
        if "uncertain_reference" not in d and not p.require_stable_reference:
            p.uncertain_reference = "ignore_with_flag"
        if p.uncertain_reference not in ("block", "exclude_pairs", "ignore_with_flag"):
            raise ValueError(f"unknown uncertain_reference {p.uncertain_reference!r}")
        if p.rule not in ("single", "weighted", "pareto"):
            raise ValueError(f"unknown ranking rule {p.rule!r}")
        if p.rule == "single" and len(p.criteria) != 1:
            raise ValueError("rule 'single' needs exactly one criterion")
        if p.looks_correction not in ("none", "bonferroni"):
            raise ValueError(f"unknown looks_correction {p.looks_correction!r}")
        return p
