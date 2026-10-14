import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _require_experiment(cfg, out) -> Path:
    exp = Path(out) / cfg["name"]
    if not (exp / "manifest.json").exists():
        _die(f"experiment directory not found or incomplete: {exp} (run `rlsignal run` first)")
    return exp


def _require_estimator(cfg) -> dict:
    est = cfg.get("estimator")
    if not est:
        _die("config has no `estimator:` section (needed by estimate / refine / analyze in estimator mode)")
    return est


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rlsignal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "analyze", "pilot-cost", "all", "estimate", "refine", "preregister", "validate"):
        p = sub.add_parser(name)
        p.add_argument("config")
        p.add_argument("--out", default="results")
        p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
    except Exception as e:  # noqa: BLE001 -- CLI surface: one-line error, exit 2
        _die(f"failed to load config {a.config}: {e}")
    if a.cmd == "pilot-cost":
        from .runner import estimate_cost

        print(json.dumps(estimate_cost(cfg), indent=2))
        return
    if a.cmd in ("run", "all"):
        from .runner import run_experiment

        exp, complete = run_experiment(cfg, a.out, force=a.force)
        print(f"experiment dir: {exp} complete={complete}")
    if a.cmd == "preregister":
        from .validation.holdout import preregister_holdout

        print("registered:", preregister_holdout(cfg, a.out, force=a.force))
        return
    if a.cmd == "validate":
        from .validation.holdout import validate_holdout

        doc = validate_holdout(cfg, a.out)
        print(json.dumps(doc["claim_supported"], indent=2))
        print("report:", Path(a.out) / cfg["name"] / "holdout_report.md")
        return
    if a.cmd == "estimate":
        from .analyze import analyze_experiment

        exp = _require_experiment(cfg, a.out)
        est_cfg = _require_estimator(cfg)
        try:
            res = analyze_experiment(exp, estimator_cfg=est_cfg, budget_cfg=cfg.get("budget"))
        except Exception as e:  # noqa: BLE001
            _die(f"estimate failed for {exp}: {e}")
        print(json.dumps((res.get("estimator") or {}).get("decision"), indent=2, default=str))
        return
    if a.cmd == "refine":
        from .analyze import analyze_experiment
        from .estimator import suggest_refinement_steps
        from .params import EstimatorParams
        from .runner import refine_experiment

        exp = _require_experiment(cfg, a.out)
        est_cfg = _require_estimator(cfg)
        snaps = cfg["schedule"].get("snapshot_grid") or []
        if not snaps and not cfg["schedule"].get("keep_snapshots"):
            _die(
                "refine needs stored snapshots: set schedule.snapshot_grid (or keep_snapshots: true) and re-run "
                "training so snapshots exist under runs/*/seed*/snapshots/"
            )
        try:
            res = analyze_experiment(exp, estimator_cfg=est_cfg, budget_cfg=cfg.get("budget"))
        except Exception as e:  # noqa: BLE001
            _die(f"refine: analysis of {exp} failed: {e}")
        if not res.get("estimator"):
            _die("refine: analysis produced no estimator section")
        steps = suggest_refinement_steps(res["estimator"], snaps, max_new=EstimatorParams.from_cfg(est_cfg).max_refine)
        print("refining budgets:", steps)
        if steps:
            out = refine_experiment(cfg, a.out, steps)
            print(out)
            if out.get("missing_snapshots") and not out.get("evaluated"):
                _die(
                    "no snapshots found for the requested budgets (missing: "
                    + ", ".join(out["missing_snapshots"][:5])
                    + ("..." if len(out["missing_snapshots"]) > 5 else "")
                    + "); re-run with schedule.snapshot_grid covering those steps"
                )
            analyze_experiment(exp, estimator_cfg=est_cfg, budget_cfg=cfg.get("budget"))
        return
    if a.cmd in ("analyze", "all"):
        from .analyze import analyze_experiment

        exp = _require_experiment(cfg, a.out)
        try:
            analyze_experiment(exp, estimator_cfg=cfg.get("estimator"), budget_cfg=cfg.get("budget"))
        except Exception as e:  # noqa: BLE001
            _die(f"analyze failed for {exp}: {e}")
        print(f"report: {exp / 'report.md'}")


if __name__ == "__main__":
    main()
