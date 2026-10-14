import argparse
import json
from pathlib import Path
from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rlsignal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "analyze", "pilot-cost", "all", "estimate", "refine", "preregister", "validate"):
        p = sub.add_parser(name)
        p.add_argument("config")
        p.add_argument("--out", default="results")
        p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
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
    if a.cmd == "estimate":      # estimator only, printed as JSON (the full result is also in analysis.json)
        from .analyze import analyze_experiment
        res = analyze_experiment(Path(a.out) / cfg["name"], estimator_cfg=cfg.get("estimator"), budget_cfg=cfg.get("budget"))
        print(json.dumps((res.get("estimator") or {}).get("decision"), indent=2, default=str))
        return
    if a.cmd == "refine":        # evaluate extra budgets from stored snapshots near the stopping boundary
        from .analyze import analyze_experiment
        from .estimator import suggest_refinement_steps
        from .runner import refine_experiment
        exp = Path(a.out) / cfg["name"]
        res = analyze_experiment(exp, estimator_cfg=cfg.get("estimator"), budget_cfg=cfg.get("budget"))
        snaps = cfg["schedule"].get("snapshot_grid", [])
        steps = suggest_refinement_steps(res["estimator"], snaps, max_new=int((cfg.get("estimator") or {}).get("max_refine", 3)))
        print("refining budgets:", steps)
        if steps:
            print(refine_experiment(cfg, a.out, steps))
            analyze_experiment(exp, estimator_cfg=cfg.get("estimator"), budget_cfg=cfg.get("budget"))
        return
    if a.cmd in ("analyze", "all"):
        from .analyze import analyze_experiment
        analyze_experiment(Path(a.out) / cfg["name"], estimator_cfg=cfg.get("estimator"), budget_cfg=cfg.get("budget"))
        print(f"report: {Path(a.out) / cfg['name'] / 'report.md'}")


if __name__ == "__main__":
    main()
