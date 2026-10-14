"""Re-analyse the STORED real pilot (PPO on CrowdNav, results/crowdnav_pilot_v2) with the reference-based estimator.

This is a re-analysis of existing evaluations, not new RL training. The stored pilot predates the estimator, so:
  * no dedicated reference evaluation (eval_set "R") exists -> the reference ranking reuses the 30 evaluation scenarios;
  * no probe fingerprints exist -> policy-level JS is unavailable; pooled-action JS, feature energy and task gap are used.
Usage: python experiments/reanalyze_pilot.py [results_dir] [out_dir]"""
import json
import shutil
import sys
from pathlib import Path
import yaml
from rlsignal.analyze import analyze_experiment

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/crowdnav_pilot_v2")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "experiments/pilot_v2_reanalysis")
cfg = yaml.safe_load(open("configs/crowdnav_pilot.yaml"))
VARIANTS = {
    "pareto_success_collision": cfg["estimator"],
    "success_only": {**cfg["estimator"], "rule": "single",
                     "criteria": [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}]},
    "success_only_exclude_uncertain": {**cfg["estimator"], "rule": "single", "uncertain_reference": "exclude_pairs",
                                       "criteria": [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}]},
    "success_only_bonferroni": {**cfg["estimator"], "rule": "single", "uncertain_reference": "exclude_pairs",
                                "looks_correction": "bonferroni",
                                "criteria": [{"name": "success", "source": "metric", "key": "success", "direction": "higher"}]},
}
for name, est in VARIANTS.items():
    work = Path("/tmp") / f"reanalysis_{name}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    for f in ("manifest.json",):
        shutil.copy(src / f, work / f)
    shutil.copytree(src / "runs", work / "runs", ignore=shutil.ignore_patterns("model.zip"))
    res = analyze_experiment(work, estimator_cfg=est, budget_cfg={"mode": "estimator"})
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(work / "analysis.json", dest / "analysis.json")
    shutil.copy(work / "report.md", dest / "report.md")
    (dest / "estimator.json").write_text(json.dumps(res["estimator"], indent=2, default=str))
    d = res["estimator"]["decision"]
    print(f"{name}: {d['status']} min={d['minimum_reliable_budget']} rec={d['recommended_budget']} "
          f"provisional={d.get('provisional_budget_if_reference_were_trusted')}")
