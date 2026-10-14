"""Experiment runner: resumable, config-hash guarded, transition-accounted."""

from __future__ import annotations
import json
import shutil
import numpy as np
import time
from pathlib import Path

from . import __version__
from .adapters import build_adapter
from .config import checkpoint_grid, compat_hash
from .probe import build_probe_states, fingerprint
from .rollout import evaluate_policy
from .trainers import build_trainer


class IncompatibleConfigError(RuntimeError):
    pass


def _prepare_manifest(cfg, exp: Path, adapter, force):
    h = compat_hash(cfg)
    mp = exp / "manifest.json"
    if mp.exists():
        old = json.loads(mp.read_text(encoding="utf-8"))
        if old["compat_hash"] != h:
            if not force:
                raise IncompatibleConfigError(
                    f"stored results in {exp} were made with compat_hash={old['compat_hash']} "
                    f"but the config now hashes to {h} (adapter/rewards/algo/eval changed). "
                    "Use a new experiment name or pass force=True to quarantine the old runs."
                )
            shutil.move(str(exp / "runs"), str(exp / f"runs_stale_{old['compat_hash']}"))
    man = {
        "name": cfg["name"],
        "compat_hash": h,
        "config": cfg,
        "env_info": adapter.describe(),
        "rlsignal_version": __version__,
    }
    mp.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return man


def _done_records(path: Path):
    if not path.exists():
        return set()
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["checkpoint"], r["eval_set"], r.get("null_idx")) for r in recs}


def run_one(cfg, adapter, man, reward_id, seed, exp: Path, deadline=None, log=print):
    rspec = cfg["rewards"][reward_id]
    ev = cfg["eval"]
    kind, n_act = man["env_info"]["action_kind"], man["env_info"]["n_actions"]
    rdir = exp / "runs" / reward_id / f"seed{seed}"
    rdir.mkdir(parents=True, exist_ok=True)
    evals, model_path, state_path = rdir / "evals.jsonl", rdir / "model.zip", rdir / "state.json"
    done = _done_records(evals)
    trainer = build_trainer(cfg, adapter, rspec, seed)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"train_wall_s": 0.0}
    if model_path.exists():
        trainer.load(str(model_path))
    grid = checkpoint_grid(cfg["schedule"]["checkpoints"], trainer.rollout_size)
    snap_grid = checkpoint_grid(cfg["schedule"].get("snapshot_grid", []), trainer.rollout_size)
    keep_snap = bool(cfg["schedule"].get("keep_snapshots", False)) or bool(snap_grid)
    ref_cfg = cfg.get("reference") or {}
    ref_step = checkpoint_grid([ref_cfg["budget"]], trainer.rollout_size)[0] if ref_cfg.get("budget") else max(grid)
    seeds_E = [ev["eval_seed_base"] + i for i in range(ev["n_episodes"])]
    seeds_R = [int(ref_cfg.get("eval_seed_base", 50000)) + i for i in range(int(ref_cfg.get("n_episodes", 0)))]
    n_null = int(ev.get("n_null_sets", 8))
    rng_E = int(ev.get("eval_rng_seed", 12345))  # fixed policy-sampling stream for every E evaluation
    det = bool(ev.get("deterministic", False))
    pcfg = cfg.get("behavior_probe") or {}
    probe_obs = None
    if pcfg.get("enabled", False):
        probe_obs, pid = build_probe_states(
            adapter, int(pcfg.get("n_states", 64)), int(pcfg.get("seed_base", 20000)), int(pcfg.get("stride", 3))
        )
        (rdir / "probe").mkdir(exist_ok=True)
        idf = rdir / "probe" / "probe_id.txt"
        stored_pid = idf.read_text(encoding="utf-8").strip() if idf.exists() else None
        if stored_pid is not None and stored_pid != pid:
            raise IncompatibleConfigError(f"{rdir}: probe states changed (stored id {stored_pid} != {pid})")
        idf.write_text(pid, encoding="utf-8")

    def record(ckpt, eval_set, null_idx=None):
        if (ckpt, eval_set, null_idx) in done:
            return
        t0 = time.time()
        # E and the matched null use the SAME env seeds; they differ only in the action-sampling stream.
        # R (reference evaluation) uses a larger, DISJOINT scenario set and its own sampling stream.
        rng_seed = rng_E if eval_set == "E" else (rng_E + 5000 if eval_set == "R" else rng_E + 1000 + int(null_idx))
        eps = evaluate_policy(
            trainer, adapter, rspec, seeds_R if eval_set == "R" else seeds_E, det, kind, n_act, rng_seed=rng_seed
        )
        tw = state["train_wall_s"]
        rec = {
            "checkpoint": ckpt,
            "eval_set": eval_set,
            "null_idx": null_idx,
            "transitions": trainer.transitions,
            "n_envs": trainer.n_envs,
            "steps_per_env": trainer.transitions // trainer.n_envs,
            "train_wall_s": tw,
            "transitions_per_s": trainer.transitions / tw if tw > 0 else None,
            "eval_wall_s": time.time() - t0,
            "episodes": eps,
        }
        with open(evals, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def record_extras(ckpt):
        """Reference evaluation and probe fingerprint of the CURRENT model, labelled with the checkpoint it is."""
        if trainer.transitions != ckpt:
            return
        if seeds_R and ckpt == ref_step:
            record(ckpt, "R")
        if probe_obs is not None and not (rdir / "probe" / f"ckpt_{ckpt}.npy").exists():
            fp = fingerprint(trainer, probe_obs, kind, rng_E + 7000)
            if fp is not None:
                np.save(rdir / "probe" / f"ckpt_{ckpt}.npy", fp)

    def snapshot(ckpt):
        if keep_snap and trainer.transitions == ckpt:
            (rdir / "snapshots").mkdir(exist_ok=True)
            trainer.save(str(rdir / "snapshots" / f"ckpt_{ckpt}.zip"))

    if trainer.transitions == 0:
        record(0, "E")
        for k in range(n_null):
            record(0, "null", k)
        record_extras(0)
    for ckpt in sorted(set(grid) | set(snap_grid)):
        if trainer.transitions >= ckpt:
            if trainer.transitions == ckpt and ckpt in grid:
                record(ckpt, "E")  # no-op if already stored
                record_extras(ckpt)
            elif ckpt in grid and (ckpt, "E", None) not in done:
                # The run has already moved past this budget (a grid point inserted below the trained frontier):
                # evaluating the current model under this label would be wrong. Use `rlsignal refine` on a snapshot.
                log(
                    f"[skip] {reward_id}/seed{seed}: budget {ckpt} is behind the trained frontier "
                    f"({trainer.transitions}); evaluate it from a snapshot with `refine`"
                )
            continue
        if deadline and time.time() > deadline:
            log(f"[budget] wall-clock budget exhausted at {reward_id}/seed{seed}; stopping")
            return False
        t0 = time.time()
        trainer.train(ckpt - trainer.transitions)
        state["train_wall_s"] += time.time() - t0
        trainer.save(str(model_path))  # save BEFORE recording: a crash in between must not leave an
        state_path.write_text(
            json.dumps(state), encoding="utf-8"
        )  # evaluation that belongs to a model that is later retrained
        snapshot(ckpt)
        if ckpt in grid:
            record(ckpt, "E")
            record_extras(ckpt)
        log(f"{reward_id}/seed{seed} ckpt={ckpt} tps={trainer.transitions / max(state['train_wall_s'], 1e-9):.0f}")
    return True


def _worker(args):
    cfg, reward_id, seed, exp, deadline = args
    import torch

    torch.set_num_threads(1)
    adapter = build_adapter(cfg)
    man = json.loads((Path(exp) / "manifest.json").read_text(encoding="utf-8"))
    return run_one(cfg, adapter, man, reward_id, seed, Path(exp), deadline, log=lambda *_: None)


def run_experiment(cfg, out_root, force=False, log=print):
    exp = Path(out_root) / cfg["name"]
    exp.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(cfg)
    man = _prepare_manifest(cfg, exp, adapter, force)
    comp = cfg.get("compute", {})
    max_wall = comp.get("max_wall_seconds")
    deadline = time.time() + max_wall if max_wall else None
    jobs = [(cfg, r, s, str(exp), deadline) for r in cfg["rewards"] for s in cfg["schedule"]["seeds"]]
    workers = int(comp.get("workers", 1))
    if workers <= 1:
        complete = True
        for _, r, s, _, _ in jobs:
            complete &= bool(run_one(cfg, adapter, man, r, s, exp, deadline, log))
            if deadline and time.time() > deadline:
                return exp, False
        return exp, complete
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_worker, jobs))
    return exp, all(results)


def estimate_cost(cfg, n_transitions=1024, log=print):
    """Pilot: time a short run to project the cost of the full grid."""
    adapter = build_adapter(cfg)
    rid = next(iter(cfg["rewards"]))
    tr = build_trainer(cfg, adapter, cfg["rewards"][rid], 0)
    n = max(tr.rollout_size, -(-n_transitions // tr.rollout_size) * tr.rollout_size)
    t0 = time.time()
    tr.train(n)
    tps = n / (time.time() - t0)
    kind, n_act = adapter.describe()["action_kind"], adapter.describe()["n_actions"]
    t1 = time.time()
    evaluate_policy(tr, adapter, cfg["rewards"][rid], range(5), False, kind, n_act, rng_seed=0)
    eval_s = (time.time() - t1) / 5 * cfg["eval"]["n_episodes"]
    sch = cfg["schedule"]
    horizon = max(sch["checkpoints"])
    runs = len(cfg["rewards"]) * len(sch["seeds"])
    n_ckpt = len(sch["checkpoints"]) + 2
    total_s = runs * (horizon / tps + n_ckpt * eval_s)
    return {
        "transitions_per_s": tps,
        "eval_s_per_checkpoint": eval_s,
        "runs": runs,
        "projected_total_hours": total_s / 3600,
        "projected_train_hours": runs * horizon / tps / 3600,
    }


def refine_experiment(cfg, out_root, steps, log=print):
    """Coarse-to-fine EVALUATION: evaluate additional budgets from stored snapshots (schedule.snapshot_grid /
    keep_snapshots). Nothing is trained and the main run's model, optimiser and RNG are never touched: a fresh
    trainer loads the snapshot and is discarded. Records use the same format as ordinary checkpoints."""
    exp = Path(out_root) / cfg["name"]
    adapter = build_adapter(cfg)
    man = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    kind, n_act = man["env_info"]["action_kind"], man["env_info"]["n_actions"]
    ev, pcfg = cfg["eval"], cfg.get("behavior_probe") or {}
    seeds_E = [ev["eval_seed_base"] + i for i in range(ev["n_episodes"])]
    added, missing = 0, []
    for rid, rspec in cfg["rewards"].items():
        for seed in cfg["schedule"]["seeds"]:
            rdir = exp / "runs" / rid / f"seed{seed}"
            done = _done_records(rdir / "evals.jsonl")
            for step in steps:
                snap = rdir / "snapshots" / f"ckpt_{step}.zip"
                if (step, "E", None) in done:
                    continue
                if not snap.exists():
                    missing.append(str(snap))
                    continue
                tr = build_trainer(cfg, adapter, rspec, seed)
                tr.load(str(snap))
                eps = evaluate_policy(
                    tr,
                    adapter,
                    rspec,
                    seeds_E,
                    bool(ev.get("deterministic", False)),
                    kind,
                    n_act,
                    rng_seed=int(ev.get("eval_rng_seed", 12345)),
                )
                rec = {
                    "checkpoint": step,
                    "eval_set": "E",
                    "null_idx": None,
                    "transitions": tr.transitions,
                    "n_envs": tr.n_envs,
                    "refined_from_snapshot": True,
                    "episodes": eps,
                }
                with open(rdir / "evals.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                if pcfg.get("enabled", False):
                    obs, _ = build_probe_states(
                        adapter,
                        int(pcfg.get("n_states", 64)),
                        int(pcfg.get("seed_base", 20000)),
                        int(pcfg.get("stride", 3)),
                    )
                    fp = fingerprint(tr, obs, kind, int(ev.get("eval_rng_seed", 12345)) + 7000)
                    if fp is not None:
                        (rdir / "probe").mkdir(exist_ok=True)
                        np.save(rdir / "probe" / f"ckpt_{step}.npy", fp)
                added += 1
                log(f"refined {rid}/seed{seed} @ {step}")
    return {"evaluated": added, "missing_snapshots": missing}
