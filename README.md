# rlsignal (CrowdNav edition)

Standalone tool that answers: **how many training transitions does a proxy run need before the *ranking* of candidate
reward functions is trustworthy?** (Stage-II budget calibration.) It trains PPO (SB3) on CrowdNav
(vita-epfl/CrowdNav) with several candidate rewards plus negative controls, stores every evaluation, and analyses the
stored data. The adapter interface (`rlsignal/adapters/base.py`) is environment independent; a generic Gymnasium
adapter is included.

## Quick start
```bash
git clone https://github.com/vita-epfl/CrowdNav.git            # ./CrowdNav next to pyproject.toml (or set $CROWDNAV_PATH)
pip install matplotlib cython cmake
export CMAKE_POLICY_VERSION_MINIMUM=3.5                          # Windows: set CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install git+https://github.com/sybrenstuvel/Python-RVO2.git   # needs a C++ compiler (Windows: MSVC Build Tools)
pip install -e ".[sb3,crowdnav,dev]"
pytest -q                                                      # unit + runner plumbing tests; CrowdNav adapter tests need ./CrowdNav and rvo2
rlsignal pilot-cost configs/crowdnav_pilot.yaml                # measure transitions/s, project total cost
rlsignal run        configs/crowdnav_pilot.yaml                # resumable; extending the grid reuses finished work
rlsignal analyze    configs/crowdnav_pilot.yaml                # writes results/<name>/analysis.json + report.md
```

## CLI
Every command takes `<config.yaml>` plus `--out <results root>` (default `results`) and `--force`.

| command | what it does |
|---|---|
| `pilot-cost` | short timed training run; prints projected cost of the full grid |
| `run` | trains / resumes all (reward, seed) runs and stores evaluations under `<out>/<name>/runs/` |
| `analyze` | analyses stored evaluations, writes `analysis.json` and `report.md` |
| `all` | `run` followed by `analyze` |
| `estimate` | `analyze`, then prints only the estimator decision as JSON |
| `refine` | evaluates extra budgets from stored snapshots near the stopping boundary, then re-analyses |
| `preregister` | freezes a hold-out validation protocol (see below) |
| `validate` | evaluates a hold-out experiment against its pre-registration |

`--force` quarantines old runs when the compatibility hash changed (`run`) or overwrites a registration (`preregister`).

## Configuration
Top-level sections: `name`, `adapter`, `rewards`, `algo`, `eval`, `schedule`, `compute`, `features`, `detection`,
`ranking` (legacy), `budget`, `reference`, `behavior_probe`, `estimator`, `holdout`. Only `adapter`, `rewards`, `algo`
and `eval` enter the compatibility hash (machine-specific `adapter.crowdnav_path` excluded), so grids, thresholds and
analysis settings can change without invalidating stored runs. `load_config` rejects duplicate YAML keys.
See `configs/crowdnav_pilot.yaml` for a fully documented example.

## What changed versus v0.1 (and why) — see `legacy_results/` for the run that exposed the problems
| Problem in v0.1 | Fix |
|---|---|
| Zero-reward control drifted away from pi_0 exactly like `standard`, yet "stable signal at 4096" was reported | Controls (`null_control`, `random_control`) are the drift reference. **Signal = distinguishable from every control** (seed-level energy test + separation ratio + persistence). "Change vs pi_0" is still reported but labelled as including drift |
| 20 null statistics were only 4 unique values (pi_0 depends on the seed, not the reward) | Null bank deduplicated per **seed**; K=8 null sets per run (`eval.n_null_sets`), report prints independent-seed count and unique values |
| Null used different env seeds than the tested statistic; unpaired test on paired data | Matched null (same env seeds as `E`, different action-sampling stream) + **paired** permutation test (labels swapped within scenario) |
| Budget came from "80 % of reward pairs distinguishable" | Removed as a criterion (kept as diagnostic). Replaced first by the legacy **ranking mode** (Kendall tau between the ranking at budget T and a horizon reference on disjoint seed halves), now superseded by the reference-based estimator (below) |
| Persistence counted grid indices (geometric grid) | Persistence in **steps** (`persist_ratio`: signal must hold up to ratio x onset) |
| 4 seeds, bootstrap CIs over 35 distinct resamples, `p=1.00` printed | 8 seeds in the config; CIs withheld below `min_seeds_ci`; key renamed `boot_detect_frac` |
| `delta_l=0.5` hid that `cautious_social` learned its objective | Floor per reward from the null bank; `delta_l=0.25` |
| 13 collinear features, length-confounded `act_distinct_frac` | `features.exclude` (outcome indicators that sum to 1, distinct-action fraction) |
| Hard-coded Windows path inside the compat hash and report | Path from `$CROWDNAV_PATH` / `./CrowdNav`; machine keys excluded from the hash |
| Eval before `save`; evaluation consumed the training RNG | Save, then record; `Trainer.eval_context` isolates the sampling RNG (SB3: `torch.random.fork_rng`) |
| `ego` observations with world-frame actions | Actions are goal-aligned too in `ego` mode |
| Sparse `standard` reward unlearnable at pilot scale | Pilot layout `human_num: 3`, `n_envs: 8`, `ent_coef: 0.01`; 9 rewards incl. a random-reward control |

## Reading the report
- **vs controls** is the quantity to trust. `None` = not detected within the tested budget (censored), never an invented number.
- Recommended budget = minimum reliable budget x `safety_margin`, snapped up to a tested checkpoint (PPO trains in whole
  rollouts, so checkpoints are multiples of `n_envs * n_steps`). **Validate it** (hold-out section below): checkpoints of
  one long run only mimic a short run when nothing in training depends on the horizon (the SB3 trainer rejects LR schedules).

## Acceptance checks before trusting a number
1. Both controls: stable-signal rate over >= 20 control runs <= alpha.
2. Null bank: >= 80 unique values; independent-seed count printed.
3. A positive control (a reward that must learn, e.g. `goal_shaping`) is detected with a sensible CI.
4. Recommended budget validated out of sample (`preregister` / `run` / `validate`); result stable when 2 seeds are dropped or the grid changes.

## Known limits
- Scale of standardised features is fitted on the whole experiment: adding a reward changes the floors. Re-run analysis after changing the reward set.
- Resume is not bit-exact for training RNG streams (trainer is reseeded on load).
- The Stable-Baselines3 trainer (`rlsignal/trainers/sb3_ppo.py`) and the CrowdNav adapter are not exercised by the
  default unit tests (they use a toy env and a dummy trainer); run `rlsignal pilot-cost` first on a new machine.


## Estimator v2: minimum budget for reliable reward ranking

`budget.mode: estimator` (the default when an `estimator` section exists; see `configs/crowdnav_pilot.yaml`, sections
`reference`, `behavior_probe`, `estimator`). The legacy `mode: ranking` (`ranking.reference_weights`, Kendall tau on seed
halves, `ranking.ranking_analysis`) and `mode: screening` are kept unchanged for backward compatibility.

**Four questions, kept separate in code, `analysis.json` and `report.md`:** (1) has each policy's behaviour stabilised
(change between consecutive checkpoints vs the seed-to-seed noise floor), (2) has it approached *its own reward's*
reference (diagnostic), (3) is the reward ranking at budget t reliable, (4) is the reference (ranking) itself trustworthy.
The recommended budget answers (3) only, and only if (4) allows it.

**Algorithm.**
1. *Reference* = the policies of the same reward trained to `reference.budget`, all seeds, evaluated on a larger,
   disjoint scenario set (`eval_set "R"`). `reference.py` audits each one: between-seed SD of the primary criterion,
   cross-seed behavioural dispersion (JS on probe states, else bias-corrected pooled-action JS), and drift between
   `ref_trend_fraction x reference` and the reference (paired over seeds, t-interval, Bonferroni over rewards).
   Performance instability blocks the decision; behavioural dispersion is reported and blocks only with
   `ref_behavior_blocks`. `uncertain_reference: block | exclude_pairs | ignore_with_flag`.
2. *Ranking rule* is declared up front: `criteria` with mandatory `direction`, `rule: single | weighted | pareto`.
   Incompatible metrics are never added silently (`pareto` = front index, ties within a front).
3. *Reliability at budget t* (`ranking.ranking_reliability`): per replicate, seeds are split into **disjoint**
   experiment / reference groups (no shared lineage between the ranked checkpoint and the reference), resampled with
   replacement inside each group, and evaluation scenarios are resampled with replacement (nested). Reported: Kendall
   tau-b (undefined/too-few-items counted as failures, never as 0), reproduction probability of every pair the reference
   resolves (>= `resolve_prob`), P(best pick lies in the reference top set).
4. *Stopping*: at an eligible budget (`t <= reference / min_reference_ratio`) all of: P(tau>=`tau_threshold`)>=`prob`,
   >= `pair_fraction` of resolved pairs reproduced with prob >= `pair_prob`, P(top set)>=`top_set_prob`; persisting over
   `persist_k` looks and up to `persist_ratio` x the onset step; reference ranking reproducible at the reference budget;
   >= `min_seeds` seeds. Outcomes: `ranking_sufficient`, `ranking_sufficient_reference_uncertain`,
   `insufficient_evidence_at_max_budget`, `reference_uncertain`, `reference_ranking_unreliable`,
   `reference_degenerate_ties`, `insufficient_seeds`, `insufficient_candidates`, `invalid_configuration`.
5. *Repeated testing*: the selected budget is the first of several looks, so nominal probabilities are optimistic.
   Persistence is required and a Bonferroni-tightened result is always reported next to the nominal one
   (`looks_correction: bonferroni` makes it the decision rule). Neither is an exact family-wise guarantee.

**Coarse-to-fine evaluation without retraining.** Set `schedule.snapshot_grid` (fine) while `schedule.checkpoints`
stays coarse; `rlsignal refine <config>` evaluates extra snapshots near the stopping boundary in a fresh trainer,
never touching the training run. A grid point inserted *behind* a run's trained frontier is skipped with a message
(evaluating it would label the later model with the earlier budget).

**Not claimed.** No policy is declared optimal/converged because of its budget, plateau, or similarity to a reference.
Budgets are conditional on the design (seeds, episodes, rule). Defaults are placeholders to calibrate.

**Limitations.** Probe states come from random-action rollouts (not on-policy); the stored pilot
(`results/crowdnav_pilot_v2`) predates the estimator and has no `R` set or probes; with 8 seeds the split is 4/4
(70 partitions); thresholds are uncalibrated. `python experiments/reanalyze_pilot.py` reproduces the real-data
re-analysis in `experiments/pilot_v2_reanalysis/`.


## Making a recommended budget defensible: pre-registered hold-out validation

The estimator's probabilities are in-sample. To test the recommended budget out of sample (new seeds, new scenarios,
true short runs), use `configs/crowdnav_holdout.yaml` (training setup is checked to equal the reference experiment):

```bash
rlsignal preregister configs/crowdnav_holdout.yaml --out results   # freezes rule, targets, seeds AND the reference truth (hashed)
rlsignal run         configs/crowdnav_holdout.yaml --out results   # 7 rewards x 16 NEW seeds, stops at 32768 (8192 = negative control)
rlsignal validate    configs/crowdnav_holdout.yaml --out results   # -> holdout_report.md / holdout_validation.json
```
Criteria (`holdout.targets`): P(tau-b >= 0.6) >= 0.8, >= 90% of reference-resolved pairs reproduced with prob >= 0.8,
and P(regret <= 0.05 success) >= 0.8. Verdict per budget: `confirmed` (95% CI lower bound meets the target),
`consistent_not_confirmed` (point estimate only), `refuted`. `validate` refuses to run if the registration or the config
changed. The 8192 budget must NOT be confirmed; if it is, the test has no power. The registration file is written to
`<out>/<holdout name>.preregistration.json`.

## Project layout
| path | contents |
|---|---|
| `rlsignal/cli.py` | `rlsignal` command-line entry point |
| `rlsignal/config.py` | YAML loading (duplicate-key check), compatibility hash, checkpoint grid |
| `rlsignal/runner.py` | resumable training/evaluation runs, reference evaluation, probes, snapshots, `refine` |
| `rlsignal/rollout.py` | fixed-protocol policy evaluation producing per-episode summaries |
| `rlsignal/data.py` | `RunData`, experiment loader, standardised episode features |
| `rlsignal/stats.py` | energy-distance permutation tests, BH adjustment, JS divergence, change point |
| `rlsignal/detectors.py` | onset detectors (vs pi_0, vs controls, own-return learning), discrimination diagnostic |
| `rlsignal/analyze.py` | analysis pipeline, budget recommendation, `report.md` writer |
| `rlsignal/params.py` | `EstimatorParams` and `Criterion` (all estimator thresholds) |
| `rlsignal/metrics.py` | criterion extraction, Kendall tau-b with explicit undefined cases, ranking rules |
| `rlsignal/ranking.py` | legacy `ranking_analysis` and the estimator's `ranking_reliability` |
| `rlsignal/reference.py` | audit of reference policies (stability, dispersion, drift) |
| `rlsignal/behavior.py` | behavioural convergence / stabilisation diagnostics |
| `rlsignal/probe.py` | fixed probe states and policy fingerprints |
| `rlsignal/estimator.py` | reference-based budget estimator (decision + four questions) |
| `rlsignal/adapters/` | environment adapters (`base`, `crowdnav`, `gym_generic`) |
| `rlsignal/trainers/` | trainer contract and the SB3 PPO trainer |
| `rlsignal/validation/holdout.py` | pre-registered hold-out validation |
| `rlsignal/validation/synthetic.py` | synthetic runs with planted ground truth (test fixtures) |
| `configs/` | pilot, full-horizon and hold-out configurations |
| `experiments/` | re-analysis script and its stored output for the real pilot |
| `legacy_results/` | v0.1 pilot kept as evidence (not loadable by the v2 loader) |
| `tests/` | pytest suite |
