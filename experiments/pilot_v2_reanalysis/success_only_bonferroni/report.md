# Learning-signal report: crowdnav_pilot_v2

Algorithm: `ppo_sb3`; compat hash `7a293e47a699`

## Calibration (matched null bank: pi_0 re-evaluated on the same env seeds, different action sampling)
- behavioural effect floor (energy distance, q=0.95): 0.1815
- null tests: 64 from 8 independent seeds (64 unique values); false-positive rate of the paired test: 0.000
- controls used as the drift reference: ['null_control', 'random_control']
- learning effect floor (return z) per reward: cautious_social=0.48, goal_shaping=0.50, null_control=0.25, progress_social=0.51, random_control=0.25, reckless_fast=0.41, speed_only=0.29, standard=0.43, success_collision=0.42

## Onsets (checkpoint = total transitions; `None` = not detected within the tested budget)

`vs pi_0` includes pure training drift; `vs controls` is the quantity to trust.

| reward | seeds | change vs pi_0 | stable vs pi_0 | stable vs controls | learning onset | direction |
|---|---|---|---|---|---|---|
| cautious_social | 8 | 16384 (ci=[8192, 16384]) | 16384 (ci=[8192, 16384]) | 16384 | 200704 (ci=[16384, 200704]) | increase |
| goal_shaping | 8 | 2048 (ci=[2048, 2048]) | 2048 (ci=[2048, 2048]) | 2048 | 2048 (ci=[2048, 2048]) | increase |
| null_control * | 8 | 65536 (ci=[16384, 100352]) | 65536 (ci=[16384, 100352]) | n/a (control) | None | None |
| progress_social | 8 | 2048 (ci=[2048, 2048]) | 2048 (ci=[2048, 2048]) | 2048 | 2048 (ci=[2048, 2048]) | increase |
| random_control * | 8 | 65536 (ci=[8192, 100352]) | 65536 (ci=[16384, 100352]) | n/a (control) | None | None |
| reckless_fast | 8 | 2048 (ci=[2048, 2048]) | 2048 (ci=[2048, 2048]) | 2048 | 2048 (ci=[2048, 2048]) | increase |
| speed_only | 8 | 4096 (ci=[4096, 4096]) | 4096 (ci=[4096, 4096]) | 4096 | 4096 (ci=[2048, 4096]) | increase |
| standard | 8 | 16384 (ci=[16384, 65536]) | 16384 (ci=[16384, 65536]) | 65536 | 200704 (ci=[65536, 300032]) | increase |
| success_collision | 8 | 16384 (ci=[16384, 32768]) | 16384 (ci=[16384, 32768]) | 32768 | 100352 (ci=[65536, 300032]) | increase |

(* = control reward)

## Reward discrimination (diagnostic only, not a budget criterion)

- never distinguished pairs: ['cautious_social|standard', 'cautious_social|success_collision', 'goal_shaping|progress_social', 'null_control|random_control', 'standard|success_collision']
- smallest attainable per-test p ~0.0002 with 8 vs 8 seeds

## Ranking agreement with the horizon reference

- reference step: 300032; reference reliability (median tau between seed halves): 0.90

| step | median tau | P(tau>=thr) | P(top-1 correct) |
|---|---|---|---|
| 2048 | 0.05 | 0.00 | 0.14 |
| 4096 | -0.24 | 0.00 | 0.00 |
| 8192 | -0.14 | 0.00 | 0.00 |
| 16384 | 0.62 | 0.65 | 0.47 |
| 32768 | 0.52 | 0.45 | 0.65 |
| 65536 | 0.71 | 0.92 | 0.34 |
| 100352 | 0.81 | 0.96 | 0.57 |
| 150528 | 0.81 | 1.00 | 0.34 |
| 200704 | 0.81 | 1.00 | 0.56 |
| 300032 | 0.90 | 1.00 | 0.34 |
- first agreeing budget: 65536, stable: 65536; 

## Reference-based budget estimator (answers four SEPARATE questions)

- **decision: `insufficient_evidence_at_max_budget`**; minimum reliable budget: None; recommended: None; ranking agreement with the reference did not become persistent at an eligible budget (eligible: budget <= reference/3.0; persistence: step ratio 2.0)
- design: 8 seeds split into 4 experiment / 4 disjoint reference seeds, 7 candidates, 30 eval episodes, 500 replicates (70 distinct partitions); reference step 300032; rule `single` on success (higher is better)

### 3. Ranking reliability vs the reference ranking (tau-b undefined cases counted as failures)

| step | median tau-b | P(tau>=thr) | P(undefined) | resolved pairs reproduced | P(best in ref top set) | agree |
|---|---|---|---|---|---|---|
| 2048 | n/a | 0.00 | 1.00 | 0.00 | 0.31 | False |
| 4096 | n/a | 0.00 | 1.00 | 0.00 | 0.32 | False |
| 8192 | 0.49 | 0.10 | 0.63 | 0.00 | 0.56 | False |
| 16384 | 0.82 | 0.95 | 0.00 | 1.00 | 1.00 | False |
| 32768 | 0.86 | 0.94 | 0.00 | 1.00 | 1.00 | False |
| 65536 | 0.79 | 0.95 | 0.00 | 1.00 | 1.00 | False |
| 100352 | 0.78 | 0.91 | 0.00 | 1.00 | 1.00 | False |
| 150528 | 0.78 | 0.90 | 0.00 | 1.00 | 1.00 | False |
| 200704 | 0.78 | 0.95 | 0.00 | 1.00 | 1.00 | False |
| 300032 | 0.70 | 0.86 | 0.00 | 1.00 | 1.00 | False |

- reference ranking resolves 8/21 pairs; top set: ['goal_shaping', 'progress_social']
- stable budget: nominal 16384, Bonferroni-tightened None (over 6 looks)

### 4. Reference audit (large budget does not imply convergence)

| reward | seeds | primary mean | between-seed SD | performance | behaviour | reasons |
|---|---|---|---|---|---|---|
| cautious_social | 8 | 0.000 | 0.000 | stable | uncertain | reference behaviours differ across seeds: mean JS 0.214 bits > 0.15 |
| goal_shaping | 8 | 0.742 | 0.056 | stable | uncertain | reference behaviours differ across seeds: mean JS 0.179 bits > 0.15 |
| progress_social | 8 | 0.750 | 0.053 | stable | uncertain | reference behaviours differ across seeds: mean JS 0.169 bits > 0.15 |
| reckless_fast | 8 | 0.008 | 0.015 | stable | stable | - |
| speed_only | 8 | 0.000 | 0.000 | stable | uncertain | reference behaviours differ across seeds: mean JS 0.277 bits > 0.15 |
| standard | 8 | 0.000 | 0.000 | stable | uncertain | reference behaviours differ across seeds: mean JS 0.215 bits > 0.15 |
| success_collision | 8 | 0.054 | 0.153 | uncertain | uncertain | reference seeds disagree on success: between-seed SD 0.153 > 0.15; reference behaviours differ across seeds: mean JS 0.219 bits > 0.15 |

### 1-2. Behaviour (diagnostic only; similarity to a reference is not quality)

| reward | status | stable step | per-metric stable step (threshold) |
|---|---|---|---|
| cautious_social | not_converged_within_max_budget | None | action_marginal_js: 2048 (0.214), feature_energy: None (1.217), task_gap: 2048 (0.100) |
| goal_shaping | not_converged_within_max_budget | None | action_marginal_js: 8192 (0.179), feature_energy: None (0.419), task_gap: None (0.067) |
| progress_social | not_converged_within_max_budget | None | action_marginal_js: None (0.169), feature_energy: None (0.544), task_gap: None (0.064) |
| reckless_fast | not_converged_within_max_budget | None | action_marginal_js: None (0.120), feature_energy: None (0.344), task_gap: None (0.014) |
| speed_only | not_converged_within_max_budget | None | action_marginal_js: 32768 (0.277), feature_energy: None (0.829), task_gap: 2048 (0.100) |
| standard | not_converged_within_max_budget | None | action_marginal_js: 2048 (0.215), feature_energy: None (1.004), task_gap: 2048 (0.100) |
| success_collision | converged | 100352 | action_marginal_js: 2048 (0.219), feature_energy: 100352 (1.564), task_gap: 100352 (0.108) |

### Caveats

- reference behaviour differs across seeds for: cautious_social, goal_shaping, progress_social, speed_only, standard, success_collision (reported; does not block a task-criterion ranking)
- no dedicated reference evaluation (eval_set 'R') stored: the reference ranking reuses the same evaluation scenarios as the budgets; scenario noise is then shared and agreement is somewhat optimistic
- The first budget that passes is selected among several looks, so nominal probabilities are optimistic for the selected budget (winner's curse). Persistence over later budgets is required, and the Bonferroni-tightened result is reported alongside; neither is an exact family-wise guarantee.

> No policy is claimed optimal or converged because of its budget, a plateau, or similarity to a reference. Reference policies are large-budget policies of the same reward, audited but not proven converged. The budget is conditional on the experiment design (seeds, episodes, ranking rule).

## Budget

- mode: estimator
- minimum_reliable_budget: None
- recommended_stage2_steps: None
- reason: ranking agreement with the reference did not become persistent at an eligible budget (eligible: budget <= reference/3.0; persistence: step ratio 2.0)
- null_control: control run drifts away from pi_0 (expected: this is the drift level other rewards must exceed)
- random_control: control run drifts away from pi_0 (expected: this is the drift level other rewards must exceed)
