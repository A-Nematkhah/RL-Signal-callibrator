# Learning-signal report: crowdnav_pilot

Adapter: `{'name': 'crowdnav', 'crowdnav_path': 'C:/Users/nematkhah/Downloads/rl-signal-calibrator/CrowdNav', 'kinematics': 'holonomic', 'n_speeds': 5, 'n_rotations': 16, 'obs_mode': 'ego', 'env_config': {}}`; algorithm: `ppo_sb3`; compat hash `e39931db38b0`

## Calibration (null bank: pi_0 vs pi_0 on disjoint eval seeds)
- behavioural effect floor (energy distance, 0.95): 0.3692
- learning effect floor (return z): 0.500
- null false-positive rate of the permutation test: 0.050 (n=20)

## Onsets (checkpoint = total transitions; `None` = not detected within tested budget)

| reward | seeds | behavioural change | stable signal | learning onset | direction | change point |
|---|---|---|---|---|---|---|
| cautious_social | 4 | 4096 (p=1.00, ci=[4096, 12032]) | 4096 (p=1.00, ci=[4096, 12032]) | None | None | None |
| goal_shaping | 4 | 2048 (p=1.00, ci=[1024, 2048]) | 2048 (p=1.00, ci=[1024, 2048]) | 1024 (p=1.00, ci=[1024, 1024]) | increase | 2048 |
| null_control | 4 | 4096 (p=1.00, ci=[4096, 8192]) | 4096 (p=1.00, ci=[4096, 8192]) | None | None | None |
| reckless_fast | 4 | 2048 (p=1.00, ci=[1024, 2048]) | 2048 (p=1.00, ci=[1024, 2048]) | 1024 (p=1.00, ci=[1024, 1024]) | increase | 2048 |
| standard | 4 | 4096 (p=1.00, ci=[4096, 8192]) | 4096 (p=1.00, ci=[4096, 8192]) | None | None | None |

## Reward discrimination

- first: 120064, stable: None
- never distinguished pairs: ['cautious_social|null_control', 'cautious_social|standard', 'null_control|standard']
- smallest attainable per-pair p ~0.0286 with 4 seeds per reward

## Budget

- mode: screening
- minimum_reliable_budget: None
- recommended_stage2_steps: None
- reason: reward-discrimination criterion not met and persistent within the tested budget
- cautious_social: no return change beyond the null-calibrated effect floor
- null_control: no return change beyond the null-calibrated effect floor
- standard: no return change beyond the null-calibrated effect floor
