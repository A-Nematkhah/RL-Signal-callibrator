def build_trainer(cfg, adapter, reward_spec, seed):
    name = cfg["algo"]["name"]
    if name == "ppo_sb3":
        from .sb3_ppo import SB3PPOTrainer
        return SB3PPOTrainer(adapter, reward_spec, seed, cfg["algo"])
    raise ValueError(f"unknown algorithm {name!r}")
