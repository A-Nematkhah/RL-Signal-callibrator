from .base import EnvAdapter


def build_adapter(cfg: dict) -> EnvAdapter:
    a = dict(cfg["adapter"])
    name = a.pop("name")
    if name == "crowdnav":
        from .crowdnav import CrowdNavAdapter
        return CrowdNavAdapter(**a)
    if name == "gymnasium":
        from .gym_generic import GymAdapter
        return GymAdapter(**a)
    raise ValueError(f"unknown adapter {name!r}")
