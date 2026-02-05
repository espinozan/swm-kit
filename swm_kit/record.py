from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from .world import make_world, set_policy


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_dataset(env: str, out: str, episodes: int = 100, seed: int = 0) -> Path:
    _set_seed(seed)
    out_path = Path(out)
    world = make_world(env=env, num_envs=1, image_shape=(64, 64, 3))
    policy = swm.RandomPolicy()
    set_policy(world, policy)
    world.record_dataset(out_path.as_posix(), episodes=episodes, seed=seed)
    return out_path
