from pathlib import Path

import stable_worldmodel as swm

from .world import make_world, set_policy


def record_dataset(env: str, out: str, episodes: int = 100, seed: int = 0) -> Path:
    out_path = Path(out)
    world = make_world(env=env, num_envs=1, image_shape=(64, 64, 3))
    policy = swm.RandomPolicy()
    set_policy(world, policy)
    world.record_dataset(out_path.as_posix(), episodes=episodes, seed=seed)
    return out_path
