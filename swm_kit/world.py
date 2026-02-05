from typing import Any, Iterable, Tuple

import stable_worldmodel as swm


def make_world(env: str, num_envs: int = 1, image_shape: Tuple[int, int, int] = (64, 64, 3)) -> swm.World:
    return swm.World(env=env, num_envs=num_envs, image_shape=image_shape)


def set_policy(world: swm.World, policy: Any) -> None:
    world.set_policy(policy)


def step(world: swm.World) -> Iterable[Any]:
    return world.step()


def infos(world: swm.World) -> Any:
    return world.infos()
