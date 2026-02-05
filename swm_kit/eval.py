from pathlib import Path
from typing import List

import numpy as np
import stable_worldmodel as swm
import torch

from .train import _build_mlp
from .world import make_world, set_policy


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(model_path: Path) -> torch.nn.Module:
    payload = torch.load(model_path, map_location="cpu")
    model = _build_mlp(payload["input_dim"], payload["output_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def evaluate(env: str, model_path: str, episodes: int = 10, seed: int = 0) -> float:
    _set_seed(seed)
    model = _load_model(Path(model_path))
    world = make_world(env=env, num_envs=1, image_shape=(64, 64, 3))

    planner = swm.CEMPlanner(model=model)
    policy = swm.WorldModelPolicy(planner=planner)
    set_policy(world, policy)

    rewards: List[float] = []
    for _ in range(episodes):
        episode_rewards = []
        for _, _, reward, done, _ in world.rollout():
            episode_rewards.append(float(reward))
            if np.any(done):
                break
        rewards.append(float(np.mean(episode_rewards)) if episode_rewards else 0.0)

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    print(f"{mean_reward:.4f}")
    return mean_reward
