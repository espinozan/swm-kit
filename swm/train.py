from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch


def _load_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        obs_key = "obs" if "obs" in f else "observations"
        act_key = "actions" if "actions" in f else "action"
        obs = np.asarray(f[obs_key])
        actions = np.asarray(f[act_key])
    return obs, actions


def _build_mlp(input_dim: int, output_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, output_dim),
    )


def train(dataset_path: str, model_out: str, epochs: int = 5, batch_size: int = 256) -> Path:
    data_path = Path(dataset_path)
    model_path = Path(model_out)
    obs, actions = _load_arrays(data_path)

    obs = obs.astype(np.float32)
    actions = actions.astype(np.float32)
    x_obs = obs[:-1]
    y_obs = obs[1:]
    x = np.concatenate([x_obs.reshape(len(x_obs), -1), actions[:-1].reshape(len(x_obs), -1)], axis=1)
    y = y_obs.reshape(len(y_obs), -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_mlp(x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    payload = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "output_dim": y.shape[1],
    }
    torch.save(payload, model_path)
    return model_path
