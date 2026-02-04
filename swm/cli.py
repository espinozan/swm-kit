from pathlib import Path

import typer

from .eval import evaluate
from .record import record_dataset
from .train import train

app = typer.Typer(add_completion=False, help="swm-kit: ultraminimal stable-worldmodel utilities")


@app.command()
def record(env: str, out: str, episodes: int = 100, seed: int = 0) -> None:
    """Record a dataset using stable-worldmodel."""
    record_dataset(env=env, out=out, episodes=episodes, seed=seed)
    typer.echo(str(Path(out)))


@app.command()
def train_model(data: str = typer.Option(..., "--data"), out: str = typer.Option(..., "--out")) -> None:
    """Train a tiny world model from an HDF5 dataset."""
    train(dataset_path=data, model_out=out)
    typer.echo(str(Path(out)))


@app.command()
def eval_model(env: str, model: str = typer.Option(..., "--model"), episodes: int = 10) -> None:
    """Evaluate a planner using a trained model."""
    evaluate(env=env, model_path=model, episodes=episodes)


if __name__ == "__main__":
    app()
