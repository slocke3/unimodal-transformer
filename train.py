"""
Train a DiscreteTrajectoryTransformer on tokenized quadratic map trajectories.

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --run_name my_run
"""
import argparse
import os
import json

import yaml
import torch

from src.dataset import make_splits
from src.model import DiscreteTrajectoryTransformer
from src.trainer import Trainer, TrainerConfig


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg, n_bins, context_len):
    m = cfg["model"]
    return DiscreteTrajectoryTransformer(
        n_bins=n_bins,
        context_len=context_len,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        d_ff=m.get("d_ff"),
        dropout=m["dropout"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run_name", default=None, help="Override run_name in config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or cfg.get("run_name", "run")

    d = cfg["data"]
    t = cfg["training"]
    p = cfg["paths"]

    os.makedirs(p["checkpoint_dir"], exist_ok=True)
    os.makedirs(p["figure_dir"], exist_ok=True)
    os.makedirs(p["cache_dir"], exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Config: {args.config}")

    train_loader, val_loader, test_loader = make_splits(
        n_trajectories=d["n_trajectories"],
        r_range=tuple(d["r_range"]),
        context_len=d["context_len"],
        n_bins=d["n_bins"],
        traj_len=d["traj_len"],
        burn_in=d["burn_in"],
        train_frac=d["train_frac"],
        val_frac=d["val_frac"],
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        seed=d["seed"],
    )

    model = build_model(cfg, n_bins=d["n_bins"], context_len=d["context_len"])

    trainer_cfg = TrainerConfig(
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        max_epochs=t["max_epochs"],
        patience=t["patience"],
        grad_clip=t["grad_clip"],
        log_every=t["log_every"],
        eval_every=t["eval_every"],
        save_dir=p["checkpoint_dir"],
    )

    trainer = Trainer(model, train_loader, val_loader, config=trainer_cfg, run_name=run_name)
    history = trainer.train()

    history_path = os.path.join(p["checkpoint_dir"], f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    config_path = os.path.join(p["checkpoint_dir"], f"{run_name}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()