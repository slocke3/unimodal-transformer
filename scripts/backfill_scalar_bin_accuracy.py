"""Bin scalar-model predictions and backfill 64-bin evaluation metrics.

The window-sweep ``eval_per_r.npz`` files store only aggregate RMS, so binned
accuracy cannot be reconstructed without the checkpoints. Run this script on
the machine that still holds ``best_best.pt`` or ``best_final.pt`` in each run
directory. It reuses the saved evaluation grid and original evaluation seed,
then appends the following arrays without retraining:

  * scalar_bin_acc_per_r: fraction of scalar predictions in the target's bin;
  * scalar_bin_mae_per_r: mean absolute predicted/target bin-index difference.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_device(requested, torch):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_checkpoint(run_dir, requested):
    if requested != "auto":
        path = run_dir / requested
        return path if path.exists() else None
    for name in ("best_best.pt", "best_final.pt"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def evaluate(model, r_grid, params, device, n_output_bins, batch_size):
    import torch
    from src.maps import iterate_map, tokenize_trajectory

    model.eval()
    rng = np.random.default_rng(int(params["seed"]) + 7)
    context_len = int(params["context_len"])
    traj_len = int(params["traj_len"])
    burn_in = int(params.get("burn_in", 0))
    n_eval = int(params.get("n_eval_per_r", 30))
    n_input_bins = int(
        params["n_bins"] if params.get("n_bins_in") is None
        else params["n_bins_in"]
    )
    synonyms = int(params.get("synonyms", 1))
    window_size = context_len + 1
    accuracy = np.empty(len(r_grid), dtype=float)
    bin_mae = np.empty(len(r_grid), dtype=float)

    with torch.inference_mode():
        for i, r in enumerate(r_grid):
            raw_contexts, raw_targets = [], []
            for _ in range(n_eval):
                x0 = rng.uniform(0.05, 0.95)
                trajectory = iterate_map(
                    x0, float(r), burn_in + traj_len
                )[burn_in:]
                for start in range(len(trajectory) - window_size):
                    raw_contexts.append(
                        trajectory[start:start + context_len]
                    )
                    raw_targets.append(trajectory[start + context_len])

            context_tokens = tokenize_trajectory(
                np.asarray(raw_contexts), n_input_bins
            )
            if synonyms > 1:
                context_tokens = (
                    context_tokens * synonyms
                    + rng.integers(0, synonyms, size=context_tokens.shape)
                )
            target_tokens = tokenize_trajectory(
                np.asarray(raw_targets), n_output_bins
            )

            correct = 0
            absolute_error = 0
            count = len(target_tokens)
            for begin in range(0, count, batch_size):
                end = min(begin + batch_size, count)
                context = torch.as_tensor(
                    context_tokens[begin:end], dtype=torch.long, device=device
                )
                prediction = model(context)
                predicted_tokens = torch.floor(
                    prediction * n_output_bins
                ).to(torch.long).clamp_(0, n_output_bins - 1)
                truth = torch.as_tensor(
                    target_tokens[begin:end], dtype=torch.long, device=device
                )
                correct += int((predicted_tokens == truth).sum().item())
                absolute_error += int(
                    (predicted_tokens - truth).abs().sum().item()
                )

            accuracy[i] = correct / count
            bin_mae[i] = absolute_error / count
            if (i + 1) % 50 == 0:
                print(f"    eval {i + 1}/{len(r_grid)}", flush=True)
    return accuracy, bin_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_win_fix_mse")
    parser.add_argument("--output-bins", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint", default="auto",
        help="checkpoint filename or 'auto' for best_best.pt/best_final.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dirs = sorted(Path(args.runs).glob("win_w*_s*_seed*"))
    if args.limit is not None:
        run_dirs = run_dirs[:args.limit]
    completed = 0
    missing = []
    torch = None
    DiscreteTrajectoryTransformer = None
    device = None
    for run_dir in run_dirs:
        result_path = run_dir / "eval_per_r.npz"
        params_path = run_dir / "params.json"
        if not result_path.exists() or not params_path.exists():
            continue
        with np.load(result_path) as loaded:
            saved = {key: loaded[key] for key in loaded.files}
        if "scalar_bin_acc_per_r" in saved and not args.overwrite:
            print(f"[skip] {run_dir.name}: already evaluated")
            continue
        checkpoint = find_checkpoint(run_dir, args.checkpoint)
        if checkpoint is None:
            missing.append(run_dir.name)
            print(f"[missing] {run_dir.name}: no checkpoint", flush=True)
            continue
        with params_path.open() as handle:
            params = json.load(handle)
        if params.get("output_mode") != "scalar":
            raise ValueError(f"{run_dir.name} is not a scalar-output run")
        if params.get("input_mode", "bins") != "bins":
            raise ValueError(f"{run_dir.name} does not use tokenized inputs")

        if torch is None:
            import torch as torch_module
            from src.model import DiscreteTrajectoryTransformer as Model
            torch = torch_module
            DiscreteTrajectoryTransformer = Model
            device = resolve_device(args.device, torch)

        n_in = int(
            params["n_bins"] if params.get("n_bins_in") is None
            else params["n_bins_in"]
        ) * int(params.get("synonyms", 1))
        model = DiscreteTrajectoryTransformer(
            n_bins=int(params["n_bins"]),
            context_len=int(params["context_len"]),
            d_model=int(params["d_model"]),
            n_heads=int(params["n_heads"]),
            n_layers=int(params["n_layers"]),
            dropout=float(params["dropout"]),
            n_bins_in=n_in,
            output_mode="scalar",
        ).to(device)
        state = torch.load(
            checkpoint, map_location=device, weights_only=True
        )["model_state_dict"]
        model.load_state_dict(state)
        print(f"[eval] {run_dir.name} on {device}", flush=True)
        accuracy, bin_mae = evaluate(
            model, saved["r_grid"], params, device,
            args.output_bins, args.batch_size,
        )
        saved["scalar_bin_acc_per_r"] = accuracy
        saved["scalar_bin_mae_per_r"] = bin_mae
        saved["scalar_bin_output_bins"] = np.asarray(args.output_bins)
        temporary = result_path.with_name("eval_per_r.backfill_tmp.npz")
        np.savez(temporary, **saved)
        temporary.replace(result_path)
        completed += 1
        print(
            f"[saved] {run_dir.name}: mean bin accuracy={accuracy.mean():.4f}",
            flush=True,
        )

    print(f"completed {completed} runs; missing checkpoints for {len(missing)}")
    if missing:
        print("Checkpoints are required for: " + ", ".join(missing))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
