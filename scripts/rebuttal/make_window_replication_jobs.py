"""Prespecified seed replication and optional denser-window follow-up.

Replication: all eight original intervals with positive chaotic coverage and
eight regular controls matched by width, two new seeds each (32 CE runs).
Expansion: half-step offsets at widths .125, .25, .5 across [3,4], with three
seeds each (33 CE runs). Selection uses interval geometry, never model loss.
The stages are separate so the additional budget is an explicit decision.
"""
import argparse
import csv
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]


def original_jobs():
    jobs = {}
    for line in (ROOT / "jobs_win_fixed.txt").read_text().splitlines():
        tokens = shlex.split(line)
        flags = dict(zip(tokens[::2], tokens[1::2]))
        if flags.get("--output_mode", "bins") != "bins":
            continue
        key = (float(flags["--width"]), float(flags["--start"]))
        jobs[key] = flags
    return jobs


def selected_windows():
    # Positive-coverage windows from the original dense Lyapunov calculation.
    # Regular controls use the nearest preceding regular intervals per width.
    return [
        (1., 3., "chaotic"), (1., 2.1667, "regular"),
        (.5, 3.5, "chaotic"), (.5, 3., "regular"),
        (.25, 3.5, "chaotic"), (.25, 3.75, "chaotic"),
        (.25, 3., "regular"), (.25, 3.25, "regular"),
        *[(.125, s, "chaotic") for s in (3.5, 3.625, 3.75, 3.875)],
        *[(.125, s, "regular") for s in (3., 3.125, 3.25, 3.375)],
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["replicate", "expand"], default="replicate")
    ap.add_argument("--device-data", action="store_true",
                    help="Use GPU loader; changes minibatch RNG relative to original DataLoader.")
    args = ap.parse_args()
    originals = original_jobs()
    if args.stage == "replicate":
        windows = selected_windows(); seeds = (1, 2)
    else:
        windows = []
        for width, count in [(.125, 7), (.25, 3), (.5, 1)]:
            windows.extend((width, 3 + width/2 + j*width, "offset") for j in range(count))
        seeds = (0, 1, 2)
    lines, manifest = [], []
    root = f"runs_win_rebuttal_{args.stage}_ce"
    for width, start, group in windows:
        # Width alone selects the template in the expansion stage; all other
        # settings are the original fixed-step protocol.
        template = originals.get((width, start))
        if template is None:
            template = next(v for (w, _), v in originals.items() if w == width)
        template = dict(template)
        original = template["--out_dir"] if args.stage == "replicate" else ""
        for seed in seeds:
            name = f"win_w{width:g}_s{start:.4f}_seed{seed}"
            # Keep original three-decimal names for identical old intervals.
            if original:
                name = Path(original).name.rsplit("_seed", 1)[0] + f"_seed{seed}"
            flags = dict(template, **{"--start": f"{start:.4f}", "--seed": str(seed),
                                      "--out_dir": f"{root}/{name}"})
            # Explicitly pin defaults that matter scientifically.
            flags.update({"--traj_len": "150", "--burn_in": "0", "--dropout": "0.1",
                          "--n_bins_in": "64", "--n_bins_out": "64", "--output_mode": "bins",
                          "--batch_size": "256", "--d_model": "128", "--n_heads": "4",
                          "--n_layers": "4", "--lr": "0.0003", "--weight_decay": "0.0001",
                          "--r_eval_points": "300", "--n_eval_per_r": "30"})
            command = shlex.join([x for k, v in flags.items() for x in (k, v)])
            if args.device_data:
                command += " --device_data --num_workers 0"
            lines.append(command)
            manifest.append(dict(stage=args.stage, width=width, start=start, end=start+width,
                                 group=group, seed=seed, m=int(flags["--m"]),
                                 original_seed0=original, out_dir=flags["--out_dir"],
                                 loader="device" if args.device_data else "DataLoader"))
    out = ROOT / f"jobs_rb_window_{args.stage}.txt"
    out.write_text("\n".join(lines)+"\n")
    manifest_path = ROOT / "figures_rebuttal/window_transfer" / f"{args.stage}_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    print(f"Wrote {len(lines)} jobs ({len(windows)} intervals, seeds {seeds}) to {out.name}")
    print(f"Total optimization steps: {len(lines)*160000:,}. Original DataLoader: {not args.device_data}.")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
