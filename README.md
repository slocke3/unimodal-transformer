# unimodal-transformer

Training transformers on *symbolic sequences* from unimodal maps (the quadratic
family $f_r(x) = rx(1-x)$ and relatives), to study what they learn about the
underlying dynamics.

The trajectory is discretized by partitioning $[0,1]$ into $N$ bins, turning a
continuous orbit into a token sequence. A causal transformer is trained for
next-token prediction, and we ask how its cross-entropy relates to dynamical
invariants (Lyapunov exponent, symbolic / kneading structure) and whether it
generalizes across map families.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train a model from a config:

```bash
python train.py --config configs/base.yaml
```

Evaluate a trained checkpoint (per-$r$ cross-entropy vs Lyapunov, rollouts,
cross-family generalization):

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/base_best.pt \
    --config outputs/checkpoints/base_config.yaml
```

## Layout

```
src/
  maps.py        # map definitions, tokenization, Lyapunov exponents, family registry
  model.py       # DiscreteTrajectoryTransformer + MLP baseline
  dataset.py     # trajectory dataset + train/val/test splits
  trainer.py     # training loop, checkpointing, early stopping
  evaluation.py  # per-r / per-family evaluation + plotting
configs/         # YAML experiment configs (base.yaml + ablations/)
notebooks/
  explore.ipynb       # scratch space
  analysis.ipynb      # canonical: load saved results, make figures
  original_colab.ipynb # the original Colab notebook this was refactored from
train.py         # CLI: train a model
evaluate.py      # CLI: evaluate a checkpoint
outputs/         # checkpoints, figures, caches (gitignored)
```

## Workflow

- Each `train.py` run saves its checkpoint, config, and history under
  `outputs/checkpoints/<run_name>_*`, so a result is always traceable to the
  config that produced it.
- For ablation sweeps (bin size $N$, context length $L$, model capacity,
  positional embeddings), add configs under `configs/ablations/` and run them
  with `--run_name`.
- `notebooks/explore.ipynb` is throwaway; `notebooks/analysis.ipynb` is
  canonical and should stay clean and reproducible.

## Notes

- `outputs/` is gitignored (reproducible from scratch). The Lyapunov grid is
  cached to `outputs/cache/` since it is expensive to recompute.
