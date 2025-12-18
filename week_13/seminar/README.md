# nanotabdl — Minimal Tabular Deep Learning

Educational implementation for tabular DL seminar.

## Overview

> [!WARNING]  
> This is a heavily vieb-coded minimized and untested version of something you can see in our real research code. Do not use it for anything important!

This project is heavily inspired by our internal research template (yr-template).
It is set up in a way that enables fast and reproducible research for tabular data experiments.

The implementation uses tinygrad (experimental) and PyTorch, designed for M1 Mac.

- the code contains a lightweight experiment management utilities (including reproducible config+script scaffold and a hyperparameter tuning function)
- PyTorch and tinygrad implementations of the MLP model with a vectorized numerical embedings implementation (PLE from the embedding paper)
- For more macos tabular deep learning see also https://github.com/vaaven/TabM-MLX which is much cleaner than this and has fairer comparisons.

**Do not use this in production.** This is for educational purposes only.

## Setup

```bash
# Install dependencies
uv sync

# For LightGBM on macOS, you also need:
brew install libomp
```

## Project Structure

```
nanotabdl/
├── lib.py                    # Core library (~800 lines)
│   ├── Data loading          # Dataset, Task, preprocessing
│   ├── Metrics               # compute_metrics
│   ├── Experiment            # create_exp, run, load_config, dump_report
│   ├── Tuning                # tune() with Optuna, parallel support
│   ├── Evaluation            # evaluate() multi-seed, aggregation
│   └── Tinygrad utils        # quantile bins, etc. (probably not needed)
│
├── bin/                      # Training scripts
│   ├── go.py                 # CLI launcher for tune/evaluate
│   ├── lightgbm_.py          # LightGBM training
│   ├── mlp.py                # Tinygrad MLP (experimental)
│   └── mlp_torch.py          # PyTorch MLP
│
└── exp/                      # Experiments
    └── lightgbm/
        ├── california/tuning/config.toml
        └── adult/tuning/config.toml
```

## Usage

### Single Experiment

```bash
# Run a single experiment
uv run python -m bin.go run exp/test_lightgbm
```

### Hyperparameter Tuning

```bash
# Generate tuning configs
uv run python exp/lightgbm/configs.py

# Run tuning (sequential)
uv run python -m bin.go tune exp/lightgbm/california/tuning

# Run tuning (parallel with 4 workers)
uv run python -m bin.go tune exp/lightgbm/california/tuning -w 4 -n 100
```

### Multi-seed Evaluation

```bash
# Create evaluation config (use best config from tuning)
# Then run:
uv run python -m bin.go evaluate exp/lightgbm/california/evaluation -s 5 -w 4
```

### Python API

```python
import lib

# Load dataset
ds = lib.Dataset.load("california")
ds = lib.preprocess(ds, num_method="standard")
print(ds)  # Dataset(california, regression, d=8, train=14448, val=2064, test=4128)

# Run tuning
lib.tune("exp/lightgbm/california/tuning", n_trials=100, n_workers=4)

# Run evaluation
lib.evaluate("exp/lightgbm/california/evaluation", n_seeds=5, n_workers=4)
```

## Config Format

### Tuning Config

```toml
function = "bin.lightgbm_.main"
n_trials = 100

[space]
seed = 0
dataset = "california"

[space.model]
learning_rate = ["_tune_", "loguniform", 0.005, 0.1]
num_leaves = ["_tune_", "logint", 2, 200]
# ... more hyperparameters

[space.train]
n_estimators = 4000
early_stopping_rounds = 50
```

Supported distributions:
- `["_tune_", "uniform", low, high]` — uniform float
- `["_tune_", "loguniform", low, high]` — log-uniform float
- `["_tune_", "int", low, high]` — uniform int
- `["_tune_", "logint", low, high]` — log-uniform int
- `["_tune_", "categorical", val1, val2, ...]` — categorical choice

### Evaluation Config

```toml
function = "bin.lightgbm_.main"
n_seeds = 5

[base_config]
seed = 0  # will be replaced per-seed
dataset = "california"
# ... rest of config from best tuning run
```

## Datasets

Located in `~/new-data/`. Format:
```
~/new-data/{name}/
├── info.json          # {"task": {"type": "regression", "score": "rmse"}}
├── x_num.npy          # (n_samples, n_num_features), float32
├── x_cat.npy          # (n_samples, n_cat_features), optional
├── y.npy              # (n_samples,)
└── splits/default/
    ├── train.npy      # indices
    ├── val.npy
    └── test.npy
```

Available: california, adult, covtype2, churn, diamond, microsoft, otto, higgs-small, house, black-friday

---

## What's Done

### Experiment Infrastructure (lib.py)

- [x] Dataset loading and preprocessing (standard, quantile, noisy_quantile)
- [x] Metrics computation with score normalization (higher = better)
- [x] Experiment management (create_exp, run, load_config, dump_report)
- [x] `tune()` — Optuna-based hyperparameter tuning with optional parallelization
- [x] `evaluate()` — Multi-seed evaluation with aggregation (mean ± std)
- [x] `aggregate_metrics()` — Combine results from multiple runs
- [x] `import_function()` — Dynamic function import for parallel workers
- [x] CLI launcher (`bin/go.py`) for tune/evaluate commands

### Models

- [x] `bin/lightgbm_.py` — LightGBM training script with proper metrics
- [x] `bin/mlp.py` — Tinygrad MLP with piecewise linear embeddings (experimental)
- [x] `bin/mlp_torch.py` — PyTorch MLP with piecewise linear embeddings

### Configs

- [x] `exp/lightgbm/configs.py` — Generate tuning configs for california/adult
- [x] LightGBM search space from README specification

---

## What's Left (TODO)

### High Priority

1. **Test the full pipeline** — Run actual tuning + evaluation on california/adult
2. **Fix mlp.py / mlp_torch.py** — Currently returns only loss, needs proper metrics via Task.calculate_metrics()
3. **Add early stopping** to MLP scripts

### Medium Priority

4. **TabM implementation** — MLP + numerical embeddings (PLR or Periodic)
   - Reference: `rtdl_num_embeddings` package
   - Should support both tinygrad and PyTorch

5. **Benchmark script** — Pareto frontier plot (accuracy vs time)
   - X-axis: training time (or inference time)
   - Y-axis: test metric
   - Compare: LightGBM vs MLP vs TabM

6. **Rich progress reporting** — Better CLI output during tuning/evaluation

### Low Priority

7. **Checkpoint/resume** for tune() — Currently restarts if interrupted mid-trial
8. **Tests** — Unit tests for lib.py functions
9. **Categorical embeddings** — nn.Embedding per categorical feature for MLP/TabM

---

## Reference

- **yr-template**: `./yr-template` (symlink to ~/tyl/main) — Ground truth implementation
- **rtdl_num_embeddings**: Numerical embeddings reference implementations
- **TabM paper**: Batch ensemble trick for tabular data

## Code Style

- Minimal, nanogpt-inspired (illustrative, not production)
- No logging framework, just print
- Single-file modules preferred
- Comments explain "why", code explains "what"

## Environment

- Python 3.12
- Always use `uv run python ...`
- Always use `uv run ruff check` after edits
