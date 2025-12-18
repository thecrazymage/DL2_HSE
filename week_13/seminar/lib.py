"""
Minimal tabular data utilities.

Uses the same data format as yr-template:
- Datasets live in DATA_DIR (~/new-data by default)
- Each dataset has: info.json, x_num.npy, [x_cat.npy], y.npy, splits/
"""

import datetime
import json
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import scipy.special
import sklearn.metrics
import sklearn.preprocessing

# =============================================================================
# Config
# =============================================================================
DATA_DIR = Path.home() / "new-data"


# =============================================================================
# Types
# =============================================================================
class TaskType(Enum):
    REGRESSION = "regression"
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"


class PredictionType(Enum):
    LABELS = "labels"
    PROBS = "probs"
    LOGITS = "logits"


Part = str  # "train", "val", "test"


# =============================================================================
# Data Loading
# =============================================================================
def load_split(
    dataset_dir: Path, split_id: tuple[str, ...] = ("default",)
) -> dict[Part, np.ndarray]:
    """Load split indices from nested directory structure."""
    split = {}
    directory = dataset_dir / "splits"
    for dirname in split_id:
        directory = directory / dirname
        assert directory.is_dir(), f"Split directory not found: {directory}"
        for path in directory.iterdir():
            if path.suffix == ".npy" and not path.is_dir():
                split[path.stem] = np.load(path)
    return split


def apply_split(
    data: np.ndarray, split: dict[Part, np.ndarray]
) -> dict[Part, np.ndarray]:
    return {k: data[v] for k, v in split.items()}


# =============================================================================
# Preprocessing
# =============================================================================
def transform_num(
    X: dict[Part, np.ndarray],
    method: str = "standard",
    seed: int = 0,
) -> dict[Part, np.ndarray]:
    """
    Transform numerical features.

    Methods:
    - "standard": z-score normalization
    - "quantile": quantile transform to normal
    - "noisy_quantile": quantile with noise to break ties
    """
    if method == "standard":
        scaler = sklearn.preprocessing.StandardScaler()
        X_fit = X["train"]
    elif method in ("quantile", "noisy_quantile"):
        scaler = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            output_distribution="normal",
            subsample=1_000_000_000,
            random_state=seed,
        )
        if method == "noisy_quantile":
            X_fit = X["train"] + np.random.RandomState(seed).normal(
                0.0, 1e-5, X["train"].shape
            ).astype(X["train"].dtype)
        else:
            X_fit = X["train"]
    else:
        raise ValueError(f"Unknown method: {method}")

    scaler.fit(X_fit)
    result = {k: scaler.transform(v).astype(np.float32) for k, v in X.items()}  # type: ignore
    result = {k: np.nan_to_num(v) for k, v in result.items()}
    return result


def transform_cat(
    X: dict[Part, np.ndarray],
    method: str = "ordinal",
) -> dict[Part, np.ndarray]:
    """Transform categorical features to ordinal integers."""
    if method != "ordinal":
        raise ValueError(f"Unknown method: {method}")

    unknown_value = np.iinfo("int64").max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        dtype="int64",  # type: ignore
    ).fit(X["train"])

    result = {k: encoder.transform(v) for k, v in X.items()}
    max_values = result["train"].max(axis=0)
    for part in ["val", "test"]:
        if part in result:
            for col in range(result[part].shape[1]):
                mask = result[part][:, col] == unknown_value
                result[part][mask, col] = max_values[col] + 1
    return result


def standardize_y(
    y: dict[Part, np.ndarray],
) -> tuple[dict[Part, np.ndarray], float, float]:
    """Standardize regression labels. Returns (y_dict, mean, std)."""
    mean, std = float(y["train"].mean()), float(y["train"].std())
    return {k: ((v - mean) / std).astype(np.float32) for k, v in y.items()}, mean, std


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType = PredictionType.LABELS,
) -> dict[str, float]:
    """Compute task-appropriate metrics."""
    if task_type == TaskType.REGRESSION:
        return {
            "rmse": float(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
            "r2": float(sklearn.metrics.r2_score(y_true, y_pred)),
        }

    # Classification
    if prediction_type == PredictionType.LOGITS:
        if task_type == TaskType.BINCLASS:
            probs = scipy.special.expit(y_pred)
            labels = (probs > 0.5).astype(np.int64)
        else:
            probs = scipy.special.softmax(y_pred, axis=1)
            labels = probs.argmax(axis=1)
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
        labels = (
            (probs > 0.5).astype(np.int64)
            if task_type == TaskType.BINCLASS
            else probs.argmax(axis=1)
        )
    else:
        labels, probs = y_pred, None

    result = {"accuracy": float(sklearn.metrics.accuracy_score(y_true, labels))}
    if probs is not None:
        n_classes = 2 if task_type == TaskType.BINCLASS else probs.shape[-1]
        result["cross_entropy"] = float(
            sklearn.metrics.log_loss(y_true, probs, labels=np.arange(n_classes))
        )
        if task_type == TaskType.BINCLASS:
            result["roc_auc"] = float(sklearn.metrics.roc_auc_score(y_true, probs))
    return result


# =============================================================================
# Task
# =============================================================================

# Score sign: positive means higher is better, negative means lower is better
SCORE_SIGN: dict[str, int] = {
    "rmse": -1,
    "mae": -1,
    "cross_entropy": -1,
    "accuracy": 1,
    "r2": 1,
    "roc_auc": 1,
}


@dataclass
class Task:
    """Holds original labels and computes metrics."""

    y: dict[Part, np.ndarray]
    type_: TaskType
    score_name: str  # "rmse", "accuracy", etc.

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    def compute_n_classes(self) -> int:
        return len(np.unique(np.concatenate(list(self.y.values()))))

    def calculate_metrics(
        self,
        predictions: dict[Part, Any],
        prediction_type: PredictionType = PredictionType.LABELS,
    ) -> dict[Part, dict[str, float]]:
        result = {}
        for part in predictions:
            metrics = compute_metrics(
                self.y[part], predictions[part], self.type_, prediction_type
            )
            # Add normalized score (higher is always better)
            sign = SCORE_SIGN.get(self.score_name, 1)
            metrics["score"] = metrics[self.score_name] * sign
            result[part] = metrics
        return result


# =============================================================================
# Dataset
# =============================================================================
@dataclass
class Dataset:
    """
    Dataset = Data + Task.

    data: {'x_num': {'train': ..., 'val': ..., 'test': ...}, 'y': {...}, ...}
    task: Task (holds original labels for evaluation)
    """

    data: dict[str, dict[Part, np.ndarray]]
    task: Task
    name: str

    @classmethod
    def load(cls, name: str, split_id: tuple[str, ...] = ("default",)) -> "Dataset":
        dataset_dir = DATA_DIR / name
        assert dataset_dir.exists(), f"Dataset not found: {dataset_dir}"

        info = json.loads((dataset_dir / "info.json").read_text())
        split = load_split(dataset_dir, split_id)

        data = {}
        for key in ["x_num", "x_cat", "y"]:
            path = dataset_dir / f"{key}.npy"
            if path.exists():
                data[key] = apply_split(np.load(path), split)

        task = Task(
            y={k: v.copy() for k, v in data["y"].items()},  # copy for safety
            type_=TaskType(info["task"]["type"]),
            score_name=info["task"]["score"],
        )

        return cls(data=data, task=task, name=name)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def n_num_features(self) -> int:
        return self.data["x_num"]["train"].shape[1] if "x_num" in self.data else 0

    @property
    def n_cat_features(self) -> int:
        return self.data["x_cat"]["train"].shape[1] if "x_cat" in self.data else 0

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Part) -> int:
        return len(self.data["y"][part])

    def parts(self) -> list[Part]:
        return list(self.data["y"].keys())

    def cat_cardinalities(self) -> list[int]:
        if "x_cat" not in self.data:
            return []
        return [len(np.unique(col)) for col in self.data["x_cat"]["train"].T]

    def __repr__(self) -> str:
        parts = ", ".join(f"{p}={self.size(p)}" for p in self.parts())
        return f"Dataset({self.name}, {self.task.type_.value}, d={self.n_features}, {parts})"


# =============================================================================
# Preprocessing (returns new Dataset)
# =============================================================================
def preprocess(
    dataset: Dataset,
    num_method: str = "standard",
    cat_method: str = "ordinal",
    seed: int = 0,
) -> Dataset:
    """Apply preprocessing, returning a new Dataset (task unchanged)."""
    data = dict(dataset.data)

    if "x_num" in data:
        data["x_num"] = transform_num(data["x_num"], num_method, seed)
    if "x_cat" in data:
        data["x_cat"] = transform_cat(data["x_cat"], cat_method)

    return Dataset(data=data, task=dataset.task, name=dataset.name)


# =============================================================================
# Experiment
# =============================================================================
Report = dict[str, Any]


def load_config(exp: str | Path) -> dict[str, Any]:
    """Load config.toml from experiment directory."""
    with open(Path(exp) / "config.toml", "rb") as f:
        return tomllib.load(f)


def load_report(exp: str | Path) -> Report:
    """Load report.json from experiment directory."""
    with open(Path(exp) / "report.json") as f:
        return json.load(f)


def dump_report(exp: str | Path, report: Report) -> None:
    """Save report.json to experiment directory."""
    with open(Path(exp) / "report.json", "w") as f:
        json.dump(report, f, indent=4)


def create_exp(exp: str | Path, config: dict[str, Any], *, force: bool = False) -> Path:
    """Create experiment directory with config.toml."""
    import tomli_w

    exp = Path(exp)
    if exp.exists():
        if force:
            shutil.rmtree(exp)
        else:
            raise RuntimeError(f"Experiment already exists: {exp}")

    exp.mkdir(parents=True)
    with open(exp / "config.toml", "wb") as f:
        tomli_w.dump(config, f)
    return exp


def run(
    main_fn: Callable[[dict[str, Any], Path], Report],
    exp: str | Path,
    *,
    force: bool = False,
) -> Report | None:
    """
    Run an experiment.

    Args:
        main_fn: Function(config, exp_path) -> report
        exp: Experiment directory (must contain config.toml)
        force: If True, overwrite existing report

    Returns:
        Report dict, or None if skipped
    """
    exp = Path(exp)
    report_path = exp / "report.json"

    print("=" * 60)
    print(f"{exp} | {datetime.datetime.now()}")
    print("=" * 60)

    if report_path.exists() and not force:
        print("Already done, skipping.")
        return None

    config = load_config(exp)
    print("\nConfig:")
    pprint(config, sort_dicts=False)
    print()

    report = main_fn(config, exp)
    dump_report(exp, report)

    # Print summary
    print()
    print("-" * 40)
    if "time" in report:
        print(f"Time: {datetime.timedelta(seconds=int(report['time']))}")
    if "metrics" in report:
        for part, m in report["metrics"].items():
            score = m.get("score", next(iter(m.values())))
            print(f"  {part}: {score:.4f}")
    print("-" * 40)

    return report


# =============================================================================
# Dynamic Import
# =============================================================================
def import_function(path: str) -> Callable:
    """
    Import a function by its dotted path.

    Example: import_function("bin.lightgbm.main") -> function object
    """
    import importlib

    module_path, fn_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


# =============================================================================
# Config Sampling (for Optuna-based tuning)
# =============================================================================
def sample_config(config: dict, trial) -> dict:
    """
    Recursively sample tunable parameters from config using Optuna trial.

    Tunable parameters are marked with ["_tune_", type, ...args]:
    - ["_tune_", "uniform", low, high] -> trial.suggest_float
    - ["_tune_", "loguniform", low, high] -> trial.suggest_float(log=True)
    - ["_tune_", "int", low, high] -> trial.suggest_int
    - ["_tune_", "logint", low, high] -> trial.suggest_int(log=True)
    - ["_tune_", "categorical", val1, val2, ...] -> trial.suggest_categorical
    """
    return _sample_config_recursive(config, trial, prefix="")


def _sample_config_recursive(config: dict, trial, prefix: str) -> dict:
    result = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, list) and len(value) >= 2 and value[0] == "_tune_":
            result[key] = _sample_param(trial, full_key, value[1:])
        elif isinstance(value, dict):
            result[key] = _sample_config_recursive(value, trial, full_key)
        else:
            result[key] = value
    return result


def _sample_param(trial, name: str, spec: list):
    """Sample a single parameter based on spec."""
    dist_type, *args = spec

    if dist_type == "uniform":
        return trial.suggest_float(name, args[0], args[1])
    elif dist_type == "loguniform":
        return trial.suggest_float(name, args[0], args[1], log=True)
    elif dist_type == "int":
        return trial.suggest_int(name, args[0], args[1])
    elif dist_type == "logint":
        return trial.suggest_int(name, args[0], args[1], log=True)
    elif dist_type == "categorical":
        return trial.suggest_categorical(name, args)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


# =============================================================================
# Tuning
# =============================================================================
def _tune_worker(exp_path: str, n_trials: int, timeout: float | None) -> None:
    """
    Worker function for parallel tuning. Must be at module level for pickling.
    Each worker loads the study and config independently.
    """
    import optuna
    import tempfile

    exp = Path(exp_path)
    config = load_config(exp)
    main_fn = import_function(config["function"])
    space = config["space"]
    storage_path = exp / "study.db"

    def objective(trial):
        sampled = sample_config(space, trial)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_exp = Path(tmp) / "exp"
            create_exp(tmp_exp, sampled)
            report = main_fn(sampled, tmp_exp)
        return report["metrics"]["val"]["score"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name="tune",
        storage=f"sqlite:///{storage_path}",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)


def tune(
    exp: str | Path,
    n_trials: int = 100,
    n_workers: int = 1,
    timeout: float | None = None,
    force: bool = False,
) -> dict:
    """
    Hyperparameter tuning with Optuna.

    Config format (config.toml):
        function = "bin.lightgbm.main"
        n_trials = 100  # optional, can be overridden by argument

        [space]
        seed = 0
        dataset = "california"

        [space.model]
        learning_rate = ["_tune_", "loguniform", 0.001, 0.1]
        num_leaves = ["_tune_", "logint", 8, 128]

    Args:
        exp: Experiment directory containing config.toml
        n_trials: Number of trials to run
        n_workers: Number of parallel workers (1 = sequential)
        timeout: Optional timeout in seconds
        force: If True, delete existing study and start fresh

    Returns:
        Report dict with best config and score
    """
    import optuna
    import tempfile
    import time

    exp = Path(exp)
    config = load_config(exp)

    main_fn = import_function(config["function"])
    space = config["space"]
    n_trials = config.get("n_trials", n_trials)

    # Setup storage for parallel execution
    storage_path = exp / "study.db"
    if force and storage_path.exists():
        storage_path.unlink()

    storage = f"sqlite:///{storage_path}" if n_workers > 1 else None
    study_name = "tune" if n_workers > 1 else None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    start_time = time.time()

    def objective(trial):
        sampled = sample_config(space, trial)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_exp = Path(tmp) / "exp"
            create_exp(tmp_exp, sampled)
            report = main_fn(sampled, tmp_exp)
        return report["metrics"]["val"]["score"]

    # Progress callback for reporting
    def progress_callback(study, trial):
        n_complete = len([t for t in study.trials if t.state.is_finished()])
        best = study.best_trial
        print(
            f"Trial {trial.number}: {trial.value:.4f} | "
            f"Best: {best.value:.4f} (#{best.number}) | "
            f"Progress: {n_complete}/{n_trials}"
        )

    if n_workers == 1:
        # Sequential execution
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[progress_callback],
        )
    else:
        # Parallel execution using module-level worker
        from concurrent.futures import ProcessPoolExecutor, wait
        import threading

        trials_per_worker = n_trials // n_workers
        extra_trials = n_trials % n_workers

        # Progress monitor thread
        stop_monitor = threading.Event()

        def monitor_progress():
            last_count = 0
            while not stop_monitor.wait(timeout=5.0):
                try:
                    monitor_study = optuna.load_study(
                        study_name=study_name, storage=storage
                    )
                    n_complete = len(
                        [t for t in monitor_study.trials if t.state.is_finished()]
                    )
                    if n_complete > last_count:
                        best = monitor_study.best_trial
                        print(
                            f"Progress: {n_complete}/{n_trials} trials | "
                            f"Best: {best.value:.4f} (#{best.number})"
                        )
                        last_count = n_complete
                except Exception:
                    pass  # Study may not have trials yet

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(n_workers):
                worker_trials = trials_per_worker + (1 if i < extra_trials else 0)
                futures.append(
                    executor.submit(_tune_worker, str(exp), worker_trials, timeout)
                )
            wait(futures)
            # Check for exceptions
            for f in futures:
                f.result()

        stop_monitor.set()
        monitor_thread.join(timeout=1.0)

        # Reload study to get all results
        study = optuna.load_study(study_name=study_name, storage=storage)

    elapsed = time.time() - start_time

    # Build report
    best = study.best_trial
    report = {
        "function": "lib.tune",
        "best": {
            "config": sample_config(space, best),
            "score": best.value,
            "trial_number": best.number,
        },
        "n_trials": len(study.trials),
        "time": elapsed,
    }

    dump_report(exp, report)

    # Print summary
    print(
        f"\nTuning complete: {len(study.trials)} trials in {datetime.timedelta(seconds=int(elapsed))}"
    )
    print(f"Best score: {best.value:.4f}")

    return report


# =============================================================================
# Evaluation (multi-seed)
# =============================================================================
def _eval_worker(exp_path: str, seed: int) -> dict:
    """
    Worker function for parallel evaluation. Must be at module level for pickling.
    """
    exp = Path(exp_path)
    config = load_config(exp)
    main_fn = import_function(config["function"])
    base_config = config["base_config"]

    seed_exp = exp / str(seed)
    report_path = seed_exp / "report.json"

    # Skip if already completed
    if report_path.exists():
        print(f"  Seed {seed}: already done, loading...")
        return load_report(seed_exp)

    # Run with this seed
    seed_config = {**base_config, "seed": seed}
    create_exp(seed_exp, seed_config, force=True)

    print(f"  Seed {seed}: running...")
    report = main_fn(seed_config, seed_exp)
    dump_report(seed_exp, report)
    return report


def create_eval_from_tuning(
    tuning_exp: str | Path,
    eval_exp: str | Path | None = None,
    n_seeds: int = 5,
    force: bool = False,
) -> Path:
    """
    Create evaluation config from tuning results.

    Args:
        tuning_exp: Tuning experiment directory (must have report.json)
        eval_exp: Evaluation experiment directory (default: tuning_exp/../evaluation)
        n_seeds: Number of seeds for evaluation
        force: Overwrite existing eval config

    Returns:
        Path to created evaluation experiment directory
    """
    tuning_exp = Path(tuning_exp)
    tuning_report = load_report(tuning_exp)
    tuning_config = load_config(tuning_exp)

    if eval_exp is None:
        eval_exp = tuning_exp.parent / "evaluation"
    eval_exp = Path(eval_exp)

    eval_config = {
        "function": tuning_config["function"],
        "n_seeds": n_seeds,
        "base_config": tuning_report["best"]["config"],
    }

    create_exp(eval_exp, eval_config, force=force)
    print(f"Created evaluation config: {eval_exp}")
    print(f"  Best score from tuning: {tuning_report['best']['score']:.4f}")

    return eval_exp


def evaluate(
    exp: str | Path,
    n_seeds: int | None = None,
    n_workers: int = 1,
) -> dict:
    """
    Multi-seed evaluation for reliability estimates.

    Config format (config.toml):
        function = "bin.lightgbm_.main"
        n_seeds = 5  # optional, defaults to 5

        [base_config]
        seed = 0  # will be replaced per-seed
        dataset = "california"
        # ... rest of the config

    Creates exp/0/, exp/1/, ... subdirectories with individual runs.
    Aggregates results with mean ± std.

    Args:
        exp: Experiment directory containing config.toml
        n_seeds: Number of seeds to run (default: 5)
        n_workers: Number of parallel workers (1 = sequential)

    Returns:
        Report dict with individual experiments and aggregated metrics
    """
    import time

    exp = Path(exp)
    config = load_config(exp)
    n_seeds = config.get("n_seeds", n_seeds) or 5

    start_time = time.time()
    print(f"Evaluating with {n_seeds} seeds...")

    if n_workers == 1:
        # Sequential - use inline function for simpler code path
        reports = []
        for seed in range(n_seeds):
            report = _eval_worker(str(exp), seed)
            reports.append(report)
    else:
        # Parallel - use module-level worker
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        worker = partial(_eval_worker, str(exp))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            reports = list(executor.map(worker, range(n_seeds)))

    elapsed = time.time() - start_time

    # Aggregate metrics
    aggregated = aggregate_metrics([r["metrics"] for r in reports])

    report = {
        "function": "lib.evaluate",
        "n_seeds": n_seeds,
        "experiments": [
            {"seed": i, "metrics": r["metrics"]} for i, r in enumerate(reports)
        ],
        "aggregated": aggregated,
        "time": elapsed,
    }

    dump_report(exp, report)

    # Print summary
    print(
        f"\nEvaluation complete: {n_seeds} seeds in {datetime.timedelta(seconds=int(elapsed))}"
    )
    print("Aggregated results:")
    for part, metrics in aggregated.items():
        score = metrics.get("score", "N/A")
        print(f"  {part}: {score}")

    return report


# =============================================================================
# Metrics Aggregation
# =============================================================================
def aggregate_metrics(
    metrics_list: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, str]]:
    """
    Aggregate metrics from multiple runs into mean ± std format.

    Args:
        metrics_list: List of metrics dicts, each like:
            {"train": {"rmse": 0.5, "score": -0.5}, "val": {...}, ...}

    Returns:
        Aggregated dict like:
            {"train": {"rmse": "0.50 ± 0.02", "score": "-0.50 ± 0.02"}, ...}
    """
    if not metrics_list:
        return {}

    # Get all parts and metric names
    parts = list(metrics_list[0].keys())
    result = {}

    for part in parts:
        part_metrics = [m[part] for m in metrics_list if part in m]
        if not part_metrics:
            continue

        metric_names = list(part_metrics[0].keys())
        result[part] = {}

        for metric_name in metric_names:
            values = [pm[metric_name] for pm in part_metrics]
            mean = np.mean(values)
            std = np.std(values)
            result[part][metric_name] = f"{mean:.4f} ± {std:.4f}"

    return result


# =============================================================================
# Tinygrad Extensions (Append to lib.py)
# =============================================================================
from tinygrad import Tensor, dtypes  # noqa: E402


def tinygrad_quantile(x: Tensor, q: Tensor, dim: int = 0) -> Tensor:
    """
    Compute quantiles of tensor along specified dimension.
    Equivalent to torch.quantile with interpolation='linear'.
    """
    if dim < 0:
        dim = x.ndim + dim

    # move target dim to front
    if dim != 0:
        perm = list(range(x.ndim))
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(*perm)

    n = x.shape[0]
    rest_shape = x.shape[1:]
    m = int(np.prod(rest_shape)) if rest_shape else 1

    # flatten to 2D and sort
    x_2d = x.reshape(n, m)
    x_sorted, _ = x_2d.sort(dim=0)

    # quantile positions in [0, n-1]
    pos = q * (n - 1)
    lo = pos.floor
    hi = (lo + 1).clip(0, n - 1)

    # interpolation weight
    t = pos - lo

    lo_i = lo.cast(dtypes.int32)
    hi_i = hi.cast(dtypes.int32)

    # expand indices for gather: (len(q),) -> (len(q), m)
    lo_idx = lo_i.reshape(-1, 1).expand(lo_i.shape[0], m)
    hi_idx = hi_i.reshape(-1, 1).expand(hi_i.shape[0], m)

    # gather and interpolate
    v_lo = x_sorted.gather(0, lo_idx)
    v_hi = x_sorted.gather(0, hi_idx)

    result = v_lo * (1.0 - t).reshape(-1, 1) + v_hi * t.reshape(-1, 1)

    out_shape = (q.shape[0],) + rest_shape
    return result.reshape(*out_shape)


def compute_quantile_bins(X: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """
    Compute quantile bins for numerical features using Tinygrad.
    Returns a list of numpy arrays (edges) for each feature.
    """
    # Move to Tensor for fast sorting/quantile on GPU
    t_X = Tensor(X)
    t_q = Tensor(np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32))

    # Compute quantiles: (n_bins+1, n_features)
    quantiles = tinygrad_quantile(t_X, t_q, dim=0)

    # Bring back to CPU/Numpy to unify duplicate bins and format for the model
    # (The bin edge creation is a one-time setup cost, fine to do on CPU after calculation)
    q_np = quantiles.numpy().T  # (n_features, n_bins+1)

    bins = []
    for feature_edges in q_np:
        # unique() automatically sorts
        unique_edges = np.unique(feature_edges)
        if len(unique_edges) < 2:
            # Fallback for constant features: slightly expand to avoid div/0
            unique_edges = np.array(
                [feature_edges[0] - 1e-3, feature_edges[0] + 1e-3], dtype=np.float32
            )
        bins.append(unique_edges)

    return bins
