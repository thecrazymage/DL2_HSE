#!/usr/bin/env python3
"""LightGBM training script for tabular data."""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import lib


def make_eval_fn(task: lib.Task, part_y: np.ndarray):
    """
    Create custom eval function for LightGBM that matches lib.Task scoring.

    This ensures early stopping uses the same metric as tuning/evaluation.
    """
    higher_is_better = lib.SCORE_SIGN.get(task.score_name, 1) > 0

    def eval_fn(y_pred: np.ndarray, dataset: lgb.Dataset):
        y_true = dataset.get_label()

        # For multiclass, LightGBM flattens predictions: reshape
        if task.is_multiclass:
            n_classes = task.compute_n_classes()
            y_pred = y_pred.reshape(-1, n_classes, order="F")

        # Determine prediction type
        pred_type = (
            lib.PredictionType.LABELS
            if task.is_regression
            else lib.PredictionType.PROBS
        )

        metrics = lib.compute_metrics(y_true, y_pred, task.type_, pred_type)
        score = metrics[task.score_name]

        return task.score_name, score, higher_is_better

    return eval_fn


def main(config: dict, exp_path: Path) -> dict:
    """
    Train a LightGBM model.

    Config format:
        seed = 0
        dataset = "california"

        [model]
        learning_rate = 0.05
        num_leaves = 31
        # ... other LightGBM parameters

        [train]
        n_estimators = 4000
        early_stopping_rounds = 50
    """
    seed = config.get("seed", 0)
    np.random.seed(seed)

    # Load and preprocess data
    ds = lib.Dataset.load(config["dataset"])
    ds = lib.preprocess(ds, num_method="standard", cat_method="ordinal", seed=seed)

    # Prepare features: concatenate numerical and categorical
    def get_features(part: str) -> np.ndarray:
        features = []
        if "x_num" in ds.data:
            features.append(ds.data["x_num"][part])
        if "x_cat" in ds.data:
            features.append(ds.data["x_cat"][part].astype(np.float32))
        return np.concatenate(features, axis=1) if len(features) > 1 else features[0]

    X_train = get_features("train")
    X_val = get_features("val")
    X_test = get_features("test")

    y_train = ds.data["y"]["train"]
    y_val = ds.data["y"]["val"]

    # Determine categorical feature indices
    cat_feature_indices = []
    if "x_num" in ds.data and "x_cat" in ds.data:
        n_num = ds.data["x_num"]["train"].shape[1]
        n_cat = ds.data["x_cat"]["train"].shape[1]
        cat_feature_indices = list(range(n_num, n_num + n_cat))

    # LightGBM parameters
    model_params = config.get("model", {})
    train_params = config.get("train", {})

    # Set objective based on task type
    if ds.task.is_regression:
        objective = "regression"
    elif ds.task.is_binclass:
        objective = "binary"
    else:
        objective = "multiclass"
        model_params["num_class"] = ds.task.compute_n_classes()

    lgb_params = {
        "objective": objective,
        "metric": "None",  # Use custom eval function instead
        "seed": seed,
        "verbose": -1,
        "force_col_wise": True,
        **model_params,
    }

    # Custom eval function that matches lib.Task scoring
    eval_fn = make_eval_fn(ds.task, y_val)

    n_estimators = train_params.get("n_estimators", 4000)
    early_stopping_rounds = train_params.get("early_stopping_rounds", 50)

    # Create datasets
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_set,
        categorical_feature=cat_feature_indices if cat_feature_indices else "auto",
    )

    # Train
    start_time = time.time()

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),  # Suppress logging
    ]

    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        feval=eval_fn,
        callbacks=callbacks,
    )

    train_time = time.time() - start_time

    # Predict
    predict = model.predict

    predictions = {
        "train": predict(X_train),
        "val": predict(X_val),
        "test": predict(X_test),
    }

    # Compute metrics
    prediction_type = (
        lib.PredictionType.LABELS if ds.task.is_regression else lib.PredictionType.PROBS
    )
    metrics = ds.task.calculate_metrics(predictions, prediction_type)

    return {
        "function": "bin.lightgbm_.main",
        "metrics": metrics,
        "best_iteration": model.best_iteration,
        "time": train_time,
    }


if __name__ == "__main__":
    # Quick test
    test_config = {
        "seed": 0,
        "dataset": "diamond",
        "model": {
            "learning_rate": 0.05,
            "num_leaves": 31,
        },
        "train": {
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
        },
    }

    exp_dir = Path("exp/test_lightgbm")
    lib.create_exp(exp_dir, test_config, force=True)
    lib.run(main, exp_dir, force=True)
