#!/usr/bin/env python3
"""
CLI launcher for tuning and evaluation.

Usage:
    uv run python -m bin.go tune exp/lightgbm/california/tuning -n 100 -w 4
    uv run python -m bin.go eval exp/lightgbm/california/evaluation -s 5 -w 4
    uv run python -m bin.go tune-eval exp/lightgbm/california/tuning -n 100 -s 5 -w 4
    uv run python -m bin.go make-eval exp/lightgbm/california/tuning
"""
import argparse
from pathlib import Path

import lib


def cmd_tune(args):
    """Run hyperparameter tuning."""
    lib.tune(
        exp=args.exp,
        n_trials=args.trials,
        n_workers=args.workers,
        timeout=args.timeout,
        force=args.force,
    )


def cmd_evaluate(args):
    """Run multi-seed evaluation."""
    lib.evaluate(
        exp=args.exp,
        n_seeds=args.seeds,
        n_workers=args.workers,
    )


def cmd_run(args):
    """Run a single experiment."""
    config = lib.load_config(args.exp)
    main_fn = lib.import_function(config["function"])
    lib.run(main_fn, args.exp, force=args.force)


def cmd_make_eval(args):
    """Create evaluation config from tuning results."""
    lib.create_eval_from_tuning(
        tuning_exp=args.exp,
        eval_exp=args.output,
        n_seeds=args.seeds,
        force=args.force,
    )


def cmd_tune_eval(args):
    """Run tuning followed by evaluation."""
    # Run tuning
    lib.tune(
        exp=args.exp,
        n_trials=args.trials,
        n_workers=args.workers,
        timeout=args.timeout,
        force=args.force,
    )

    # Create evaluation config from tuning results
    eval_exp = lib.create_eval_from_tuning(
        tuning_exp=args.exp,
        n_seeds=args.seeds,
        force=True,  # Always overwrite eval config after fresh tuning
    )

    # Run evaluation
    lib.evaluate(
        exp=eval_exp,
        n_seeds=args.seeds,
        n_workers=args.workers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="nanotabdl experiment launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument("exp", type=Path, help="Tuning experiment directory")
    tune_parser.add_argument("-n", "--trials", type=int, default=100, help="Number of trials")
    tune_parser.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers")
    tune_parser.add_argument("-t", "--timeout", type=float, default=None, help="Timeout in seconds")
    tune_parser.add_argument("-f", "--force", action="store_true", help="Restart from scratch")
    tune_parser.set_defaults(func=cmd_tune)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", aliases=["eval"], help="Run multi-seed evaluation")
    eval_parser.add_argument("exp", type=Path, help="Evaluation experiment directory")
    eval_parser.add_argument("-s", "--seeds", type=int, default=5, help="Number of seeds")
    eval_parser.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers")
    eval_parser.set_defaults(func=cmd_evaluate)

    # tune-eval command (combined)
    te_parser = subparsers.add_parser("tune-eval", help="Tune then evaluate (all-in-one)")
    te_parser.add_argument("exp", type=Path, help="Tuning experiment directory")
    te_parser.add_argument("-n", "--trials", type=int, default=100, help="Number of trials")
    te_parser.add_argument("-s", "--seeds", type=int, default=5, help="Number of eval seeds")
    te_parser.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers")
    te_parser.add_argument("-t", "--timeout", type=float, default=None, help="Tuning timeout")
    te_parser.add_argument("-f", "--force", action="store_true", help="Restart tuning from scratch")
    te_parser.set_defaults(func=cmd_tune_eval)

    # make-eval command
    me_parser = subparsers.add_parser("make-eval", help="Create eval config from tuning")
    me_parser.add_argument("exp", type=Path, help="Tuning experiment directory")
    me_parser.add_argument("-o", "--output", type=Path, default=None, help="Output dir (default: ../evaluation)")
    me_parser.add_argument("-s", "--seeds", type=int, default=5, help="Number of seeds")
    me_parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing")
    me_parser.set_defaults(func=cmd_make_eval)

    # run command (single experiment)
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("exp", type=Path, help="Experiment directory")
    run_parser.add_argument("-f", "--force", action="store_true", help="Force re-run")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
