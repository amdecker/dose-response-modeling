"""
Cross-validation script for MCMC models.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from data import get_data
from mcmc import stratum_log_predictive_density
from run import MODELS, get_model


def summarize(fold_results):
    """
    Summary over folds. Deviance is per held-out stratum, with its standard
    error across strata (like PSIS-LOO). The other metrics are averaged over
    folds
    """
    density = np.concatenate([r["stratum_lpd"] for r in fold_results])
    summary = {
        "deviance": float(-2 * density.mean()),
        "deviance_se": float(2 * density.std(ddof=1) / np.sqrt(len(density))),
        "n_strata": int(len(density)),
    }
    for metric in ["mse", "rmse", "mae"]:
        values = [r[metric] for r in fold_results]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))

    print("\n" + "=" * 80)
    print("Cross-Validation Summary")
    print("=" * 80)
    print(
        f"DEVIANCE PER STRATUM: {summary['deviance']:.4f} ± {summary['deviance_se']:.4f}"
        f" ({summary['n_strata']} strata)"
    )
    for metric in ["mse", "rmse", "mae"]:
        print(
            f"{metric.upper()}: {summary[f'{metric}_mean']:.4f}"
            f" ± {summary[f'{metric}_std']:.4f}"
        )
    return summary


def run_cross_validation(
    model_name,
    n_splits=5,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    output_dir=None,
    fold_index=None,
    random_seed=42,
    **kwargs,
):
    """
    Run K-fold cross-validation for MCMC models. Folds hold out whole strata.

    Parameters:
    -----------
    model_name : str
        Name of the model to use
    n_splits : int
        Number of folds for cross-validation
    num_warmup : int
        Number of warmup samples for MCMC
    num_samples : int
        Number of samples to draw from posterior
    num_chains : int
        Number of MCMC chains to run
    output_dir : str
        Directory to save results
    fold_index : int, optional
        If specified, only run this fold (0-indexed)
    random_seed : int
        Random seed for shuffling strata into folds
    **kwargs : dict
        Additional model-specific parameters

    Returns:
    --------
    results : dict
        Dictionary containing cross-validation results
    """
    if fold_index is not None and not 0 <= fold_index < n_splits:
        raise ValueError(f"fold_index must be in [0, {n_splits}), got {fold_index}")

    # Settings that must match for fold results to be combined
    settings = {
        "n_splits": n_splits,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "random_seed": random_seed,
        **kwargs,
    }

    # Load data
    print("Loading data...")
    data = get_data()

    y = data["solid"].values
    X = data[["colon10", "agex", "age", "sex", "pyr", "stratum_id"]]

    # Initialize K-fold over strata
    kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    groups = X["stratum_id"].values

    # Storage for results
    fold_results = []

    # If fold_index is specified, only run that fold
    if fold_index is not None:
        print(f"\nRunning fold {fold_index} for {model_name} model")
        folds_to_run = [(fold_index, list(kf.split(X, groups=groups))[fold_index])]
    else:
        print(f"\nRunning {n_splits}-fold cross-validation for {model_name} model")
        folds_to_run = list(enumerate(kf.split(X, groups=groups)))

    print(f"Total samples: {len(X)}")
    print("-" * 80)

    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in folds_to_run:
        print(f"\nFold {fold_idx + 1}/{n_splits}")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize and fit model
        model = get_model(
            model_name,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            **kwargs,
        )

        print("Fitting model...")
        model.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = np.asarray(model.predict(X_test, y_test)).reshape(-1, len(y_test))
        y_pred_mean = y_pred.mean(axis=0)

        density, stratum_ids = stratum_log_predictive_density(X_test, y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred_mean)
        mae = mean_absolute_error(y_test, y_pred_mean)
        rmse = np.sqrt(mse)

        fold_result = {
            "fold": fold_idx + 1,
            "deviance": float(-2 * density.mean()),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "stratum_ids": [int(s) for s in stratum_ids],
            "stratum_lpd": [float(v) for v in density],
        }

        fold_results.append(fold_result)
        print(f"  Deviance per stratum: {fold_result['deviance']:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

        # Free the fitted model (and its posterior) before the next fold
        del model, y_pred
        gc.collect()

    results = {
        "model": model_name,
        "n_splits": n_splits,
        "settings": settings,
        "fold_results": fold_results,
    }
    if fold_index is None:
        results["summary"] = summarize(fold_results)

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON - include fold index in filename if running single fold
        if fold_index is not None:
            json_path = output_path / f"{model_name}_fold_{fold_index}_cv_results.json"
        else:
            json_path = output_path / f"{model_name}_cv_results.json"

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    return results


def aggregate_fold_results(model_name, n_splits, output_dir):
    """
    Aggregate results from individual fold files.

    Parameters:
    -----------
    model_name : str
        Name of the model
    n_splits : int
        Number of folds
    output_dir : str
        Directory containing fold results

    Returns:
    --------
    results : dict
        Aggregated cross-validation results
    """
    output_path = Path(output_dir)
    fold_files = [
        output_path / f"{model_name}_fold_{fold_idx}_cv_results.json"
        for fold_idx in range(n_splits)
    ]
    missing = [f.name for f in fold_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold results: {missing}")

    fold_results = []
    settings = None
    for fold_file in fold_files:
        with open(fold_file, "r") as f:
            fold_data = json.load(f)
        if settings is None:
            settings = fold_data["settings"]
        elif fold_data["settings"] != settings:
            raise ValueError(
                f"{fold_file.name} was run with different settings: "
                f"{fold_data['settings']} vs {settings}"
            )
        fold_results.extend(fold_data["fold_results"])

    if settings["n_splits"] != n_splits:
        raise ValueError(f"Folds were run with n_splits={settings['n_splits']}")

    results = {
        "model": model_name,
        "n_splits": n_splits,
        "settings": settings,
        "fold_results": fold_results,
        "summary": summarize(fold_results),
    }

    # Save aggregated results
    json_path = output_path / f"{model_name}_cv_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAggregated results saved to: {json_path}")

    return results


def main():
    """Main entry point for cross-validation script."""
    parser = argparse.ArgumentParser(
        description="Run K-fold cross-validation for MCMC models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help=f"Model to use: {list(MODELS.keys())}",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of MCMC samples (default: 1000)",
    )

    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1000,
        help="Number of MCMC warmup samples (default: 1000)",
    )

    parser.add_argument(
        "--num-chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )

    parser.add_argument(
        "--num-trees",
        type=int,
        default=100,
        help="Number of trees for BART models (default: 100)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )

    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Run only this fold (0-indexed). If not specified, run all folds.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for K-fold splits (default: 42)",
    )

    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate results from individual fold files",
    )

    args = parser.parse_args()
    if args.aggregate and args.output_dir is None:
        parser.error("--aggregate needs --output-dir")
    if args.fold_index is not None and not 0 <= args.fold_index < args.n_splits:
        parser.error(f"--fold-index must be in [0, {args.n_splits})")

    # If aggregating, just aggregate and exit
    if args.aggregate:
        aggregate_fold_results(
            model_name=args.model,
            n_splits=args.n_splits,
            output_dir=args.output_dir,
        )
        print("\nAggregation completed!")
        return

    # Run cross-validation
    run_cross_validation(
        model_name=args.model,
        n_splits=args.n_splits,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        output_dir=args.output_dir,
        fold_index=args.fold_index,
        random_seed=args.random_seed,
        num_trees=args.num_trees,
    )

    print("\nCross-validation completed!")


if __name__ == "__main__":
    main()
