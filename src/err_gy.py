"""
ERR/Gy for dose ranges 0 to c, like Ozasa et al. (2012), Fig. 5.

Fits separate linear slopes below and above c, so beta[0] is the ERR/Gy for 0
to c. "full" is the ordinary linear model.

Example usage:
# Fit one cut point
uv run python src/err_gy.py --task fit --model linear --dose-limit 0.1
uv run python src/err_gy.py --task fit --model linear_bart --dose-limit full

# Collect the fits into a table and Fig. 5-style plot
uv run python src/err_gy.py --task plot

# All cut points on SLURM
./run_err_gy.sh
"""

import argparse
import json
from pathlib import Path

import arviz  # noqa: F401  # before scienceplots, since arviz resets matplotlib styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from data import get_data
from mcmc import LinearBartMCMC, LinearMCMC, LinearSplitBartMCMC, LinearSplitMCMC
from run import convergence_summary

assert scienceplots  # registers the "science" and "notebook" styles

# (split model, full-range model) for each model name
MODELS = {
    "linear": (LinearSplitMCMC, LinearMCMC),
    "linear_bart": (LinearSplitBartMCMC, LinearBartMCMC),
}
MODEL_LABELS = {"linear": "Linear", "linear_bart": "Linear + BART"}

# Cut points (Gy) from Ozasa et al. (2012), Fig. 5
DOSE_LIMITS = [
    0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.125,
    0.15, 0.175, 0.2, 0.25, 0.5, 1.0, 2.0,
]  # fmt: skip

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "results" / "err_gy"


def limit_label(dose_limit):
    return "full" if dose_limit is None else f"{dose_limit:g}"


def parse_dose_limit(value):
    return None if value == "full" else float(value)


def beta_summary(draws):
    """Posterior summary of one slope"""
    draws = np.asarray(draws).reshape(-1)
    return {
        "mean": float(draws.mean()),
        "median": float(np.median(draws)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "prob_positive": float((draws > 0).mean()),
    }


def fit_dose_limit(model_name, dose_limit, settings, output_dir):
    """Fit one cut point (None for the full dose range) and save the results"""
    split_class, full_class = MODELS[model_name]
    kwargs = {
        "num_warmup": settings["num_warmup"],
        "num_samples": settings["num_samples"],
        "num_chains": settings["num_chains"],
    }
    if "bart" in model_name:
        kwargs["num_trees"] = settings["num_trees"]
        kwargs["random_seed"] = settings["random_seed"]
    if dose_limit is None:
        model = full_class(**kwargs)
    else:
        model = split_class(dose_limit=dose_limit, **kwargs)

    data = get_data()
    y = data["solid"].values
    X = data[["colon10", "agex", "age", "sex", "pyr", "stratum_id"]]

    label = limit_label(dose_limit)
    print(f"Fitting {model_name} with dose limit {label}")
    model.fit(X, y)

    beta = model.idata.posterior["beta"]
    if dose_limit is None:
        below = np.ones(len(X), dtype=bool)
        low, high = beta, None
    else:
        below = X.colon10.values < dose_limit
        low, high = beta[..., 0], beta[..., 1]

    results = {
        "model": model_name,
        "dose_limit": label,
        "settings": settings,
        "deaths_below": float(y[below].sum()),
        "rows_below": int(below.sum()),
        "person_years_below": float(X.pyr.values[below].sum()),
        "beta_low": beta_summary(low),
        "beta_high": None if high is None else beta_summary(high),
        "convergence": convergence_summary(model.idata),
    }
    print(json.dumps(results, indent=2))

    folder = Path(output_dir) / model_name
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"dl-{label}.json", "w") as f:
        json.dump(results, f, indent=2)
    beta.to_netcdf(folder / f"dl-{label}_beta.nc")
    print(f"Saved to {folder / f'dl-{label}.json'}")
    return results


def collect_results(output_dir):
    """One row per model and cut point"""
    rows = []
    for path in sorted(Path(output_dir).glob("*/dl-*.json")):
        with open(path) as f:
            result = json.load(f)
        convergence = result["convergence"]
        rows.append(
            {
                "model": result["model"],
                "dose_limit": result["dose_limit"],
                "deaths_below": result["deaths_below"],
                **{f"low_{k}": v for k, v in result["beta_low"].items()},
                **{f"high_{k}": v for k, v in (result["beta_high"] or {}).items()},
                "max_r_hat": float(convergence["max r_hat"]),
                "min_ess": float(convergence["min bulk ESS"]),
                "divergences": convergence.get("divergences"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No fits found in {output_dir}")
    df = pd.DataFrame(rows)
    df["is_full"] = df.dose_limit == "full"
    df["dose_limit_gy"] = pd.to_numeric(df.dose_limit.where(~df.is_full))
    return df.sort_values(["model", "is_full", "dose_limit_gy"]).reset_index(drop=True)


def plot_err_gy(df, path):
    """ERR/Gy for 0 to c with 95% CIs, like Ozasa et al. (2012), Fig. 5"""
    colors = {"linear": "black", "linear_bart": "C1"}
    markers = {"linear": "o", "linear_bart": "s"}
    models = [m for m in MODELS if m in set(df.model)]
    # Offset the models so the intervals don't overlap
    offsets = dict(
        zip(models, 0.04 * np.linspace(-1, 1, len(models)) * (len(models) > 1))
    )
    full_x = 2 * max(DOSE_LIMITS)

    with plt.style.context(["science", "notebook"]):
        fig, ax = plt.subplots(dpi=150)
        for model in models:
            rows = df[df.model == model]
            x = np.where(rows.is_full, full_x, rows.dose_limit_gy)
            ax.errorbar(
                x * 10 ** offsets[model],
                rows.low_mean,
                yerr=[
                    rows.low_mean - rows.low_ci_lower,
                    rows.low_ci_upper - rows.low_mean,
                ],
                fmt=markers[model],
                color=colors[model],
                markersize=4,
                elinewidth=1,
                capsize=0,
                label=MODEL_LABELS[model],
            )
            full = rows[rows.is_full]
            if len(full):
                ax.axhline(
                    full.low_mean.iloc[0],
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                )
        ax.axhline(0, color="black", linewidth=0.8, zorder=0)
        ax.set_xscale("log")
        ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
        ax.set_xticks(ticks + [full_x], [f"{t:g}" for t in ticks] + ["Full\nRange"])
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlim(0.015, full_x * 1.4)
        # Separate the full-range estimate from the cut points
        ax.axvline(
            np.sqrt(max(DOSE_LIMITS) * full_x), color="gray", linewidth=1, linestyle=":"
        )
        ax.set_xlabel("Dose Limit (Gy)")
        ax.set_ylabel("ERR/Gy")
        if len(models) > 1:
            ax.legend()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ERR/Gy for dose ranges 0 to c")
    parser.add_argument("--task", choices=["fit", "plot"], required=True)
    parser.add_argument("--model", choices=list(MODELS), help="Model to fit")
    parser.add_argument(
        "--dose-limit",
        type=str,
        help="Cut point in Gy, or 'full' for the whole dose range",
    )
    parser.add_argument("--num-samples", type=int, default=2500)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-trees", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if args.task == "fit":
        if args.model is None or args.dose_limit is None:
            parser.error("--task fit needs --model and --dose-limit")
        settings = {
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "num_trees": args.num_trees,
            "random_seed": args.random_seed,
        }
        fit_dose_limit(
            args.model, parse_dose_limit(args.dose_limit), settings, args.output_dir
        )
    else:
        df = collect_results(args.output_dir)
        output_dir = Path(args.output_dir)
        df.to_csv(output_dir / "err_gy_results.csv", index=False)
        plot_err_gy(df, output_dir / "err_gy_plot.png")
        print(df.to_string())
        print(f"Saved {output_dir / 'err_gy_results.csv'} and err_gy_plot.png")


if __name__ == "__main__":
    main()
