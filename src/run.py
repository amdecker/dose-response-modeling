"""
Run MCMC models and save inference data.
"""

import argparse
import os
from pathlib import Path

import arviz as az

from data import get_data
from figure_data import compute_figure_data
from mcmc import (
    BartLinearLeavesMCMC,
    BartMCMC,
    LinearBartMCMC,
    LinearMCMC,
    LinearQuadraticBartMCMC,
    LinearQuadraticMCMC,
    QuadraticBartMCMC,
    QuadraticMCMC,
    VaryingCoefficientBartMCMC,
)

MODELS = {
    "linear": LinearMCMC,
    "linear_quadratic": LinearQuadraticMCMC,
    "quadratic": QuadraticMCMC,
    "bart": BartMCMC,
    "bart_linear_leaves": BartLinearLeavesMCMC,
    "varying_coefficient_bart": VaryingCoefficientBartMCMC,
    "linear_bart": LinearBartMCMC,
    "linear_quadratic_bart": LinearQuadraticBartMCMC,
    "quadratic_bart": QuadraticBartMCMC,
}


def convergence_summary(idata):
    """
    Max r_hat, min bulk ESS and divergences. For BART, use differences between
    probe rows since the overall level isn't identified
    """
    posterior = idata.posterior
    if "bart_probe" in posterior:
        probe = posterior["bart_probe"]
        contrasts = (probe - probe.isel(probe_row=0)).isel(probe_row=slice(1, None))
        posterior = posterior.drop_vars("bart_probe").assign(
            bart_probe_contrast=contrasts
        )
    summary = {
        "max r_hat": f"{float(az.rhat(posterior).to_array().max()):.3f}",
        "min bulk ESS": f"{float(az.ess(posterior).to_array().min()):.0f}",
    }
    if "diverging" in idata.sample_stats:
        summary["divergences"] = int(idata.sample_stats.diverging.sum())
    return summary


def save_netcdf(data, path):
    """
    Write to a temporary file and then rename it, so the save doesn't fail if
    something else has the old file open
    """
    temporary = path.with_name(f"{path.stem}.tmp{os.getpid()}{path.suffix}")
    try:
        data.to_netcdf(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def get_model(model_name, num_warmup, num_samples, num_chains=4, **kwargs):
    """
    Get the appropriate model class based on model name.

    Parameters:
    -----------
    model_name : str
        Name of the model to use
    num_warmup : int
        Number of warmup samples for MCMC
    num_samples : int
        Number of samples to draw from posterior
    num_chains : int
        Number of MCMC chains to run
    **kwargs : dict
        Additional model-specific parameters

    Returns:
    --------
    model : MCMCRegressor
        Instantiated model object
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODELS.keys())}"
        )

    model_class = MODELS[model_name]

    # BART models need num_trees parameter
    if "bart" in model_name.lower():
        num_trees = kwargs.get("num_trees", 100)
        return model_class(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            num_trees=num_trees,
        )
    else:
        return model_class(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )


def run_model(
    model_name,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    output_dir=None,
    figures_dir=None,
    **kwargs,
):
    """
    Run MCMC model and save inference data and the figure data for plots.ipynb

    Parameters:
    -----------
    model_name : str
        Name of the model to use
    num_warmup : int
        Number of warmup samples for MCMC
    num_samples : int
        Number of samples to draw from posterior
    num_chains : int
        Number of MCMC chains to run
    output_dir : str
        Directory to save results
    figures_dir : str, optional
        Directory to save the figure data
    **kwargs : dict
        Additional model-specific parameters

    Returns:
    --------
    idata : arviz.InferenceData
        Inference data object containing MCMC results
    """
    # Load data
    print("Loading data...")
    data = get_data()

    y = data["solid"].values
    X = data[["colon10", "agex", "age", "sex", "pyr", "stratum_id"]]

    print(f"\nRunning {model_name} model")
    print(f"Total samples: {len(X)}")
    print("MCMC settings:")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples: {num_samples}")
    print(f"  Chains: {num_chains}")
    if "bart" in model_name.lower():
        print(f"  Trees: {kwargs.get('num_trees', 100)}")
    print("-" * 80)

    # Initialize and fit model
    model = get_model(
        model_name,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        **kwargs,
    )

    print("\nFitting model...")
    model.fit(X, y)

    print("\nModel fitting completed!")

    # Convergence summary
    print("\nConvergence:")
    for name, value in convergence_summary(model.idata).items():
        print(f"  {name}: {value}")
    print("  (aim for r_hat <= 1.01, ESS >= 400, no divergences)")

    # Save inference data
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as NetCDF (arviz format)
        idata_path = output_path / f"{model_name}_idata.nc"
        save_netcdf(model.idata, idata_path)
        print(f"\nInference data saved to: {idata_path}")

    if figures_dir:
        print("\nComputing figure data...")
        figures_path = Path(figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        figures_file = figures_path / f"{model_name}_figures.nc"
        save_netcdf(compute_figure_data(model, X), figures_file)
        print(f"Figure data saved to: {figures_file}")

    return model.idata


def main():
    """Main entry point for running MCMC models."""
    parser = argparse.ArgumentParser(
        description="Run MCMC models and save inference data"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help=f"Model to use: {list(MODELS.keys())}",
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
        "--figures-dir",
        type=str,
        default=None,
        help="Directory to save the data for plots.ipynb's figures",
    )

    args = parser.parse_args()

    # Run model
    run_model(
        model_name=args.model,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        num_trees=args.num_trees,
    )

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
