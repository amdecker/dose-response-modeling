"""
Compute PSIS-LOO for a single model. There's one log likelihood value per
stratum with deaths, so this is leave-one-stratum-out.

Usage: python compute_psis_loo.py <model_name>
"""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr


def relative_efficiency(log_lik):
    """Relative efficiency from the pointwise likelihoods, like R loo's relative_eff"""
    likelihood = np.exp(log_lik - log_lik.max(dim=("chain", "draw")))
    n_samples = log_lik.sizes["chain"] * log_lik.sizes["draw"]
    ess = az.ess(likelihood.to_dataset(name="obs"), method="mean").obs
    return float(ess.mean()) / n_samples


def stratum_loo(log_likelihood):
    """PSIS-LOO over strata"""
    log_lik = log_likelihood.obs
    idata = az.InferenceData(log_likelihood=log_likelihood)
    return az.loo(
        idata, pointwise=True, reff=relative_efficiency(log_lik), scale="deviance"
    )


def compute_psis_loo(model_name: str, idata_dir=None):
    """
    Compute PSIS-LOO for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'linear', 'bart', 'linear_quadratic')
    idata_dir : str, optional
        Directory with the InferenceData file (default: results/idata)
    """
    # Read idata file
    default_folder = Path(__file__).parent.parent / "results" / "idata"
    idata_folder = Path(idata_dir) if idata_dir else default_folder
    nc_file = idata_folder / f"{model_name}_idata.nc"

    if not nc_file.exists():
        raise FileNotFoundError(f"InferenceData file not found: {nc_file}")

    # Only load the log_likelihood group
    print(f"Loading {model_name} log likelihood from {nc_file}")
    log_likelihood = xr.open_dataset(nc_file, group="log_likelihood").load()

    # Compute PSIS-LOO
    print(f"Computing PSIS-LOO for {model_name}...")
    loo = stratum_loo(log_likelihood)

    # Save results
    results_folder = Path(__file__).parent.parent / "results" / "psis_loo"
    results_folder.mkdir(parents=True, exist_ok=True)

    output_file = results_folder / f"{model_name}_loo.json"
    # Extract relevant metrics
    results = {
        "model": model_name,
        "loo": float(loo.elpd_loo),
        "loo_se": float(loo.se),
        "p_loo": float(loo.p_loo),
        "n_samples": int(loo.n_samples),
        "n_data_points": int(loo.n_data_points),
        "warning": loo.warning,
        "scale": loo.scale,
        "max_pareto_k": float(loo.pareto_k.max()),
        "n_pareto_k_above_0.7": int((loo.pareto_k > 0.7).sum()),
        # Pointwise values per stratum, for paired comparison
        "stratum_ids": [int(s) for s in loo.loo_i.stratum.values],
        "loo_i": [float(v) for v in loo.loo_i.values],
    }

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"LOO: {results['loo']:.4f} ± {results['loo_se']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute PSIS-LOO for a given model")
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model (e.g., 'linear', 'bart', 'linear_quadratic')",
    )

    parser.add_argument(
        "--idata-dir",
        type=str,
        default=None,
        help="Directory with the InferenceData file (default: results/idata)",
    )

    args = parser.parse_args()

    compute_psis_loo(args.model_name, args.idata_dir)


if __name__ == "__main__":
    main()
