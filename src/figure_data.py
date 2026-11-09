"""
Data for the plots in plots.ipynb. Computed right after fitting in run.py since
the BART trees aren't saved.

Every model gets the ERR at the reference profile (agex 30, age 70, averaged
over sex) on a dose grid. Models with an effect modifier also get its partial
dependence on agex, age and sex. Full BART gets each sampled row's log RR at
1 Gy instead.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from mcmc import BartMCMC, ParametricBartMCMC, ParametricMCMC

# Gy. Linear and log spaced so both the linear and log x-axis plots look smooth
DOSES = np.unique(np.r_[np.linspace(0, 3, 61), np.geomspace(1e-3, 3, 100)])

# The reference profile, one row per sex
REFERENCE_ROWS = pd.DataFrame(
    {
        "colon10": 0.0,
        "agex": 30.0,
        "age": 70.0,
        "sex": [-1.0, 1.0],
        "pyr": 1.0,
        "stratum_id": 0,
    }
)

EFFECT_MOD_GRIDS = {
    "agex": np.linspace(0, 85, 18),
    "age": np.linspace(10, 100, 19),
    "sex": np.array([-1.0, 1.0]),
}

# Dose at which pure BART's effect modification is shown
EFFECT_MOD_DOSE = 1.0


def sample_rows(X):
    """Random rows, plus some old ones so the ends of the curves are covered"""
    return pd.concat(
        [
            X[X.age > 90].sample(10, random_state=0),
            X[X.agex > 80].sample(10, random_state=1),
            X.sample(180, random_state=2),
        ]
    )


def _chain_draw(values, model):
    """Reshape samples x ... to chain x draw x ..."""
    posterior = model.idata.posterior
    return np.asarray(values).reshape(
        posterior.sizes["chain"], posterior.sizes["draw"], *np.shape(values)[-1:]
    )


def _flat_beta(model):
    beta = model.idata.posterior.beta.values
    return beta.reshape(-1, *beta.shape[2:])


def err_reference(model):
    """ERR at the reference profile for every posterior draw: chain x draw x dose"""
    if isinstance(model, ParametricMCMC):
        # effect_mod is 1 at the reference profile
        dose_data = {"colon10": jnp.asarray(DOSES)}
        curves = jax.vmap(lambda b: model.predict_err(b, dose_data))(_flat_beta(model))
    elif isinstance(model, ParametricBartMCMC):
        # BART effect_mod is also 1 at the reference profile
        beta = np.moveaxis(_flat_beta(model), 0, -1)[..., None]
        curves = model.err(beta, DOSES)
    else:
        # Full BART: exp(g(d, x) - g(0, x)) - 1, averaged over sex
        log_rr = bart_log_rr_vs_zero(model, REFERENCE_ROWS, DOSES)
        curves = (np.exp(log_rr) - 1).mean(axis=-1).reshape(-1, len(DOSES))
    return _chain_draw(curves, model)


def bart_log_rr_vs_zero(model, rows, doses):
    """g(d, x) - g(0, x) for full BART: chain x draw x dose x row"""
    frames = [rows.assign(colon10=float(d)) for d in [0.0, *doses]]
    prediction = model.predict(pd.concat(frames), return_full_predictive=True)
    log_rr = prediction.posterior_predictive.log_rr.values
    log_rr = log_rr.reshape(*log_rr.shape[:2], len(frames), len(rows))
    return log_rr[:, :, 1:] - log_rr[:, :, :1]


def log_effect_mod_pd(model, rows, variable, grid):
    """
    Partial dependence of the log effect modifier on one covariate. Returns the
    ICE curves (row x grid, posterior mean) and the partial dependence for every
    draw (chain x draw x grid)
    """
    frames = [rows.assign(**{variable: float(value)}) for value in grid]
    prediction = model.predict(pd.concat(frames), return_full_predictive=True)
    if isinstance(model, ParametricMCMC):
        log_mod = np.log(np.asarray(prediction["effect_mod"]))
    else:
        log_mod = prediction.posterior_predictive.rel_log_effect_mod.values
    log_mod = log_mod.reshape(-1, len(grid), len(rows))
    ice = log_mod.mean(axis=0).T
    return ice, _chain_draw(log_mod.mean(axis=-1), model)


def compute_figure_data(model, X):
    """Everything plots.ipynb needs for this model"""
    data = xr.Dataset(
        {"err_reference": (("chain", "draw", "dose"), err_reference(model))},
        coords={"dose": DOSES},
        attrs={
            "err_reference": "ERR at age at exposure 30, attained age 70, "
            "averaged over sex"
        },
    )

    rows = sample_rows(X)
    if isinstance(model, BartMCMC):
        log_rr = bart_log_rr_vs_zero(model, rows, [EFFECT_MOD_DOSE])[:, :, 0]
        data["log_rr_at_dose"] = ("row", log_rr.mean(axis=(0, 1)))
        data.attrs["log_rr_at_dose"] = (
            f"posterior mean log relative risk at {EFFECT_MOD_DOSE} Gy relative to "
            "0 Gy, per sampled row"
        )
    else:
        for variable, grid in EFFECT_MOD_GRIDS.items():
            ice, pd_draws = log_effect_mod_pd(model, rows, variable, grid)
            data = data.assign_coords({f"{variable}_grid": grid})
            data[f"ice_{variable}"] = (("row", f"{variable}_grid"), ice)
            data[f"pd_{variable}"] = (("chain", "draw", f"{variable}_grid"), pd_draws)

    for column in ["agex", "age", "sex"]:
        data[f"row_{column}"] = ("row", rows[column].to_numpy(dtype=float))
    return data
