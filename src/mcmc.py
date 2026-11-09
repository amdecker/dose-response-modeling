import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import pytensor.tensor as pt
from jax import random
from jax.scipy.special import gammaln
from numpyro.infer import MCMC, NUTS, Predictive, init_to_uniform
from scipy.spatial import cKDTree
from scipy.special import gammaln as np_gammaln
from scipy.special import logsumexp as np_logsumexp
from sklearn.base import BaseEstimator, RegressorMixin

from utils import df_to_jax_dict

# Set the number of JAX devices to 8 for parallel processing
numpyro.set_host_device_count(8)


# Each stratum's baseline is integrated out (see calculus.ipynb), which leaves a
# multinomial likelihood for how a stratum's deaths are split across its rows.
# So predictions are expected deaths given the stratum's total deaths.

# Log likelihood for draws with RR <= 0. Finite so PGBART's particle weights
# don't become NaN
INVALID_LOG_LIK = -1e20

# Mean and sd of the Normal prior on each dose coefficient
BETA_PRIOR = (0, 10)


# Positive relative risks
# -----------------------
# RR = 1 + effect_mod * sum_k beta[k] * basis[k](dose) has to be positive in
# every row, and NUTS diverges against that hard edge. So each coefficient is
# sampled as beta[k] = bound[k] + exp(z[k]), where bound[k] is the smallest value
# that keeps RR positive in the rows it is responsible for (rows where its dose
# term is the first nonzero one). With the Normal prior on beta[k] plus the
# Jacobian this is the same posterior, just without the edge.


def combine_dose_terms(beta, basis):
    """sum_k beta[k] * basis[k], with a scalar beta when there is one term."""
    if len(basis) == 1:
        return beta * basis[0]
    return sum(beta[k] * term for k, term in enumerate(basis))


def positive_rr_rows(basis):
    """For each dose coefficient, the indices of the rows it is responsible for."""
    terms = np.stack([np.asarray(term) for term in basis], axis=1)
    if (terms < 0).any():
        raise ValueError("Dose terms must be non-negative")
    nonzero = terms > 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), -1)
    return [np.flatnonzero(first == k) for k in range(len(basis))]


def positive_rr_init(basis, rows):
    """Initial values that put every coefficient at 0 when effect_mod is 1"""
    values = {}
    for k, term in enumerate(basis):
        if len(rows[k]) == 0:
            values[f"beta_{k}"] = 0.0
        else:
            # bound = -1 / max(term) when the later betas are 0 and effect_mod is 1
            values[f"beta_z{k}"] = -np.log(np.asarray(term)[rows[k]].max())
    return values


def positive_rr_beta(basis, effect_mod, rows, backend):
    """
    Dose coefficients that keep RR positive in every row (see "Positive relative
    risks" above). `backend` is NumpyroBeta or PyMCBeta
    """
    n = len(basis)
    beta = [None] * n
    for k in reversed(range(n)):
        if len(rows[k]) == 0:
            beta[k] = backend.free(f"beta_{k}")
            continue
        index = rows[k]
        rest = sum(beta[j] * basis[j][index] for j in range(k + 1, n))
        bound = backend.xnp.max((-1 / effect_mod[index] - rest) / basis[k][index])
        beta[k] = backend.above(f"beta_z{k}", bound)
    return beta[0] if n == 1 else backend.xnp.stack(beta)


class NumpyroBeta:
    """numpyro sampling sites for positive_rr_beta"""

    xnp = jnp

    @staticmethod
    def free(name):
        return numpyro.sample(name, dist.Normal(*BETA_PRIOR))

    @staticmethod
    def above(name, bound):
        z = numpyro.sample(name, dist.ImproperUniform(dist.constraints.real, (), ()))
        beta = bound + jnp.exp(z)
        # Prior on beta, and the Jacobian of z -> beta
        numpyro.factor(f"{name}_prior", dist.Normal(*BETA_PRIOR).log_prob(beta) + z)
        return beta


class PyMCBeta:
    """PyMC variables for positive_rr_beta"""

    xnp = pt

    @staticmethod
    def free(name):
        return pm.Normal(name, *BETA_PRIOR)

    @staticmethod
    def above(name, bound):
        z = pm.Flat(name)
        beta = bound + pt.exp(z)
        # Prior on beta, and the Jacobian of z -> beta
        pm.Potential(f"{name}_prior", pm.logp(pm.Normal.dist(*BETA_PRIOR), beta) + z)
        return beta


def _init_near_null(site=None, values=None):
    """
    Start the dose coefficients at 0 and everything else within 0.1 of 0
    (effect modifier near 1). With numpyro's default radius of 2, some chains
    start with huge effect modifiers and get stuck
    """
    if values and site["name"] in values:
        return jnp.asarray(values[site["name"]])
    return init_to_uniform(site, radius=0.1)


def encode_strata(X):
    """Stratum codes 0, 1, ... for the rows of X"""
    return pd.factorize(X["stratum_id"])[0]


def stratified_log_lik(log_rr, y, pyr, stratum, n_strata):
    """
    Per-stratum multinomial log likelihood and each row's expected deaths given
    its stratum's total.

    Log weights are centered within each stratum before exp (doesn't change the
    likelihood), otherwise the unidentified level of log_rr in BartMCMC can
    overflow
    """
    stratum_size = jax.ops.segment_sum(jnp.ones_like(pyr), stratum, n_strata)
    log_weight = jnp.log(pyr) + log_rr
    stratum_mean = jax.ops.segment_sum(log_weight, stratum, n_strata) / stratum_size
    log_weight = log_weight - stratum_mean[stratum]
    weight = jnp.exp(log_weight)
    stratum_weight = jax.ops.segment_sum(weight, stratum, n_strata)
    stratum_deaths = jax.ops.segment_sum(y, stratum, n_strata)
    log_lik = (
        jax.ops.segment_sum(y * log_weight - gammaln(y + 1), stratum, n_strata)
        - stratum_deaths * jnp.log(stratum_weight)
        + gammaln(stratum_deaths + 1)
    )
    expected = stratum_deaths[stratum] * weight / stratum_weight[stratum]
    return log_lik, expected


def stratified_log_lik_pt(log_rr, y, pyr, stratum):
    """PyTensor version of stratified_log_lik"""
    n_strata = stratum.max() + 1

    def segment_sum(values):
        return pt.inc_subtensor(pt.zeros((n_strata,))[stratum], values)

    stratum_size = segment_sum(pt.ones_like(pyr))
    log_weight = pt.log(pyr) + log_rr
    stratum_mean = segment_sum(log_weight) / stratum_size
    log_weight = log_weight - stratum_mean[stratum]
    weight = pt.exp(log_weight)
    stratum_weight = segment_sum(weight)
    stratum_deaths = segment_sum(y)
    log_lik = (
        segment_sum(y * log_weight - pt.gammaln(y + 1))
        - stratum_deaths * pt.log(stratum_weight)
        + pt.gammaln(stratum_deaths + 1)
    )
    expected = stratum_deaths[stratum] * weight / stratum_weight[stratum]
    return log_lik, expected


def add_stratified_likelihood_pm(
    log_rr, y, pyr, stratum, strata_with_deaths, valid=None
):
    """
    Add the stratified likelihood to the current PyMC model, plus "rate" and
    "stratum_log_lik" (for PSIS-LOO). If `valid` is false the log likelihood is
    INVALID_LOG_LIK
    """
    log_lik, expected = stratified_log_lik_pt(log_rr, y, pyr, stratum)
    pm.Deterministic("rate", expected)
    pm.Deterministic("stratum_log_lik", log_lik[strata_with_deaths])
    total = log_lik.sum()
    if valid is not None:
        total = pt.switch(valid, total, INVALID_LOG_LIK)
    pm.Potential("obs", total)


def strata_with_deaths(stratum, y):
    """
    Codes of strata with at least one death. The others always have log
    likelihood 0, so they're left out of the pointwise log likelihood
    """
    deaths = np.bincount(stratum, weights=y, minlength=stratum.max() + 1)
    return np.flatnonzero(deaths > 0)


def stratum_ids_with_deaths(X, y):
    """stratum_id of the strata with deaths"""
    stratum = encode_strata(X)
    return pd.unique(X["stratum_id"])[strata_with_deaths(stratum, np.asarray(y))]


def move_stratum_log_lik(idata, stratum_ids):
    """Move stratum_log_lik from the posterior to the log_likelihood group"""
    log_lik = idata.posterior["stratum_log_lik"]
    log_lik = log_lik.rename({log_lik.dims[-1]: "stratum"})
    log_lik = log_lik.assign_coords(stratum=stratum_ids)
    del idata.posterior["stratum_log_lik"]
    idata.add_groups(log_likelihood=log_lik.to_dataset(name="obs"))
    return idata


def stratum_log_predictive_density(X, y, expected):
    """
    Log posterior predictive density of each stratum's split of deaths, for
    strata with deaths. `expected` is draws x rows of expected deaths
    """
    y = np.asarray(y, dtype=float)
    expected = np.asarray(expected).reshape(-1, len(y))
    stratum = encode_strata(X)
    n_strata = stratum.max() + 1
    stratum_deaths = np.bincount(stratum, weights=y, minlength=n_strata)

    # Multinomial log pmf per draw and stratum
    with np.errstate(divide="ignore", invalid="ignore"):
        log_share = np.log(expected) - np.log(stratum_deaths[stratum])
        row_terms = np.where(y > 0, y * log_share, 0.0) - np_gammaln(y + 1)
    log_lik = np.zeros((expected.shape[0], n_strata))
    np.add.at(log_lik.T, stratum, row_terms.T)
    log_lik += np_gammaln(stratum_deaths + 1)

    has_deaths = stratum_deaths > 0
    density = np_logsumexp(log_lik[:, has_deaths], axis=0) - np.log(expected.shape[0])
    stratum_ids = pd.unique(X["stratum_id"])[has_deaths]
    return density, stratum_ids


class MCMCRegressor(RegressorMixin, BaseEstimator, ABC):
    """Generic MCMC regressor"""

    def __init__(self, num_warmup=1000, num_samples=1000, num_chains=4):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains

    def fit(self, X, y):
        self.is_fitted_ = True

    @abstractmethod
    def predict(self, X, y=None, return_full_predictive=False):
        pass

    def score(self, X, y):
        """Mean deviance per stratum (same quantity as PSIS-LOO)"""
        density, _ = stratum_log_predictive_density(X, y, self.predict(X, y))
        return -2 * density.mean()


class ParametricMCMC(MCMCRegressor, ABC):
    """
    Numpyro NUTS sampler with parametric dose response with parametric effect
    modification
    """

    def __init__(
        self,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        effect_mod=True,
        rng_key=None,
    ):
        super().__init__(num_warmup, num_samples, num_chains)
        self.effect_mod = effect_mod
        self.rng_key = rng_key

    @property
    def _rng_key(self):
        return random.PRNGKey(0) if self.rng_key is None else self.rng_key

    def model(self, data_dict, rows=None, store_rows=True):
        """
        When fitting, beta is sampled with positive_rr_beta using `rows`. When
        predicting (rows=None), beta is a plain site filled in from the
        posterior, since the bounds shouldn't be recomputed for new data.

        Per-row quantities are only stored if `store_rows`, to keep the
        posterior small
        """
        record = numpyro.deterministic if store_rows else (lambda name, value: value)
        basis = self.dose_basis(data_dict["colon10"])

        if self.effect_mod:
            tau = numpyro.sample("tau", dist.Normal(0, 10))
            nu = numpyro.sample("nu", dist.Normal(0, 10))
            sigma = numpyro.sample("sigma", dist.Uniform(-1, 1))

            sex_mod = 1 + sigma * data_dict["sex"]
            effect_mod = record(
                "effect_mod",
                jnp.exp(
                    tau * (data_dict["agex"] - 30) / 10
                    + nu * jnp.log(data_dict["age"] / 70)
                )
                * sex_mod,
            )
        else:
            effect_mod = jnp.ones_like(data_dict["colon10"])

        if rows is None:
            shape = () if len(basis) == 1 else (len(basis),)
            beta = numpyro.sample("beta", dist.Normal(*BETA_PRIOR).expand(shape))
        else:
            beta = numpyro.deterministic(
                "beta", positive_rr_beta(basis, effect_mod, rows, NumpyroBeta)
            )

        excess_relative_risk = combine_dose_terms(beta, basis) * effect_mod
        relative_risk = 1 + excess_relative_risk

        log_lik, expected = stratified_log_lik(
            jnp.log(relative_risk),
            data_dict["y"],
            data_dict["pyr"],
            data_dict["stratum"],
            data_dict["n_strata"],
        )
        record("rate", expected)
        numpyro.deterministic(
            "stratum_log_lik", log_lik[data_dict["strata_with_deaths"]]
        )
        numpyro.factor("obs", log_lik.sum())

    @abstractmethod
    def dose_basis(self, dose):
        """Non-negative dose terms, ERR = sum_k beta[k] * term[k]"""

    def predict_err(self, beta, data_dict):
        """ERR at the reference profile"""
        return combine_dose_terms(beta, self.dose_basis(data_dict["colon10"]))

    @staticmethod
    def _data_dict(X, y):
        data_dict = df_to_jax_dict(X[["colon10", "agex", "age", "sex", "pyr"]])
        stratum = encode_strata(X)
        y = np.zeros(len(X)) if y is None else np.asarray(y, dtype=float)
        data_dict["y"] = jnp.array(y)
        data_dict["stratum"] = jnp.array(stratum)
        data_dict["n_strata"] = int(stratum.max() + 1)
        data_dict["strata_with_deaths"] = jnp.array(strata_with_deaths(stratum, y))
        return data_dict

    def fit(self, X, y):
        super().fit(X, y)
        data_dict = self._data_dict(X, y)
        # Same float32 doses as the model, so rows near a cut point match
        basis = [np.asarray(term) for term in self.dose_basis(data_dict["colon10"])]
        rows = positive_rr_rows(basis)

        self.mcmc = MCMC(
            NUTS(
                lambda: self.model(data_dict, rows=rows, store_rows=False),
                init_strategy=partial(
                    _init_near_null, values=positive_rr_init(basis, rows)
                ),
            ),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        self.mcmc.run(self._rng_key)
        self.idata = move_stratum_log_lik(
            az.from_numpyro(self.mcmc, log_likelihood=False),
            stratum_ids_with_deaths(X, y),
        )

        self.mcmc.print_summary()

        return self

    def predict(self, X, y=None, return_full_predictive=False):
        """Posterior predictive. "rate" needs y because of the stratum totals"""
        if y is None and not return_full_predictive:
            raise ValueError(
                "Predicted deaths are conditional on stratum totals: pass y"
            )
        data_dict = self._data_dict(X, y)

        # beta itself, not the z sites
        samples = self.mcmc.get_samples()
        predictive = Predictive(
            lambda: self.model(data_dict),
            posterior_samples={
                name: samples[name]
                for name in ("beta", "tau", "nu", "sigma")
                if name in samples
            },
        )
        posterior_predictive_samples = predictive(self._rng_key)
        predicted = np.asarray(posterior_predictive_samples["rate"])
        if return_full_predictive:
            return posterior_predictive_samples
        return predicted


class LinearMCMC(ParametricMCMC):
    def dose_basis(self, dose):
        return [dose]


def split_linear_basis(dose, dose_limit):
    """
    Separate linear slopes below and above dose_limit (Ozasa et al. 2012, Fig. 5).
    beta[0] is the ERR/Gy for 0 to dose_limit
    """
    below = dose < dose_limit
    return [dose * below, dose * (1 - below)]


class LinearSplitMCMC(ParametricMCMC):
    """Linear dose response with separate slopes below and above `dose_limit`"""

    def __init__(
        self,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        effect_mod=True,
        rng_key=None,
        dose_limit=1.0,
    ):
        super().__init__(num_warmup, num_samples, num_chains, effect_mod, rng_key)
        self.dose_limit = dose_limit

    def dose_basis(self, dose):
        return split_linear_basis(dose, self.dose_limit)


class LinearQuadraticMCMC(ParametricMCMC):
    def dose_basis(self, dose):
        return [dose, dose**2]


class QuadraticMCMC(ParametricMCMC):
    def dose_basis(self, dose):
        return [dose**2]


class ChainSeededPGBART(pmb.PGBART):
    """
    PGBART uses NumPy's global RNG, so in pymc-bart 0.11 parallel chains come
    out identical. Seed it from the chain's own RNG instead
    """

    def set_rng(self, rng):
        super().set_rng(rng)
        np.random.seed(self.rng.integers(2**32))
        self.normal.update()
        self.uniform.update()


_trim_tree = pmb.tree.Tree.trim


def _trim_tree_linear_leaves(self):
    """
    pymc-bart 0.11 keeps a fitted value per training row on stored linear
    leaves. Prediction only needs linear_params, so drop them
    """
    tree = _trim_tree(self)
    for node in tree.tree_structure.values():
        if node.is_leaf_node() and node.linear_params is not None:
            node.value = np.zeros(1)
    return tree


pmb.tree.Tree.trim = _trim_tree_linear_leaves


class ProgressLogger:
    """
    Print progress every `every` iterations of each chain, since PyMC's progress
    bar doesn't work in SLURM logs
    """

    def __init__(self, num_warmup, num_samples, every=100):
        self.total = num_warmup + num_samples
        self.every = every
        self.start = time.monotonic()

    def __call__(self, trace, draw):
        iteration = draw.draw_idx + 1
        if iteration % self.every and not draw.is_last:
            return
        elapsed = time.monotonic() - self.start
        phase = "tune" if draw.tuning else "draw"
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] chain {draw.chain}: "
            f"{iteration}/{self.total} ({phase}), {elapsed / 60:.1f} min elapsed, "
            f"{elapsed / iteration:.2f} s/iteration",
            flush=True,
        )


def evaluate_tree_sets(all_trees, X):
    """Output of each stored BART tree set at the rows of X: tree sets x rows"""
    output = np.zeros((len(all_trees), X.shape[0]))
    for index, tree_set in enumerate(all_trees):
        for output_trees in tree_set:
            for tree in output_trees:
                output[index] += np.asarray(tree.predict(x=X, shape=1)).reshape(-1)
    return output


class PyMCRegressor(MCMCRegressor, ABC):
    """
    PyMC models with a BART component.

    pymc-bart's stored tree sets aren't in draw order, so they can't be paired
    with the other parameters from the same draw. So the posterior keeps the BART
    output at a few probe rows, and predict uses those to match each draw to its
    tree set
    """

    BART_NAME = None
    # Sex uses OneHotSplitRule, since the continuous rule jitters repeated values
    # in training but not in prediction
    SPLIT_RULES = None
    N_PROBE_ROWS = 25

    def __init__(
        self,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        num_trees=100,
        random_seed=0,
    ):
        super().__init__(num_warmup, num_samples, num_chains)
        self.num_trees = num_trees
        self.random_seed = random_seed

    @abstractmethod
    def _bart_inputs(self, X):
        """BART covariate matrix for the rows of X"""

    @abstractmethod
    def _model_data(self, X, y):
        """Values for the model's pm.Data containers"""

    @staticmethod
    def _stratum_data(X, y):
        stratum = encode_strata(X)
        y = np.zeros(len(X)) if y is None else np.asarray(y, dtype=float)
        return {
            "y": y,
            "pyr": X.pyr.values,
            "stratum": stratum,
            "strata_with_deaths": strata_with_deaths(stratum, y),
        }

    def _sample(self, bart_inputs, stratum_ids, initvals=None):
        """Sample, then keep only the BART output at the probe rows"""
        stored = [rv.name for rv in self.model.free_RVs] + ["stratum_log_lik"]
        if "beta" in self.model.named_vars and "beta" not in stored:
            stored.append("beta")
        self.idata = pm.sample(
            tune=self.num_warmup,
            draws=self.num_samples,
            chains=self.num_chains,
            step=[ChainSeededPGBART([self.bart])],
            init="adapt_diag",
            initvals=initvals,
            var_names=stored,
            random_seed=self.random_seed,
            callback=ProgressLogger(self.num_warmup, self.num_samples),
        )
        self.idata = move_stratum_log_lik(self.idata, stratum_ids)

        rng = np.random.default_rng(self.random_seed)
        n_rows = bart_inputs.shape[0]
        probe = np.sort(
            rng.choice(n_rows, size=min(self.N_PROBE_ROWS, n_rows), replace=False)
        )
        self.probe_X_ = bart_inputs[probe]
        bart_output = self.idata.posterior[self.BART_NAME]
        self.idata.posterior["bart_probe"] = bart_output.isel(
            {bart_output.dims[-1]: probe}
        ).rename({bart_output.dims[-1]: "probe_row"})
        del self.idata.posterior[self.BART_NAME]
        self.draw_tree_set_ = None

    def _bart_draws(self, bart_inputs):
        """BART output at the given rows for every posterior draw, in draw order"""
        all_trees = list(self.bart.owner.op.all_trees)
        values = evaluate_tree_sets(all_trees, np.vstack([self.probe_X_, bart_inputs]))
        n_probe = len(self.probe_X_)

        if self.draw_tree_set_ is None:
            probe = self.idata.posterior["bart_probe"]
            probe_draws = probe.values.reshape(-1, probe.shape[-1])
            distance, self.draw_tree_set_ = cKDTree(values[:, :n_probe]).query(
                probe_draws
            )
            if distance.max() > 1e-6 * (1 + np.abs(probe_draws).max()):
                raise RuntimeError(
                    "Could not match posterior draws to BART tree sets "
                    f"(largest mismatch {distance.max():.3g})"
                )

        posterior = self.idata.posterior
        shape = (posterior.sizes["chain"], posterior.sizes["draw"], -1)
        return values[self.draw_tree_set_, n_probe:].reshape(shape)

    def _predictive_model(self, X, y, bart_inputs):
        """The model with its data set to the rows of X"""
        with self.model:
            pm.set_data({"X": bart_inputs, **self._model_data(X, y)})
        return self.model

    def _predict(self, X, y, deterministics, return_full_predictive):
        if y is None and not return_full_predictive:
            raise ValueError(
                "Predicted deaths are conditional on stratum totals: pass y"
            )
        bart_inputs = self._bart_inputs(X)
        bart = self._bart_draws(bart_inputs)

        model = self._predictive_model(X, y, bart_inputs)
        posterior = self.idata.posterior
        values = posterior[
            [rv.name for rv in model.free_RVs if rv.name != self.BART_NAME]
        ].copy()
        values[self.BART_NAME] = (("chain", "draw", "bart_row"), bart)

        with model:
            predictions = pm.compute_deterministics(
                values, var_names=deterministics, progressbar=False
            )
        predictions[self.BART_NAME] = values[self.BART_NAME].isel(
            bart_row=slice(0, len(X))
        )
        if return_full_predictive:
            return az.InferenceData(posterior_predictive=predictions)
        return predictions["rate"].values


class BartMCMC(PyMCRegressor):
    """
    BART for the log relative risk. The level of g is absorbed by the stratum
    baselines, so the ERR is exp(g(d, x) - g(0, x)) - 1
    """

    BART_NAME = "log_rr"
    RESPONSE = "constant"
    SPLIT_RULES = [
        pmb.ContinuousSplitRule,  # colon10
        pmb.ContinuousSplitRule,  # agex
        pmb.ContinuousSplitRule,  # age
        pmb.OneHotSplitRule,  # sex
    ]

    def _bart_inputs(self, X):
        return X[["colon10", "agex", "age", "sex"]].to_numpy(dtype=float)

    def _model_data(self, X, y):
        return self._stratum_data(X, y)

    def fit(self, X, y):
        super().fit(X, y)
        bart_inputs = self._bart_inputs(X)
        # Rough guess as to the scale of log rr values
        init_y = np.random.default_rng(self.random_seed).normal(0, 1, size=len(y))

        with pm.Model() as self.model:
            X_mutable = pm.Data("X", bart_inputs)
            data = {
                name: pm.Data(name, values)
                for name, values in self._model_data(X, y).items()
            }

            log_rr = pmb.BART(
                "log_rr",
                X=X_mutable,
                Y=init_y,
                m=self.num_trees,
                split_rules=self.SPLIT_RULES,
                response=self.RESPONSE,
            )
            add_stratified_likelihood_pm(
                log_rr,
                data["y"],
                data["pyr"],
                data["stratum"],
                data["strata_with_deaths"],
            )
            self.bart = log_rr
            self._sample(bart_inputs, stratum_ids_with_deaths(X, y))

    def predict(self, X, y=None, return_full_predictive=False):
        return self._predict(X, y, ["rate"], return_full_predictive)


class BartLinearLeavesMCMC(BartMCMC):
    """BartMCMC with linear leaves (experimental in pymc-bart)"""

    RESPONSE = "linear"


class VaryingCoefficientBartMCMC(BartMCMC):
    """
    BART for the ERR per Gy: RR(d, x) = 1 + d * g(d, x), with g a BART (linear
    leaves) of dose, age at exposure, attained age and sex.

    RR is exactly 1 at zero dose, so unlike BartMCMC there's no unidentified
    level. Draws with RR <= 0 get INVALID_LOG_LIK, and log_rr is NaN there
    """

    BART_NAME = "err_per_gy"
    RESPONSE = "linear"

    def _model_data(self, X, y):
        return {"dose": X.colon10.values, **self._stratum_data(X, y)}

    def fit(self, X, y):
        MCMCRegressor.fit(self, X, y)
        bart_inputs = self._bart_inputs(X)
        # Rough guess as to the scale of the excess relative risk per Gy
        init_y = np.random.default_rng(self.random_seed).normal(0, 1, size=len(y))

        with pm.Model() as self.model:
            X_mutable = pm.Data("X", bart_inputs)
            data = {
                name: pm.Data(name, values)
                for name, values in self._model_data(X, y).items()
            }

            err_per_gy = pmb.BART(
                self.BART_NAME,
                X=X_mutable,
                Y=init_y,
                m=self.num_trees,
                split_rules=self.SPLIT_RULES,
                response=self.RESPONSE,
            )
            relative_risk = 1 + data["dose"] * err_per_gy
            pm.Deterministic("log_rr", pt.log(relative_risk))
            positive = relative_risk > 0
            add_stratified_likelihood_pm(
                pt.log(pt.switch(positive, relative_risk, 1.0)),
                data["y"],
                data["pyr"],
                data["stratum"],
                data["strata_with_deaths"],
                valid=pt.all(positive),
            )
            self.bart = err_per_gy
            self._sample(bart_inputs, stratum_ids_with_deaths(X, y))

    def predict(self, X, y=None, return_full_predictive=False):
        return self._predict(X, y, ["rate", "log_rr"], return_full_predictive)


class ParametricBartMCMC(PyMCRegressor, ABC):
    """
    Parametric dose response with a BART effect modifier. The effect modifier
    is divided by its sex-average at agex 30, age 70, so beta is identified and
    means the same thing as in ParametricMCMC
    """

    BART_NAME = "log_effect_mod"
    SPLIT_RULES = [
        pmb.ContinuousSplitRule,  # agex
        pmb.ContinuousSplitRule,  # age
        pmb.OneHotSplitRule,  # sex
    ]

    # Reference profile (agex, age, sex), appended to the BART inputs
    REFERENCE_X = np.array([[30.0, 70.0, -1.0], [30.0, 70.0, 1.0]])

    @abstractmethod
    def dose_basis(self, dose):
        """Non-negative dose terms, ERR = sum_k beta[k] * term[k]"""

    def err(self, beta, dose):
        """ERR at the reference profile"""
        return combine_dose_terms(beta, self.dose_basis(dose))

    def _bart_inputs(self, X):
        """Effect modifier covariates with the reference profile appended"""
        covariates = X[["agex", "age", "sex"]].to_numpy(dtype=float)
        return np.vstack([covariates, self.REFERENCE_X])

    def _model_data(self, X, y):
        return {"dose": X.colon10.values, **self._stratum_data(X, y)}

    def _dose_response(self, data, log_effect_mod, beta=None, rows=None):
        """
        Add the RR and likelihood to the current model. beta is sampled with
        positive_rr_beta when fitting and given when predicting
        """
        n_ref = len(self.REFERENCE_X)
        # Divide by the sex-average at the reference profile
        log_reference = pm.math.logsumexp(log_effect_mod[-n_ref:]) - np.log(n_ref)
        rel_log_effect_mod = pm.Deterministic(
            "rel_log_effect_mod", log_effect_mod[:-n_ref] - log_reference
        )
        effect_mod = pm.math.exp(rel_log_effect_mod)
        basis = self.dose_basis(data["dose"])
        if beta is None:
            beta = pm.Deterministic(
                "beta", positive_rr_beta(basis, effect_mod, rows, PyMCBeta)
            )
        relative_risk = 1 + combine_dose_terms(beta, basis) * effect_mod
        # positive_rr_beta already does this for training rows, but guard
        # against rounding and new rows
        positive = relative_risk > 0
        add_stratified_likelihood_pm(
            pt.log(pt.switch(positive, relative_risk, 1.0)),
            data["y"],
            data["pyr"],
            data["stratum"],
            data["strata_with_deaths"],
            valid=pt.all(positive),
        )

    def fit(self, X, y):
        super().fit(X, y)
        bart_inputs = self._bart_inputs(X)
        basis = self.dose_basis(X.colon10.values)
        rows = positive_rr_rows(basis)
        # Rough guess as to the scale of log effect modifiers (around 0)
        init_y = np.random.default_rng(self.random_seed).normal(
            0, 1, size=len(bart_inputs)
        )

        with pm.Model() as self.model:
            X_mutable = pm.Data("X", bart_inputs)
            data = {
                name: pm.Data(name, values)
                for name, values in self._model_data(X, y).items()
            }

            log_effect_mod = pmb.BART(
                "log_effect_mod",
                X=X_mutable,
                Y=init_y,
                m=self.num_trees,
                split_rules=self.SPLIT_RULES,
            )
            self._dose_response(data, log_effect_mod, rows=rows)
            self.bart = log_effect_mod
            self._sample(
                bart_inputs,
                stratum_ids_with_deaths(X, y),
                initvals=positive_rr_init(basis, rows),
            )

    def _predictive_model(self, X, y, bart_inputs):
        """Copy of the model with beta and the BART output as plain inputs"""
        with pm.Model() as model:
            data = {
                name: pm.Data(name, values)
                for name, values in self._model_data(X, y).items()
            }
            log_effect_mod = pm.Flat(self.BART_NAME, shape=len(bart_inputs))
            beta = pm.Flat("beta", shape=self.idata.posterior["beta"].shape[2:])
            self._dose_response(data, log_effect_mod, beta=beta)
        return model

    def predict(self, X, y=None, return_full_predictive=False):
        return self._predict(
            X, y, ["rate", "rel_log_effect_mod"], return_full_predictive
        )


class LinearBartMCMC(ParametricBartMCMC):
    """Linear dose response with BART effect modifier"""

    def dose_basis(self, dose):
        return [dose]


class LinearSplitBartMCMC(ParametricBartMCMC):
    """Split linear dose response with BART effect modifier"""

    def __init__(
        self,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        num_trees=100,
        random_seed=0,
        dose_limit=1.0,
    ):
        super().__init__(num_warmup, num_samples, num_chains, num_trees, random_seed)
        self.dose_limit = dose_limit

    def dose_basis(self, dose):
        return split_linear_basis(dose, self.dose_limit)


class LinearQuadraticBartMCMC(ParametricBartMCMC):
    """Linear-quadratic dose response with BART effect modifier"""

    def dose_basis(self, dose):
        return [dose, dose**2]


class QuadraticBartMCMC(ParametricBartMCMC):
    """Quadratic dose response with BART effect modifier"""

    def dose_basis(self, dose):
        return [dose**2]
