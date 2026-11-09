import os

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_raw_data():
    """
    Get LSS data.

    Returns
    -------
    pd.DataFrame
        The raw LSS data.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, "src", "data", "lss14", "lss14.csv")
    raw_data = pd.read_csv(data_path)
    return raw_data


def scale_data(data):
    """
    Return a copy of the data with colon dose scaled to Grays and map the sex
    variable so -1 is male and 1 is female.
    """
    scaled = data.copy()
    # 1 is male, 2 is female
    scaled["sex"] = scaled["sex"].map({1: -1, 2: 1})

    scaled["colon10"] /= 1000
    return scaled


def add_stratum_id(data):
    """
    Add a stratum_id column to the data.
    The stratum_id is a unique identifier for each combination of city, sex, age
    at exposure category (agexcat), and age category (agecat).
    """
    data["stratum_id"] = pd.factorize(
        pd.Series(list(zip(data.city, data.sex, data.agexcat, data.agecat)))
    )[0]
    return data


def get_weighted_average(data, column):
    """
    Calculate the weighted average of a column in the data.
    The weights are given by the 'pyr' column.
    """
    return (data[column] * data["pyr"]).sum() / data["pyr"].sum()


def get_binned_data(data):
    """
    Bin the data by dose category and calculate the sum of pyr and solid cases
    for each dose category. Also calculate the weighted averages of colon dose,
    sex, and age for each dose category, weighted by pyr.
    """
    binned_data = data.groupby("dosecat")[["pyr", "solid"]].apply(
        lambda group: group.sum()
    )

    for col in ["colon10", "sex", "agex", "age"]:
        binned_data[col] = data.groupby("dosecat")[[col, "pyr"]].apply(
            lambda df: get_weighted_average(df, col)
        )

    return binned_data


def get_data():
    """
    Get the processed LSS data.

    - Converts mGy to Gy
    - add stratum id for stratification based on city, sex, age at exposure category
      (agexcat), and age category (agecat).

    Returns
    -------
    pd.DataFrame
        The processed LSS data
    """
    data = get_raw_data()
    data = scale_data(data)
    data = add_stratum_id(data)

    data.reset_index(drop=True, inplace=True)
    return data


def get_binned_err(data, reference_dosecat=1, ci=0.95, max_iter=100, tol=1e-10):
    """
    ERR in each dose category relative to the reference category, with Wald
    confidence intervals. Uses the same stratified likelihood as the models with
    one relative risk per dose category, fit with Newton's method.

    Params:
    -------
    data : pd.DataFrame
        LSS data with solid, pyr, dosecat, colon10 and stratum_id columns.
    reference_dosecat : int, optional
        Dose category whose relative risk is fixed at 1 (default 1, 0-5 mGy).
    ci : float, optional
        Confidence level of the intervals (default 0.95).
    max_iter : int, optional
        Maximum number of Newton iterations (default 100).
    tol : float, optional
        Convergence tolerance on the largest Newton step (default 1e-10).

    Returns
    -------
    pd.DataFrame
        Indexed by dosecat, with the person-year-weighted mean colon dose
        (colon10), err, lower and upper.
    """
    cells = data.groupby(["stratum_id", "dosecat"], as_index=False)[
        ["solid", "pyr"]
    ].sum()
    stratum = pd.factorize(cells.stratum_id)[0]
    dosecats = np.sort(cells.dosecat.unique())
    category = np.searchsorted(dosecats, cells.dosecat)
    free = dosecats != reference_dosecat
    n_strata, n_categories = stratum.max() + 1, len(dosecats)

    deaths = cells.solid.to_numpy(dtype=float)
    pyr = cells.pyr.to_numpy(dtype=float)
    stratum_deaths = np.bincount(stratum, weights=deaths, minlength=n_strata)
    category_deaths = np.bincount(category, weights=deaths, minlength=n_categories)

    log_rr = np.zeros(n_categories)
    for _ in range(max_iter):
        weight = pyr * np.exp(log_rr[category])
        stratum_weight = np.bincount(stratum, weights=weight, minlength=n_strata)
        share = weight / stratum_weight[stratum]

        # Share of each stratum's expected deaths in each dose category
        shares = np.zeros((n_strata, n_categories))
        np.add.at(shares, (stratum, category), share)

        expected = (stratum_deaths[:, None] * shares).sum(axis=0)
        gradient = category_deaths - expected
        information = np.diag(expected) - shares.T @ (stratum_deaths[:, None] * shares)

        step = np.linalg.solve(information[np.ix_(free, free)], gradient[free])
        log_rr[free] += step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError(f"Newton's method did not converge in {max_iter} iterations")

    standard_error = np.zeros(n_categories)
    standard_error[free] = np.sqrt(
        np.diag(np.linalg.inv(information[np.ix_(free, free)]))
    )
    z = norm.ppf(0.5 + ci / 2)

    binned = get_binned_data(data).loc[dosecats]
    return pd.DataFrame(
        {
            "colon10": binned.colon10.values,
            "err": np.exp(log_rr) - 1,
            "lower": np.exp(log_rr - z * standard_error) - 1,
            "upper": np.exp(log_rr + z * standard_error) - 1,
        },
        index=pd.Index(dosecats, name="dosecat"),
    )
