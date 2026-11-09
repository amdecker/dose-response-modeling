import jax.numpy as jnp


def df_to_jax_dict(data):
    """
    Convert pandas DataFrame to dictionary of JAX arrays for NumPyro model.

    Params:
    -------
        data: pd.DataFrame

    Returns:
        Dictionary with JAX arrays
    """
    return {col: jnp.array(data[col].values) for col in data}


pretty_model_names = {
    "linear": "Linear",
    "linear_quadratic": "Linear-Quadratic",
    "quadratic": "Quadratic",
    "linear_bart": "Linear & BART Effect Mod",
    "linear_quadratic_bart": "Linear-Quadratic & BART Effect Mod",
    "quadratic_bart": "Quadratic & BART Effect Mod",
    "bart": "Full BART",
    "bart_linear_leaves": "Full BART (Linear Leaves)",
    "varying_coefficient_bart": "Varying-Coefficient BART",
}
