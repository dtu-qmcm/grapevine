import jax
from jax import numpy as jnp
from jaxtyping import Scalar


def rosenbrock2dmult(x: jax.Array, args: jax.Array) -> Scalar:
    """2 dimensional version of the Rosenbrock funtion.

    Solution is x * args = (1, 1).

    """
    xt = x * args
    return (1 - xt[0]) ** 2 + 100 * (xt[1] - xt[0] ** 2) ** 2


def rmm(s, p, km_s, km_p, vmax, k_eq):
    """Reversible Michaelis Menten rate law"""
    num = vmax / km_s * (s - p / k_eq)
    denom = 1 + s / km_s + p / km_p
    return num / denom


def ma(s, p, kf, keq):
    """Mass action rate law"""
    return kf * (s - p / keq)


def linear_pathway_steady_state(y, args):
    """Example function for testing root-finding algorithms.

    Example usage:

    ```python

    from collections import OrderedDict
    import jax.numpy as jnp
    import optimistix as optx

    solver = optx.Newton(rtol=1e-9, atol=1e-9)
    params = OrderedDict(
        log_km=jnp.array([2.0, 3.0]),
        log_vmax=jnp.array(0.0),
        log_keq=jnp.array([1.0, 1.0, 1.0]),
        log_kf=jnp.array([1.0, -1.0]),
        log_conc_ext=jnp.array([1.0, 0.0]),
    )
    guess = jnp.array([0.01, 0.01])

    sol = optx.root_find(
        linear_pathway_steady_state,
        solver,
        guess,
        args=params
    )
    ```
    """
    S = jnp.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]).transpose()
    c_m1_int, c_m2_int = y
    km, vmax, keq, kf, conc_ext = map(jnp.exp, args.values())
    keq_r1, keq_r2, keq_r3 = keq
    kf_r1, kf_r3 = kf
    c_m1_ext, c_m2_ext = conc_ext
    km_m1, km_m2 = km
    v = jnp.array(
        [
            ma(c_m1_ext, c_m1_int, kf_r1, keq_r1),
            rmm(c_m1_int, c_m2_int, km_m1, km_m2, vmax, keq_r2),
            ma(c_m2_int, c_m2_ext, kf_r3, keq_r3),
        ]
    )
    return (S @ v)[jnp.array([1, 2])]
