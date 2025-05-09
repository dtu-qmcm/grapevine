"""A steady kinetic model of a linear pathway with this structure:

    Aext <-r1-> Aint <-r2-> Bint <-r3-> Bext

Reactions r1 and r3 behave according to the law of mass action, and reaction r2 according to the Michaelis Menten rate law.

We assume we have measurements of Aint and Bint, as well as plenty of information about all the kinetic parameters and boundary conditions, and that the pathway is in a steady state, so that the concentrations c_m1_int and c_m2_int are not changing.

Note that for this model a guess is a three-tuple where the first value is the previous root-finder solution, the second value is the previous parameter PyTree and the final value is the cumulative number of Newton steps taken in this trajectory.

"""

from collections import OrderedDict
from functools import partial

from jax import numpy as jnp
from jax.scipy.stats import norm
import jax
import optimistix as optx
from grapevine.heuristics import (
    guess_previous,
    guess_implicit,
    guess_implicit_cg,
)

jax.config.update("jax_enable_x64", True)

TRUE_PARAMS = OrderedDict(
    log_km=jnp.array([2.0, 2.0]),
    log_vmax=jnp.array(3.0),
    log_keq=jnp.array([1.0, 1.0, 1.0]),
    log_kf=jnp.array([1.0, -1.0]),
    log_conc_ext=jnp.array([1.0, 0.0]),
)

DEFAULT_GUESS = jnp.array([0.1, 0.1])
SOLVER = optx.Newton(rtol=1e-5, atol=1e-5)
ERROR_SD = 0.05
PARAM_SD = 0.02


def get_default_guess(*args):
    return DEFAULT_GUESS


def rmm(s, p, km_s, km_p, vmax, k_eq):
    """Reversible Michaelis Menten rate law"""
    num = vmax / km_s * (s - p / k_eq)
    denom = 1 + s / km_s + p / km_p
    return num / denom


def ma(s, p, kf, keq):
    """Mass action rate law"""
    return kf * (s - p / keq)


def dxdt(x, args):
    """Rate of change of balanced metabolite concentrations."""
    S = jnp.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]).transpose()
    c_m1_int, c_m2_int = x
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


def joint_logdensity(params, obs, guess_info, gfunc):
    """Compute the joint log density (log posterior) for the kinetic model.

    Args:
        params: OrderedDict containing model parameters (log_km, log_vmax, log_keq, log_kf, log_conc_ext)
        obs: Array of observed metabolite concentrations
        guess: Tuple of (previous_solution, previous_params, previous_steps)
        gfunc: Function to generate initial guess for the root finder

    Returns:
        tuple: (log_density, (steady_state, params, total_steps))
            - log_density: Sum of log prior and log likelihood
            - steady_state: Computed steady state concentrations
            - params: Model parameters used
            - total_steps: Cumulative number of Newton steps
    """
    last_solution, _, previous_steps = guess_info
    use_default = jnp.isclose(last_solution, DEFAULT_GUESS).all()
    guess = jax.lax.cond(
        use_default, get_default_guess, gfunc, guess_info, params
    )

    def solve(params):
        sol = optx.root_find(
            dxdt,
            SOLVER,
            guess,
            args=params,
            max_steps=int(1e5),
        )
        return sol.value, jnp.array(sol.stats["num_steps"])

    steady, steps_i = solve(params)
    log_km, log_vmax, log_keq, log_kf, log_conc_ext = params.values()
    log_prior = jnp.sum(
        norm.logpdf(log_km, loc=TRUE_PARAMS["log_km"], scale=0.1).sum()
        + norm.logpdf(log_vmax, loc=TRUE_PARAMS["log_vmax"], scale=0.1).sum()
        + norm.logpdf(log_keq, loc=TRUE_PARAMS["log_keq"], scale=0.1).sum()
        + norm.logpdf(log_kf, loc=TRUE_PARAMS["log_kf"], scale=0.1).sum()
        + norm.logpdf(
            log_conc_ext, loc=TRUE_PARAMS["log_conc_ext"], scale=0.1
        ).sum()
    )
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(steady), scale=jnp.full(obs.shape, ERROR_SD)
    ).sum()
    steps = previous_steps + steps_i
    return log_prior + log_likelihood, (steady, params, steps)


def simulate(key, params, guess):
    sol = optx.root_find(dxdt, SOLVER, guess, args=params)
    return sol.value, jnp.exp(
        jnp.log(sol.value)
        + jax.random.normal(key, shape=sol.value.shape) * ERROR_SD
    )


joint_logdensity_guess_default = partial(
    joint_logdensity, gfunc=lambda g, p: DEFAULT_GUESS
)
joint_logdensity_guess_previous = partial(
    joint_logdensity, gfunc=guess_previous
)
joint_logdensity_guess_implicit = partial(
    joint_logdensity, gfunc=partial(guess_implicit, target_function=dxdt)
)
joint_logdensity_guess_implicit_cg = partial(
    joint_logdensity, gfunc=partial(guess_implicit_cg, target_function=dxdt)
)
