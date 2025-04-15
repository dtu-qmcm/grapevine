"""An example comparing GrapeNUTS and NUTS on a representative problem.

The problem is a steady kinetic model of a linear pathway with this structure:

    Aext <-r1-> Aint <-r2-> Bint <-r3-> Bext

Reactions r1 and r3 behave according to the law of mass action, and reaction r2 according to the Michaelis Menten rate law. We assume we have measurements of Aint and Bint, as well as plenty of information about all the kinetic parameters and boundary conditions, and that the pathway is in a steady state, so that the concentrations c_m1_int and c_m2_int are not changing.

To formulate this situation as a statistical modelling problem, there is a function `linear_pathway_steady_state` that specifies a steady state problem, i.e. finding values for c_m1_int and c_m2_int that put the system in a steady state.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by repeatedly choosing some true parameter values at random by perturbing the dictionary `TRUE_PARAMS`, then using these parameters to simulate some measurements of c_m1_int and c_m2_int. Then the log posterior is sampled using NUTS and GrapeNUTS, and the relative ess/second valeus are printed.

"""

from collections import OrderedDict
from functools import partial
import operator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import polars as pl
from jax.scipy.stats import norm

from grapevine.example_functions import linear_pathway_steady_state as sv
from grapevine.util import run_grapenuts, run_nuts, time_run

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
ERROR_SD = 0.05
PARAM_SD = 0.02
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "linear.csv"
TRUE_PARAMS = OrderedDict(
    log_km=jnp.array([2.0, 2.0]),
    log_vmax=jnp.array(3.0),
    log_keq=jnp.array([1.0, 1.0, 1.0]),
    log_kf=jnp.array([1.0, -1.0]),
    log_conc_ext=jnp.array([1.0, 0.0]),
)
DEFAULT_GUESS = (jnp.array([0.1, 0.1]), TRUE_PARAMS, 0)
DEFAULT_GUESS2 = (
    jnp.array([0.4, 0.4]),
    OrderedDict({k: v * 0.2 for k, v in TRUE_PARAMS.items()}),
    0,
)
N_WARMUP = 2000
N_SAMPLE = 1000
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.9
N_TEST = 6

solver = optx.Newton(rtol=1e-5, atol=1e-5)


@jax.jit
def smart_guess_no_jac(old, params):
    old_x, old_p, _ = old
    delta_p = jax.tree.map(lambda old, new: new - old, old_p, params)
    _, jvpp = jax.jvp(lambda p: sv(old_x, p), (old_p,), (delta_p,))

    def matvec(v):
        "Compute Jx @ v for any vector v"
        return jax.jvp(lambda x: sv(x, old_p), (old_x,), (v,))[1]

    dx = -jax.scipy.sparse.linalg.cg(matvec, jvpp)[0]
    return old_x + dx


@jax.jit
def smart_guess(old, params):
    old_x, old_p, _ = old
    delta_p = jax.tree.map(lambda old, new: new - old, old_p, params)
    _, jvpp = jax.jvp(lambda p: sv(old_x, p), (old_p,), (delta_p,))
    jacx = jax.jacfwd(sv, argnums=0)(old_x, old_p)
    u = -(jnp.linalg.inv(jacx))
    return old_x + u @ jvpp


def dumb_guess(guess, params):
    return guess[0]


def very_dumb_guess(guess, params):
    return DEFAULT_GUESS[0]


def joint_logdensity(params, obs, guess, gfunc):
    is_default = guess[0][0] == 0.1
    _, _, previous_steps = guess
    guess = jax.lax.cond(is_default, dumb_guess, gfunc, guess, params)

    def solve(params):
        sol = optx.root_find(
            sv,
            solver,
            guess,
            args=params,
            max_steps=int(1e5),
        )
        return sol.value, sol.stats["num_steps"]

    steady, steps_i = solve(params)
    log_km, log_vmax, log_keq, log_kf, log_conc_ext = params.values()
    log_prior = jnp.sum(
        norm.logpdf(log_km, loc=TRUE_PARAMS["log_km"], scale=0.1).sum()
        + norm.logpdf(log_vmax, loc=TRUE_PARAMS["log_vmax"], scale=0.1).sum()
        + norm.logpdf(log_keq, loc=TRUE_PARAMS["log_keq"], scale=0.1).sum()
        + norm.logpdf(log_kf, loc=TRUE_PARAMS["log_kf"], scale=0.1).sum()
        + norm.logpdf(log_conc_ext, loc=TRUE_PARAMS["log_conc_ext"], scale=0.1).sum()
    )
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(steady), scale=jnp.full(obs.shape, ERROR_SD)
    ).sum()
    steps = previous_steps + steps_i
    return log_prior + log_likelihood, (steady, params, steps)


def simulate(key, params, guess):
    guess, _, _ = guess
    sol = optx.root_find(
        sv,
        solver,
        guess,
        args=params,
    )
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * ERROR_SD
    )


def compare_single(key: jax.Array, params) -> dict:
    sim_key, grapenuts_key, nuts_key = jax.random.split(key, 3)
    # simulate
    _, sim = simulate(sim_key, params, DEFAULT_GUESS)
    # posteriors
    posterior_logdensity_funcs = (
        partial(joint_logdensity, obs=sim, gfunc=gfunc)
        for gfunc in (
            very_dumb_guess,
            dumb_guess,
            smart_guess,
            smart_guess_no_jac,
        )
    )
    run_fns = (
        eqx.filter_jit(
            partial(
                run_grapenuts,
                logdensity_fn=pld,
                rng_key=grapenuts_key,
                init_parameters=params,
                default_guess=DEFAULT_GUESS,
                num_warmup=N_WARMUP,
                num_samples=N_SAMPLE,
                initial_step_size=INIT_STEPSIZE,
                max_num_doublings=MAX_TREEDEPTH,
                is_mass_matrix_diagonal=False,
                target_acceptance_rate=TARGET_ACCEPT,
                progress_bar=False,
            )
        )
        for pld in posterior_logdensity_funcs
    )
    # results
    results = [time_run(run_fn) for run_fn in run_fns]
    result_nuts = results[0]
    perf = [result["neff"] / result["n_newton_steps"] for result in results]
    _, pr_gn_basic, pr_gn_smart, pr_gn_nojac = [perf_i / perf[0] for perf_i in perf]
    return {
        "neff_nuts": result_nuts["neff"],
        "performance_ratio_gn_basic": pr_gn_basic,
        "performance_ratio_gn_smart": pr_gn_smart,
        "performance_ratio_gn_nojac": pr_gn_nojac,
    }


def generate_random_params(key, params_in, sd):
    out = OrderedDict()
    for k, v in params_in.items():
        key_iter, key = jax.random.split(key)
        out[k] = v + jax.random.normal(key_iter, v.shape) * sd
    return out


def run_comparison(n_test: int):
    key = jax.random.key(SEED)
    keys = jax.random.split(key, n_test)
    results = []
    for i, keyi in enumerate(keys):
        compare_key, param_key = jax.random.split(keyi)
        params = generate_random_params(param_key, TRUE_PARAMS, PARAM_SD)
        result = compare_single(compare_key, params)
        result["rep"] = i
        results.append(result)
    return pl.from_records(results)


def main():
    results = run_comparison(n_test=N_TEST)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    results.write_csv(CSV_OUTPUT_FILE)
    print("Results:")
    print(results)


if __name__ == "__main__":
    main()
