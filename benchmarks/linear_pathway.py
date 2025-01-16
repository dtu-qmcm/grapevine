"""An example comparing GrapeNUTS and NUTS on a representative problem.

The problem is a steady kinetic model of a linear pathway with this structure:

    Aext <-r1-> Aint <-r2-> Bint <-r3-> Bext

Reactions r1 and r3 behave according to the law of mass action, and reaction r2 according to the Michaelis Menten rate law. We assume we have measurements of Aint and Bint, as well as plenty of information about all the kinetic parameters and boundary conditions, and that the pathway is in a steady state, so that the concentrations c_m1_int and c_m2_int are not changing.

To formulate this situation as a statistical modelling problem, there are two functions `rmm` and `ma` that specify rate laws, and another function `fn` that specifies a steady state problem, i.e. finding values for c_m1_int and c_m2_int that put the system in a steady state.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by first choosing some true parameter values (see dictionary `TRUE_PARAMS`), and then simulating some measurements of c_m1_int and c_m2_int using these parameters: see function `simulate` for how this works. Then the log posterior is sampled using NUTS and GrapeNUTS, and the times are printed.

"""

import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import arviz as az
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import optimistix as optx
import polars as pl
from blackjax import nuts
from blackjax import window_adaptation as nuts_window_adaptation
from blackjax.util import run_inference_algorithm
from jax.scipy.stats import norm

from grapevine import run_grapenuts, get_idata

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
SD = 0.05
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "linear_pathway.csv"
TRUE_PARAMS = OrderedDict(
    log_km=jnp.array([2.0, 3.0]),
    log_vmax=jnp.array(0.0),
    log_keq=jnp.array([1.0, 1.0, 1.0]),
    log_kf=jnp.array([1.0, -1.0]),
    log_conc_ext=jnp.array([1.0, 0.0]),
)
DEFAULT_GUESS = jnp.array([0.01, 0.01])
N_WARMUP = 1000
N_SAMPLE = 1000
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.95

ode_solver = diffrax.Tsit5()
steady_state_cond = diffrax.steady_state_event()
steady_state_event = diffrax.Event(steady_state_cond)
adjoint = diffrax.ImplicitAdjoint(
    linear_solver=lineax.AutoLinearSolver(well_posed=False)
)
controller = diffrax.PIDController(pcoeff=0.1, icoeff=0.3, rtol=1e-9, atol=1e-9)


@eqx.filter_jit
def rmm(s, p, km_s, km_p, vmax, k_eq):
    """Reversible Michaelis Menten rate law"""
    num = vmax / km_s * (s - p / k_eq)
    denom = 1 + s / km_s + p / km_p
    return num / denom


@eqx.filter_jit
def ma(s, p, kf, keq):
    """Mass action rate law"""
    return kf * (s - p / keq)


@eqx.filter_jit
def fn(y, args):
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


solver = optx.Newton(rtol=1e-9, atol=1e-9)


@eqx.filter_jit
def joint_logdensity_grapenuts(params, obs, guess):
    sol = optx.root_find(fn, solver, guess, args=params)
    log_km, log_vmax, log_keq, log_kf, log_conc_ext = params.values()
    log_prior = jnp.sum(
        norm.logpdf(log_km, loc=TRUE_PARAMS["log_km"], scale=0.1).sum()
        + norm.logpdf(log_vmax, loc=TRUE_PARAMS["log_vmax"], scale=0.1).sum()
        + norm.logpdf(log_keq, loc=TRUE_PARAMS["log_keq"], scale=0.1).sum()
        + norm.logpdf(log_kf, loc=TRUE_PARAMS["log_kf"], scale=0.1).sum()
        + norm.logpdf(log_conc_ext, loc=TRUE_PARAMS["log_conc_ext"], scale=0.1).sum()
    )
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(sol.value), scale=jnp.full(obs.shape, SD)
    ).sum()
    return log_prior + log_likelihood, sol.value


@eqx.filter_jit
def joint_logdensity_grapenuts_jac(params, obs, guess):
    jac = jax.jacfwd(fn, argnums=0)(guess, params)
    inv_jac = jnp.linalg.inv(jac)

    def f_aux(t, x, args):
        inv_jac, params = args
        return -inv_jac @ fn(x, params) * jnp.log(0.2) / jnp.log(0.8) / (1 - t)

    term = diffrax.ODETerm(f_aux)

    sol = diffrax.diffeqsolve(
        terms=term,
        solver=ode_solver,
        t0=0.0,
        t1=0.99999,
        dt0=0.01,
        y0=guess,
        max_steps=None,
        args=(inv_jac, params),
        stepsize_controller=controller,
        adjoint=adjoint,
    )
    if sol.ys is None:
        raise ValueError("No steady state found!")
    log_km, log_vmax, log_keq, log_kf, log_conc_ext = params.values()
    log_prior = jnp.sum(
        norm.logpdf(log_km, loc=TRUE_PARAMS["log_km"], scale=0.1).sum()
        + norm.logpdf(log_vmax, loc=TRUE_PARAMS["log_vmax"], scale=0.1).sum()
        + norm.logpdf(log_keq, loc=TRUE_PARAMS["log_keq"], scale=0.1).sum()
        + norm.logpdf(log_kf, loc=TRUE_PARAMS["log_kf"], scale=0.1).sum()
        + norm.logpdf(log_conc_ext, loc=TRUE_PARAMS["log_conc_ext"], scale=0.1).sum()
    )
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(sol.ys[0]), scale=jnp.full(obs.shape, SD)
    ).sum()
    return log_prior + log_likelihood, sol.ys[0]


@eqx.filter_jit
def joint_logdensity_nuts(params, obs):
    ld, _ = joint_logdensity_grapenuts(params, obs, DEFAULT_GUESS)
    return ld


@eqx.filter_jit
def simulate(key, params, guess):
    sol = optx.root_find(fn, solver, guess, args=params)
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * SD
    )


def time_grapenuts_run(key, posterior_logdensity, true_params, default_guess):
    run_fn = partial(
        run_grapenuts,
        logdensity_fn=posterior_logdensity,
        rng_key=key,
        init_parameters=true_params,
        default_guess=default_guess,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLE,
        initial_step_size=INIT_STEPSIZE,
        max_num_doublings=MAX_TREEDEPTH,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=TARGET_ACCEPT,
        progress_bar=False,
    )
    _ = run_fn()  # dummy run for jitting
    start = time.time()
    out = run_fn()
    end = time.time()
    runtime = end - start
    idata = get_idata(*out)
    neff = az.ess(idata.sample_stats)["energy"].item()  # type: ignore
    return {
        "algorithm": "grapeNUTS",
        "time": runtime,
        "neff": neff,
    }


def time_nuts_run(key, posterior_logdensity, true_params):
    key_warmup, key_sampling = jax.random.split(key)
    warmup = nuts_window_adaptation(
        nuts,
        posterior_logdensity,
        progress_bar=False,
        initial_step_size=INIT_STEPSIZE,
        max_num_doublings=MAX_TREEDEPTH,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=TARGET_ACCEPT,
    )

    def run_fn():
        (initial_state, tuned_parameters), _ = warmup.run(
            key_warmup,
            true_params,
            num_steps=N_WARMUP,  #  type: ignore
        )
        kernel = nuts(posterior_logdensity, **tuned_parameters)
        (_, out) = run_inference_algorithm(
            key_sampling,
            kernel,
            N_SAMPLE,
            initial_state,
        )
        return out

    _ = run_fn()  # dummy run for jitting
    start = time.time()
    out = run_fn()
    end = time.time()
    runtime = end - start
    idata = get_idata(*out)
    neff = az.ess(idata.sample_stats)["energy"].item()  # type: ignore
    return {
        "algorithm": "NUTS",
        "time": runtime,
        "neff": neff,
    }


def run_single_comparison(
    key: jax.Array, true_params: dict
) -> tuple[jax.Array, dict, dict, dict]:
    key, sim_key = jax.random.split(key)
    key, grapenuts_key = jax.random.split(key)
    key, grapenuts_key_jac = jax.random.split(key)
    key, nuts_key = jax.random.split(key)
    default_guess = DEFAULT_GUESS
    # simulate
    _, sim = simulate(sim_key, true_params, default_guess)
    # posteriors
    posterior_logdensity_gn = partial(joint_logdensity_grapenuts, obs=sim)
    posterior_logdensity_gn_jac = partial(joint_logdensity_grapenuts_jac, obs=sim)
    posterior_logdensity_nuts = partial(joint_logdensity_nuts, obs=sim)
    # results
    result_gn = time_grapenuts_run(
        grapenuts_key,
        posterior_logdensity_gn,
        true_params,
        default_guess,
    )
    result_gn_jac = time_grapenuts_run(
        grapenuts_key_jac,
        posterior_logdensity_gn_jac,
        true_params,
        default_guess,
    )
    result_nuts = time_nuts_run(
        nuts_key,
        posterior_logdensity_nuts,
        true_params,
    )
    return (
        key,
        result_gn,
        result_gn_jac,
        result_nuts,
    )


def run_comparison(n_test: int):
    key = jax.random.key(SEED)
    results = []
    for i in range(n_test):
        key, result_gn, result_gn_jac, result_nuts = run_single_comparison(
            key,
            TRUE_PARAMS,
        )
        result_gn["repeat"] = i
        result_gn_jac["repeat"] = i
        result_nuts["repeat"] = i
        results += [
            result_gn,
            result_gn_jac,
            result_nuts,
        ]
    return pl.from_records(results).with_columns(
        (pl.col("neff") / pl.col("time")).alias("neff/s")
    )


def main():
    results = run_comparison(n_test=1)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    print("Mean results:")
    results.write_csv(CSV_OUTPUT_FILE)
    print(results)


if __name__ == "__main__":
    main()
