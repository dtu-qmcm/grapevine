"""An example comparing GrapeNUTS and NUTS on a representative problem.

The problem is a steady kinetic model of a linear pathway with this structure:

    Aext <-r1-> Aint <-r2-> Bint <-r3-> Bext

Reactions r1 and r3 behave according to the law of mass action, and reaction r2 according to the Michaelis Menten rate law. We assume we have measurements of Aint and Bint, as well as plenty of information about all the kinetic parameters and boundary conditions, and that the pathway is in a steady state, so that the concentrations c_m1_int and c_m2_int are not changing.

To formulate this situation as a statistical modelling problem, there are two functions `rmm` and `ma` that specify rate laws, and another function `fn` that specifies a steady state problem, i.e. finding values for c_m1_int and c_m2_int that put the system in a steady state.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by first choosing some true parameter values (see dictionary `TRUE_PARAMS`), and then simulating some measurements of c_m1_int and c_m2_int using these parameters: see function `simulate` for how this works. Then the log posterior is sampled using NUTS and GrapeNUTS, and the times are printed.

"""

import logging
import timeit
from collections import OrderedDict
from functools import partial
from pathlib import Path

import arviz as az
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import polars as pl
from blackjax import nuts
from blackjax import window_adaptation as nuts_window_adaptation
from blackjax.util import run_inference_algorithm
from cmdstanpy import CmdStanModel
from jax.scipy.stats import norm

from grapevine import run_grapenuts, get_idata

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
SD = 0.05
HERE = Path(__file__).parent
STAN_FILE = HERE / "linear_pathway.stan"
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


# override timeit template:
# see https://stackoverflow.com/questions/24812253/how-can-i-capture-return-value-with-python-timeit-module
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


@eqx.filter_jit
def rmm(s, p, km_s, km_p, vmax, k_eq):
    """Reversible Michaelis Menten rate law"""
    num = vmax * (s - p / k_eq) / km_s
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
def joint_logdensity_nuts(params, obs):
    ld, _ = joint_logdensity_grapenuts(params, obs, DEFAULT_GUESS)
    return ld


@eqx.filter_jit
def simulate(key, params, guess):
    sol = optx.root_find(fn, solver, guess, args=params)
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * SD
    )


def fit_stan(sim):
    data = {
        "y": sim,
        "sd": SD,
        "prior_log_km": [TRUE_PARAMS["log_km"].tolist(), [0.1, 0.1]],
        "prior_log_vmax": [TRUE_PARAMS["log_vmax"].tolist(), 0.1],
        "prior_log_keq": [TRUE_PARAMS["log_keq"].tolist(), [0.1, 0.1, 0.1]],
        "prior_log_kf": [TRUE_PARAMS["log_kf"].tolist(), [0.1, 0.1]],
        "prior_log_cext": [TRUE_PARAMS["log_conc_ext"].tolist(), [0.1, 0.1]],
        "y_guess": DEFAULT_GUESS.tolist(),
        "scaling_step": 1e-9,
        "ftol": 1e-9,
        "max_steps": 1000,
    }
    model = CmdStanModel(stan_file=STAN_FILE)
    mcmc = model.sample(
        data=data,
        chains=1,
        inits={k: v.tolist() for k, v in TRUE_PARAMS.items()},
        iter_warmup=N_WARMUP,
        iter_sampling=N_SAMPLE,
        adapt_delta=TARGET_ACCEPT,
        step_size=INIT_STEPSIZE,
        max_treedepth=MAX_TREEDEPTH,
        show_progress=False,
        seed=SEED,
    )
    return mcmc


def main():
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True
    key = jax.random.key(SEED)
    key, sim_key = jax.random.split(key)
    _, sim = simulate(sim_key, TRUE_PARAMS, DEFAULT_GUESS)
    posterior_logdensity_gn = partial(joint_logdensity_grapenuts, obs=sim)
    posterior_logdensity_nuts = partial(joint_logdensity_nuts, obs=sim)
    key, grapenuts_key = jax.random.split(key)
    key, nuts_key_warmup = jax.random.split(key)
    key, nuts_key_sampling = jax.random.split(key)

    def run_grapenuts_example():
        return run_grapenuts(
            posterior_logdensity_gn,
            grapenuts_key,
            init_parameters=TRUE_PARAMS,
            default_guess=DEFAULT_GUESS,
            num_warmup=1000,
            num_samples=1000,
            initial_step_size=0.0001,
            max_num_doublings=10,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=0.95,
            progress_bar=False,
        )

    def run_nuts_example():
        warmup = nuts_window_adaptation(
            nuts,
            posterior_logdensity_nuts,
            progress_bar=False,
            initial_step_size=0.0001,
            max_num_doublings=10,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=0.95,
        )
        (initial_state, tuned_parameters), _ = warmup.run(
            nuts_key_warmup,
            TRUE_PARAMS,
            num_steps=1000,  #  type: ignore
        )
        kernel = nuts(posterior_logdensity_nuts, **tuned_parameters)
        return run_inference_algorithm(
            nuts_key_sampling,
            kernel,
            1000,
            initial_state,
        )

    def run_stan_example():
        return fit_stan(sim.tolist())

    results = []

    _ = run_stan_example()  # run once for jitting
    _ = run_grapenuts_example()  # run once for jitting
    _ = run_nuts_example()  # run once for jitting
    for _ in range(10):
        _ = run_nuts_example()  # one more time to be safe!
        time_nuts, (_, out_nuts) = timeit.timeit(run_nuts_example, number=1)
        idata_nuts = get_idata(*out_nuts)
        neff_nuts = az.ess(idata_nuts.sample_stats)["energy"].item()
        time_gn, out_gn = timeit.timeit(run_grapenuts_example, number=1)
        idata_gn = get_idata(*out_gn)
        neff_gn = az.ess(idata_gn.sample_stats)["energy"].item()
        results += [{"algorithm": "grapeNUTS", "time": time_gn, "neff": neff_gn}]
        results += [{"algorithm": "NUTS", "time": time_nuts, "neff": neff_nuts}]
        time_stan, mcmc = timeit.timeit(run_stan_example, number=1)
        idata_stan = az.from_cmdstanpy(mcmc)
        neff_stan = az.ess(idata_stan.sample_stats)["lp"].item()
        results += [{"algorithm": "Stan", "time": time_stan, "neff": neff_stan}]
    results_df = pl.from_records(results).with_columns(
        neff_per_s=pl.col("neff") / pl.col("time")
    )
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    print("Mean results:")
    results_df.write_csv(CSV_OUTPUT_FILE)
    print(results_df.group_by("algorithm").mean())


if __name__ == "__main__":
    main()
