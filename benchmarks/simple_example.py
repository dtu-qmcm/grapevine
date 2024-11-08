"""An example comparing GrapeNUTS and NUTS on a simple problem.

The problem is taken from the Stan documentation: <https://mc-stan.org/docs/stan-users-guide/algebraic-equations.html#coding-an-algebraic-system>

To formulate this situation as a statistical modelling problem, there is a function `fn` that takes in a state (`y`) and some parameters (`args`) and returns the quantities that should be zero.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by first choosing some true parameter values (see dictionary `TRUE_PARAMS`), and then simulating some measurements of c_m1_int and c_m2_int using these parameters: see function `simulate` for how this works. Then the log posterior is sampled using NUTS and GrapeNUTS, and the times are printed.

"""

import timeit
from collections import OrderedDict
from functools import partial
import logging
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

SEED = 1234
SD = 0.05
HERE = Path(__file__).parent
STAN_FILE = HERE / "simple_example.stan"
CSV_OUTPUT_FILE = HERE / "simple_example.csv"
TRUE_PARAMS = OrderedDict(theta=jnp.array([3.0, 6.0]))
DEFAULT_GUESS = jnp.array([1.0, 1.0])
N_WARMUP = 1000
N_SAMPLE = 1000
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.95


@eqx.filter_jit
def fn(y, args):
    y1, y2 = y
    theta1, theta2 = args
    return jnp.array([y1 - theta1, y1 * y2 - theta2])


solver = optx.Newton(rtol=1e-9, atol=1e-9)


@eqx.filter_jit
def joint_logdensity_grapenuts(params, obs, guess):
    theta = params["theta"]
    sol = optx.root_find(fn, solver, guess, args=theta)
    log_prior = jnp.sum(norm.logpdf(theta, loc=TRUE_PARAMS["theta"], scale=0.1).sum())
    log_likelihood = norm.logpdf(obs, loc=sol.value, scale=SD).sum()
    return log_prior + log_likelihood, sol.value


@eqx.filter_jit
def joint_logdensity_nuts(params, obs):
    ld, _ = joint_logdensity_grapenuts(params, obs, DEFAULT_GUESS)
    return ld


@eqx.filter_jit
def simulate(key, params, guess):
    theta = params["theta"]
    sol = optx.root_find(fn, solver, guess, args=theta)
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * SD
    )


def fit_stan(sim):
    data = {
        "prior_mean_theta": TRUE_PARAMS["theta"].tolist(),
        "prior_sd_theta": [0.1, 0.1],
        "y": sim,
        "sd": SD,
        "y_guess": DEFAULT_GUESS.tolist(),
        "scaling_step": 1e-3,
        "ftol": 1e-9,
        "max_steps": 256,
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
    )
    return mcmc


def main():
    # disable cmdstanpy logging
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True
    # keys
    key = jax.random.key(SEED)
    key, sim_key = jax.random.split(key)
    key, grapenuts_key = jax.random.split(key)
    key, nuts_key_warmup = jax.random.split(key)
    key, nuts_key_sampling = jax.random.split(key)
    # simulate some data
    _, sim = simulate(sim_key, TRUE_PARAMS, DEFAULT_GUESS)
    # specify posteriors
    posterior_logdensity_gn = partial(joint_logdensity_grapenuts, obs=sim)
    posterior_logdensity_nuts = partial(joint_logdensity_nuts, obs=sim)

    def run_grapenuts_example():
        return run_grapenuts(
            posterior_logdensity_gn,
            grapenuts_key,
            init_parameters=TRUE_PARAMS,
            default_guess=DEFAULT_GUESS,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLE,
            initial_step_size=INIT_STEPSIZE,
            max_num_doublings=MAX_TREEDEPTH,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=TARGET_ACCEPT,
            progress_bar=False,
        )

    def run_nuts_example():
        warmup = nuts_window_adaptation(
            nuts,
            posterior_logdensity_nuts,
            progress_bar=False,
            initial_step_size=INIT_STEPSIZE,
            max_num_doublings=MAX_TREEDEPTH,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=TARGET_ACCEPT,
        )
        (initial_state, tuned_parameters), _ = warmup.run(
            nuts_key_warmup,
            TRUE_PARAMS,
            num_steps=N_WARMUP,  #  type: ignore
        )
        kernel = nuts(posterior_logdensity_nuts, **tuned_parameters)
        return run_inference_algorithm(
            nuts_key_sampling,
            kernel,
            N_SAMPLE,
            initial_state,
        )

    def run_stan_example():
        return fit_stan(sim.tolist())

    results = []

    _ = run_stan_example()  # run once for jitting
    _ = run_grapenuts_example()  # run once for jitting
    _ = run_nuts_example()  # run once for jitting
    for _ in range(10):
        _ = run_grapenuts_example()  # one more time to be safe!
        time_gn, out_gn = timeit.timeit(run_grapenuts_example, number=1)
        idata_gn = get_idata(*out_gn)
        neff_gn = az.ess(idata_gn.sample_stats)["energy"].item()
        results += [{"algorithm": "grapeNUTS", "time": time_gn, "neff": neff_gn}]
        time_nuts, (_, out_nuts) = timeit.timeit(run_nuts_example, number=1)
        idata_nuts = get_idata(*out_nuts)
        neff_nuts = az.ess(idata_nuts.sample_stats)["energy"].item()
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
