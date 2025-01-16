"""Compare GrapeNUTS and NUTS on a standard minimisation problem."""

import time
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
from jax.scipy.stats import norm
from jaxtyping import Scalar

from grapevine import run_grapenuts, get_idata

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
SD = 0.05
DIM_LOW = 3
DIM_HIGH = 7
N_TESTS_PER_DIM = 8
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "rosenbrock.csv"
N_WARMUP = 1000
N_SAMPLE = 1000
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.99


def rosenbrock(x: jax.Array, args: jax.Array) -> Scalar:
    xt = x * args
    return (100.0 * (xt[1:] - xt[:-1] ** 2.0) ** 2.0 + (1 - xt[:-1]) ** 2.0).sum()


solver = optx.BFGS(rtol=1e-9, atol=1e-9)


@eqx.filter_jit
def joint_logdensity_grapenuts(params, obs, guess, true_params):
    theta = params["theta"]
    sol = optx.minimise(rosenbrock, solver, guess, args=theta)
    log_prior = jnp.sum(norm.logpdf(theta, loc=true_params["theta"], scale=0.1).sum())
    log_likelihood = norm.logpdf(obs, loc=sol.value, scale=SD).sum()
    return log_prior + log_likelihood, sol.value


@eqx.filter_jit
def joint_logdensity_nuts(params, obs, true_params, default_guess):
    ld, _ = joint_logdensity_grapenuts(
        params,
        obs,
        default_guess,
        true_params=true_params,
    )
    return ld


@eqx.filter_jit
def simulate(
    key: jax.Array, params: dict, guess: jax.Array
) -> tuple[jax.Array, jax.Array]:
    theta = params["theta"]
    sol = optx.minimise(rosenbrock, solver, guess, args=theta)
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
        "dim": len(default_guess),
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
        "dim": len(true_params["theta"]),
        "algorithm": "NUTS",
        "time": runtime,
        "neff": neff,
    }


def run_single_comparison(
    key: jax.Array, true_params: dict, dim: int
) -> tuple[jax.Array, dict, dict]:
    key, sim_key = jax.random.split(key)
    key, grapenuts_key = jax.random.split(key)
    key, nuts_key = jax.random.split(key)
    default_guess = jnp.full((dim,), 1.0)
    # simulate
    _, sim = simulate(sim_key, true_params, default_guess)
    # posteriors
    posterior_logdensity_gn = partial(
        joint_logdensity_grapenuts,
        obs=sim,
        true_params=true_params,
    )
    posterior_logdensity_nuts = partial(
        joint_logdensity_nuts,
        obs=sim,
        true_params=true_params,
        default_guess=default_guess,
    )
    result_gn = time_grapenuts_run(
        grapenuts_key,
        posterior_logdensity_gn,
        true_params,
        default_guess,
    )
    result_nuts = time_nuts_run(
        nuts_key,
        posterior_logdensity_nuts,
        true_params,
    )
    return key, result_gn, result_nuts


def run_comparison(dim: int, n_test: int):
    key = jax.random.key(SEED)
    results = []
    for i in range(n_test):
        key, truth_key = jax.random.split(key)
        true_theta = jax.random.uniform(truth_key, (dim,), minval=0.85, maxval=1.15)
        true_params = {"theta": true_theta}
        key, result_gn, result_nuts = run_single_comparison(
            key,
            true_params,
            dim,
        )
        result_gn["repeat"] = i
        result_nuts["repeat"] = i
        results += [result_gn, result_nuts]
    return pl.from_records(results).with_columns(
        (pl.col("neff") / pl.col("time")).alias("neff/s")
    )


def main():
    results_list = []
    for dim in range(DIM_LOW, DIM_HIGH + 1):
        print(f"Benchmarking Rosenbrock function with size {dim}...")
        results_list.append(run_comparison(dim=dim, n_test=N_TESTS_PER_DIM))
    results = pl.concat(results_list)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    print("Mean results:")
    results.write_csv(CSV_OUTPUT_FILE)
    print(results.group_by(["algorithm", "dim"]).mean().sort(["dim", "neff/s"]))


if __name__ == "__main__":
    main()
