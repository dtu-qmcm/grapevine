"""Compare GrapeNUTS and NUTS performance."""

from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import polars as pl
from jax.scipy.stats import norm

from grapevine import run_grapenuts
from grapevine.example_functions import rosenbrock
from grapevine.util import run_nuts, time_run

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
ERROR_SD = 0.005
PRIOR_SD = 0.3
DIMS = (3, 4, 5, 6, 7)
N_TESTS_PER_DIM = 6
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "rosenbrock.csv"
N_WARMUP = 3000
N_SAMPLE = 3000
INIT_STEPSIZE = 0.001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.99


solver = optx.BFGS(rtol=1e-9, atol=1e-9)


def joint_logdensity_grapenuts(params, obs, guess, true_params):
    theta = params["theta"]
    sol = optx.minimise(rosenbrock, solver, guess, args=theta, max_steps=int(1e4))
    log_prior = norm.logpdf(theta, loc=true_params["theta"], scale=PRIOR_SD).sum()

    log_likelihood = norm.logpdf(obs, loc=sol.value, scale=ERROR_SD).sum()
    return log_prior + log_likelihood, sol.value


def joint_logdensity_nuts(params, obs, true_params, default_guess):
    ld, _ = joint_logdensity_grapenuts(
        params,
        obs,
        default_guess,
        true_params=true_params,
    )
    return ld


def simulate(
    key: jax.Array, params: dict, guess: jax.Array
) -> tuple[jax.Array, jax.Array]:
    theta = params["theta"]
    sol = optx.minimise(rosenbrock, solver, guess, args=theta)
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * ERROR_SD
    )


def run_single_comparison(key: jax.Array, true_params: dict, dim: int) -> dict:
    sim_key, grapenuts_key, nuts_key = jax.random.split(key, 3)
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
    run_fn_gn = eqx.filter_jit(
        partial(
            run_grapenuts,
            logdensity_fn=posterior_logdensity_gn,
            rng_key=grapenuts_key,
            init_parameters=true_params,
            default_guess=default_guess,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLE,
            initial_step_size=INIT_STEPSIZE,
            max_num_doublings=MAX_TREEDEPTH,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=TARGET_ACCEPT,
            progress_bar=False,
        )
    )
    run_fn_nuts = eqx.filter_jit(
        partial(
            run_nuts,
            logdensity_fn=posterior_logdensity_nuts,
            rng_key=nuts_key,
            init_parameters=true_params,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLE,
            initial_step_size=INIT_STEPSIZE,
            max_num_doublings=MAX_TREEDEPTH,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=TARGET_ACCEPT,
            progress_bar=False,
        )
    )
    # results
    result_gn = time_run(run_fn_gn)
    result_nuts = time_run(run_fn_nuts)
    perf_gn = result_gn["neff"] / result_gn["time"]
    perf_nuts = result_nuts["neff"] / result_nuts["time"]
    perf_ratio = perf_gn / perf_nuts
    return {
        "neff_n": result_nuts["neff"],
        "neff_gn": result_gn["neff"],
        "time_n": result_nuts["time"],
        "time_gn": result_gn["time"],
        "perf_n": perf_nuts,
        "perf_gn": perf_gn,
        "perf_ratio": perf_ratio,
    }


def run_comparison(dim: int, n_test: int, key: jax.Array):
    results = []
    keys = jax.random.split(key, n_test)
    for i, keyi in enumerate(keys):
        rng_key, compare_key = jax.random.split(keyi)
        true_theta = jax.random.normal(key=rng_key, shape=(dim,)) * PRIOR_SD
        true_params = {"theta": true_theta}
        result = run_single_comparison(compare_key, true_params, dim)
        result["rep"] = i
        result["dim"] = dim
        results.append(result)
    return pl.from_records(results)


def main():
    results_list = []
    key = jax.random.key(SEED)
    dim_keys = jax.random.split(key, len(DIMS))
    for dim, dim_key in zip(DIMS, dim_keys):
        print(f"Benchmarking Rosenbrock function with size {dim}...")
        dim_results = run_comparison(
            dim=dim,
            n_test=N_TESTS_PER_DIM,
            key=dim_key,
        )
        results_list.append(dim_results)
        print(dim_results)
    results = pl.concat(results_list)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    print("Mean results:")
    results.write_csv(CSV_OUTPUT_FILE)
    print(results.group_by(["dim"]).mean().sort(["dim", "perf_ratio"]))


if __name__ == "__main__":
    main()
