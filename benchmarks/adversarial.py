"""Compare guessing heuristics on adversarial optimisation problem."""

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Any

import jax
import optimistix as optx
import polars as pl
from jax import numpy as jnp
from jax.scipy.stats import norm

from grapevine.benchmarking import run_benchmark, with_step_count
from grapevine.heuristics import (
    guess_default,
    guess_implicit,
    guess_previous,
    guess_implicit_cg,
)

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
N_TESTS_PER_CASE = 20
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "adversarial.csv"

RUN_GRAPENUTS_KWARGS = dict(
    num_warmup=500,
    num_samples=500,
    initial_step_size=0.001,
    max_num_doublings=10,
    is_mass_matrix_diagonal=False,
)


def f_parameterised(x, args):
    # Highly oscillatory pseudorandom-like root target with non-linear root-finding problem.
    # The root jumps unpredictably between Hamiltonian trajectory steps.
    # guess_implicit will perform worse here because of the massive gradients.
    theta = args["theta"]
    k = jnp.array(1e6)
    return x**3 + x - jnp.sin(k * theta) * jnp.cos(k * theta)


MAX_STEPS = 1000
SOLVER = optx.Newton(rtol=1e-8, atol=1e-8)
ERROR_SD = 0.1
PRIOR_SD = 0.1
DEFAULT_GUESS = jnp.full((8,), 0.0)
BASELINE_PARAMS = OrderedDict(theta=jnp.full((8,), 0.0))


def callable_name(any_callable: Callable[..., Any]) -> str:
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    try:
        return any_callable.__name__
    except AttributeError:
        return str(any_callable)


def get_solve_func(param_f, solver, max_steps):
    def solve_func(guess, params):
        sol = optx.root_find(
            param_f,
            solver,
            guess,
            args=params,
            max_steps=max_steps,
        )
        return sol.value, jnp.array(sol.stats["num_steps"])

    return solve_func


def joint_logdensity_independent(
    params,
    obs,
    guess,
    solve_func,
    prior_sd,
    error_sd,
):
    guess, previous_steps = guess
    solution, steps_here = solve_func(guess, params)
    log_prior = norm.logpdf(
        params["theta"], loc=jnp.zeros(params["theta"].shape), scale=prior_sd
    ).sum()
    # Make log probability independent of the root-finding solution
    # so the HMC step size is not forced to be extremely small.
    log_likelihood = 0.0
    steps = previous_steps + steps_here
    return log_prior + log_likelihood, (solution, steps)


def joint_logdensity_dependent(
    params,
    obs,
    guess,
    solve_func,
    prior_sd,
    error_sd,
):
    guess, previous_steps = guess
    solution, steps_here = solve_func(guess, params)
    log_prior = norm.logpdf(
        params["theta"], loc=jnp.zeros(params["theta"].shape), scale=prior_sd
    ).sum()
    log_likelihood = norm.logpdf(obs, loc=solution, scale=error_sd).sum()
    steps = previous_steps + steps_here
    return log_prior + log_likelihood, (solution, steps)


def simulate_func(
    key: jax.Array, params: dict, guess: jax.Array, error_sd, solve_func
) -> tuple[jax.Array, jax.Array]:
    sol, _ = solve_func(guess, params)
    return sol, sol + jax.random.normal(key, shape=sol.shape) * error_sd


def main():
    solve = get_solve_func(
        param_f=f_parameterised,
        solver=SOLVER,
        max_steps=MAX_STEPS,
    )
    simulate = partial(simulate_func, solve_func=solve, error_sd=ERROR_SD)

    guess_fns = {
        "guess_implicit": with_step_count(
            partial(guess_implicit, target_function=f_parameterised)
        ),
        "guess_implicit_cg": with_step_count(
            partial(guess_implicit_cg, target_function=f_parameterised)
        ),
        "guess_static": with_step_count(
            partial(guess_default, default_guess=DEFAULT_GUESS)
        ),
        "guess_previous": with_step_count(guess_previous),
    }

    all_results = []

    for name, jld_func in [
        ("Adversarial-Independent", joint_logdensity_independent),
        ("Adversarial-Dependent", joint_logdensity_dependent),
    ]:
        print(f'Benchmarking case "{name}"...')

        case_results = run_benchmark(
            random_seed=SEED,
            joint_logdensity=partial(
                jld_func,
                solve_func=solve,
                prior_sd=PRIOR_SD,
                error_sd=ERROR_SD,
            ),
            guess_fns=guess_fns,
            baseline_params=BASELINE_PARAMS,
            param_sd=PRIOR_SD,
            n_test=N_TESTS_PER_CASE,
            run_grapenuts_kwargs=RUN_GRAPENUTS_KWARGS,
            sim_func=simulate,
            default_guess=(DEFAULT_GUESS, jnp.array(0)),
        )
        case_results = case_results.with_columns(case=pl.lit(name))
        all_results.append(case_results)

        print(f"\nResults for {name}:")
        print("Runtimes:")
        print(case_results.pivot("heuristic", index="rep", values="time"))
        print("Newton steps:")
        print(
            case_results.pivot(
                "heuristic", index="rep", values="n_newton_steps"
            )
        )
        print("-" * 50)

    final_results = pl.concat(all_results)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    final_results.write_csv(CSV_OUTPUT_FILE)


if __name__ == "__main__":
    main()
