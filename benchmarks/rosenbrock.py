"""Compare GrapeNUTS and NUTS performance."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Any

import jax
import optimistix as optx
import polars as pl
from jax import numpy as jnp
from jax.scipy.stats import norm

from grapevine.benchmarking import (
    StyblinskiTang,
    run_benchmark,
    Rosenbrock,
    with_step_count,
)
from grapevine.heuristics import (
    guess_default,
    guess_implicit,
    guess_previous,
)

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
N_TESTS_PER_CASE = 6
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "test_functions.csv"

RUN_GRAPENUTS_KWARGS = dict(
    num_warmup=3000,
    num_samples=3000,
    initial_step_size=0.001,
    max_num_doublings=10,
    is_mass_matrix_diagonal=False,
    target_acceptance_rate=0.99,
)


@dataclass
class Case:
    name: str
    f: Callable
    max_steps: int
    solver: optx.Newton
    prior_sd: float
    error_sd: float
    default_guess: jax.Array
    baseline_params: dict


CASES = [
    Case(
        name="StyblinskiTang3d",
        f=StyblinskiTang(n_dimensions=3),
        max_steps=2000,
        solver=optx.Newton(rtol=1e-5, atol=1e-5),
        error_sd=0.05,
        prior_sd=0.3,
        default_guess=jnp.full((3,), -2.903),
        baseline_params=OrderedDict(theta=jnp.full((3,), 0.0)),
    ),
    Case(
        name="Rosenbrock3d",
        f=Rosenbrock(n_dimensions=3),
        max_steps=800,
        solver=optx.Newton(rtol=1e-5, atol=1e-5),
        error_sd=0.05,
        prior_sd=0.3,
        default_guess=jnp.full((3,), 1.0),
        baseline_params=OrderedDict(theta=jnp.full((3,), 0.0)),
    ),
]


def callable_name(any_callable: Callable[..., Any]) -> str:
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    try:
        return any_callable.__name__
    except AttributeError:
        return str(any_callable)


def parameterise(f):
    def f_parameterised(x, args):
        x_plus_theta = x + args["theta"]
        return jax.grad(f)(x_plus_theta)

    return f_parameterised


def get_solve_func(f, solver, max_steps):
    def solve_func(guess, params):
        sol = optx.root_find(
            parameterise(f),
            solver,
            guess,
            args=params,
            max_steps=max_steps,
        )
        return sol.value, jnp.array(sol.stats["num_steps"])

    return solve_func


def joint_logdensity(
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
    results_list = []
    for case in CASES:
        print(f"Benchmarking {case.name}...")
        solve = get_solve_func(
            f=case.f,
            solver=case.solver,
            max_steps=case.max_steps,
        )
        simulate = partial(
            simulate_func, solve_func=solve, error_sd=case.error_sd
        )

        guess_fns = {
            "guess_implicit": with_step_count(
                partial(guess_implicit, target_function=parameterise(case.f))
            ),
            "guess_static": with_step_count(
                partial(guess_default, default_guess=case.default_guess)
            ),
            "guess_previous": with_step_count(guess_previous),
        }
        case_results = run_benchmark(
            random_seed=SEED,
            joint_logdensity=partial(
                joint_logdensity,
                solve_func=solve,
                prior_sd=case.prior_sd,
                error_sd=case.error_sd,
            ),
            guess_fns=guess_fns,
            baseline_params=case.baseline_params,
            param_sd=case.prior_sd,
            n_test=N_TESTS_PER_CASE,
            run_grapenuts_kwargs=RUN_GRAPENUTS_KWARGS,
            sim_func=simulate,
            default_guess=(case.default_guess, jnp.array(0)),
        )
        case_results = case_results.with_columns(case=pl.lit(case.name))
        results_list.append(case_results)
        print("Runtimes:")
        print(case_results.pivot("heuristic", index="rep", values="time"))
        print("Effective sample sizes:")
        print(case_results.pivot("heuristic", index="rep", values="neff"))
        print("Newton steps:")
        print(
            case_results.pivot(
                "heuristic", index="rep", values="n_newton_steps"
            )
        )
    results = pl.concat(results_list)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    results.write_csv(CSV_OUTPUT_FILE)


if __name__ == "__main__":
    main()
