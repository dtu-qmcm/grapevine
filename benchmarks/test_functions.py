"""Compare guessing heuristics on textbook optimisation problems.

For each case we construct a version of the optimisation test function f that takes in two arguments: x = x1, ..., xk and theta = theta1, ..., thetak and returns gradx(f)(x + theta), i.e. the gradient with respect to x of f(x+theta). This parameterised function is then a suitable input for root-finding, with root argmin(f) + theta.

Next, we simulate N_TESTS_PER_CASE values of theta from a normal distribution with mean 0 and standard deviation prior_sd. For each value of theta, we simulate a single observation of the parameterised function from a normal distribution with mean 0 and standard deviation error_sd. This simulates observations of the root of the parameterised function.

We then perform MCMC inference on the simulated data, using the GrapeNUTS sampler and the logdensity function joint_logdensity. Each time this function and its gradients are evaluated, the sampler performs root-finding using optimistix's Newton solver.

The heuristics (guess_static, guess previous and guess_implicit) are compared based on the total number of Newton steps spent on root-finding, for each of N_TESTS_PER_CASE MCMC runs per test function. Additional metrics including runtime and effective sample size of the MCMC chains are also saved.

"""

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
    Beale,
    Easom,
    Levy,
    Rastrigin,
    StyblinskiTang,
    run_benchmark,
    Rosenbrock,
)
from grapevine.heuristics import guess_implicit, guess_previous

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
    progress_bar=False,
)


@dataclass
class Case:
    name: str
    f: Callable
    max_steps: int
    solver: optx.Newton
    prior_sd: float
    error_sd: float
    default_guess_info: tuple


CASES = [
    Case(
        name="Easom",
        f=Easom(n_dimensions=2),
        max_steps=int(1e5),
        solver=optx.Newton(rtol=1e-8, atol=1e-8),
        error_sd=0.05,
        prior_sd=0.01,
        default_guess_info=(
            jnp.full((2,), jnp.pi),
            OrderedDict(theta=jnp.full((2,), 0.0)),
            0,
        ),
    ),
    Case(
        name="Levy3d",
        f=Levy(n_dimensions=3),
        max_steps=int(1e5),
        solver=optx.Newton(rtol=1e-8, atol=1e-8),
        error_sd=0.05,
        prior_sd=0.1,
        default_guess_info=(
            jnp.full((3,), 1.0),
            OrderedDict(theta=jnp.full((3,), 0.0)),
            0,
        ),
    ),
    Case(
        name="Beale",
        f=Beale(),
        max_steps=int(1e5),
        solver=optx.Newton(rtol=1e-8, atol=1e-8),
        error_sd=0.05,
        prior_sd=0.005,
        default_guess_info=(
            jnp.array([3.0, 0.5]),
            OrderedDict(theta=jnp.full((2,), 0.0)),
            0,
        ),
    ),
    Case(
        name="Rastrigin3d",
        f=Rastrigin(n_dimensions=3),
        max_steps=int(1e5),
        solver=optx.Newton(rtol=1e-8, atol=1e-8),
        error_sd=0.05,
        prior_sd=0.01,
        default_guess_info=(
            jnp.full((3,), 0.0),
            OrderedDict(theta=jnp.full((3,), 0.0)),
            0,
        ),
    ),
    Case(
        name="Rosenbrock8d",
        f=Rosenbrock(n_dimensions=8),
        max_steps=int(1e5),
        solver=optx.Newton(rtol=1e-8, atol=1e-8),
        error_sd=0.05,
        prior_sd=0.002,
        default_guess_info=(
            jnp.full((8,), 1.0),
            OrderedDict(theta=jnp.full((8,), 0.0)),
            0,
        ),
    ),
    Case(
        name="StyblinskiTang3d",
        f=StyblinskiTang(n_dimensions=3),
        max_steps=2000,
        solver=optx.Newton(rtol=1e-5, atol=1e-5),
        error_sd=0.05,
        prior_sd=0.3,
        default_guess_info=(
            jnp.full((3,), -2.903),
            OrderedDict(theta=jnp.full((3,), 0.0)),
            0,
        ),
    ),
    Case(
        name="Rosenbrock3d",
        f=Rosenbrock(n_dimensions=3),
        max_steps=800,
        solver=optx.Newton(rtol=1e-5, atol=1e-5),
        error_sd=0.05,
        prior_sd=0.3,
        default_guess_info=(
            jnp.full((3,), 1.0),
            OrderedDict(theta=jnp.full((3,), 0.0)),
            0,
        ),
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
    guess_info,
    gfunc,
    default_guess_info,
    solve_func,
    prior_sd,
    error_sd,
):
    last_solution, _, previous_steps = guess_info
    default_guess = default_guess_info[0]
    use_default = jnp.isclose(last_solution, default_guess).all()
    guess = jax.lax.cond(
        use_default,
        lambda g, p: default_guess_info[0],
        gfunc,
        guess_info,
        params,
    )
    solution, steps_here = solve_func(guess, params)
    log_prior = norm.logpdf(
        params["theta"], loc=jnp.zeros(default_guess.shape), scale=prior_sd
    ).sum()
    log_likelihood = norm.logpdf(obs, loc=solution, scale=error_sd).sum()
    steps = previous_steps + steps_here
    return log_prior + log_likelihood, (solution, params, steps)


def simulate_func(
    key: jax.Array, params: dict, guess: jax.Array, error_sd, solve_func
) -> tuple[jax.Array, jax.Array]:
    sol, _ = solve_func(guess, params)
    return sol, sol + jax.random.normal(key, shape=sol.shape) * error_sd


def main():
    results_list = []
    for case in CASES:
        print(f'Benchmarking case "{case.name}"...')
        solve = get_solve_func(
            f=case.f,
            solver=case.solver,
            max_steps=case.max_steps,
        )
        simulate = partial(
            simulate_func, solve_func=solve, error_sd=case.error_sd
        )

        @jax.jit
        def guess_static(guess_info, p):
            return case.default_guess_info[0]

        jlds = {
            callable_name(gfunc): partial(
                joint_logdensity,
                solve_func=solve,
                prior_sd=case.prior_sd,
                error_sd=case.error_sd,
                gfunc=gfunc,
                default_guess_info=case.default_guess_info,
            )
            for gfunc in (
                partial(guess_implicit, target_function=parameterise(case.f)),
                guess_static,
                guess_previous,
            )
        }
        case_results = run_benchmark(
            random_seed=SEED,
            joint_logdensity_funcs=jlds,
            baseline_params=case.default_guess_info[1],
            param_sd=case.prior_sd,
            n_test=N_TESTS_PER_CASE,
            run_grapenuts_kwargs=RUN_GRAPENUTS_KWARGS,
            sim_func=simulate,
            default_guess_info=case.default_guess_info,
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
