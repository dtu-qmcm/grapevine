"""Make data for a figure comparing grapeNUTS and NUTS on a single trajectory.

The target problem is a 2-D minimisation problem.

The left pane of the figure shows the values of the target solution x at each
point in the trajectory, as well as the distance of the point from the default guess).

The right pane shows the histogram of total number of Newton steps taken by
each method.

"""

from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import polars as pl
from blackjax.mcmc.metrics import default_metric
from jax.scipy.stats import norm

from grapevine import run_grapenuts
from grapevine.benchmarking import Rosenbrock
from grapevine.heuristics import guess_implicit as _guess_implicit
from grapevine.heuristics import guess_previous
from grapevine.integrator import (
    GrapevineIntegratorState,
    grapevine_velocity_verlet,
)

SEED = 1234
N_WARMUP = 2000
N_SAMPLE = 500
INIT_STEPSIZE = 0.001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.999
MAX_STEPS = int(1e6)
ERROR_SD = 0.05
PRIOR_SD = 0.35

INITIAL_MOMENTUM = {"theta": jnp.array([0.1, 0.1])}
inverse_mass_matrix = jnp.array([1, 1])
metric = default_metric(inverse_mass_matrix)
TRUE_PARAMS = {"theta": jnp.array([0.0, 0.0])}
DEFAULT_GUESS = jnp.array([1.0, 1.0])
DEFAULT_GUESS_INFO = (DEFAULT_GUESS, TRUE_PARAMS, DEFAULT_GUESS, 0)
SOLVER = optx.Newton(rtol=1e-4, atol=1e-4)
TARGET_FUNCTION = Rosenbrock(n_dimensions=2)


def parameterise(f):
    def f_parameterised(x, args):
        x_plus_theta = x + args["theta"]
        return jax.grad(f)(x_plus_theta)

    return f_parameterised


def get_solve_with_steps(f, solver, max_steps):
    options = dict()
    f_struct = jax.ShapeDtypeStruct((2,), jnp.float32)
    aux_struct = None
    tags = frozenset()

    def fn(x, args):
        return parameterise(f)(x, args), None

    def solve_with_steps(y, args):
        path_to_sol = []
        step = eqx.filter_jit(
            eqx.Partial(
                solver.step, fn=fn, args=args, options=options, tags=tags
            )
        )
        terminate = eqx.filter_jit(
            eqx.Partial(
                solver.terminate, fn=fn, args=args, options=options, tags=tags
            )
        )

        # Initial state before we start solving.
        state = solver.init(fn, y, args, options, f_struct, aux_struct, tags)
        done, result = terminate(y=y, state=state)

        # Alright, enough setup. Let's do the solve!
        while not done:
            # print(f"Evaluating point {y} with value {fn(y, args)[0]}.")
            y, state, aux = step(y=y, state=state)
            done, result = terminate(y=y, state=state)
            path_to_sol.append(y)
        if result != optx.RESULTS.successful:
            print(f"Oh no! Got error {result}.")
        y, _, _ = solver.postprocess(
            fn, y, aux, args, options, state, tags, result
        )
        # print(f"Found solution {y} with value {fn(y, args)[0]}.")
        return y, path_to_sol

    return solve_with_steps


def get_solve_func(f, solver, max_steps):
    def solve_func(guess, params):
        sol = optx.root_find(
            parameterise(f),
            solver,
            guess,
            args=params,
            max_steps=max_steps,
        )
        return sol.value, sol.stats["num_steps"]

    return solve_func


solve_target = get_solve_func(
    f=TARGET_FUNCTION,
    solver=SOLVER,
    max_steps=MAX_STEPS,
)
solve_target_with_steps = get_solve_with_steps(
    f=TARGET_FUNCTION,
    solver=SOLVER,
    max_steps=MAX_STEPS,
)


@jax.jit
def guess_static(guess_info, p):
    return DEFAULT_GUESS_INFO[0]


@jax.jit
def guess_implicit(guess_info, p):
    return _guess_implicit(
        guess_info,
        p,
        target_function=parameterise(TARGET_FUNCTION),
    )


def get_initial_state(initial_position, target_logdensity):
    """Get the initial integrator state."""
    grad_func = jax.value_and_grad(target_logdensity, has_aux=True)
    (ld, gi), grad = grad_func(initial_position, guess_info=DEFAULT_GUESS_INFO)
    return GrapevineIntegratorState(
        position=initial_position,
        momentum=INITIAL_MOMENTUM,
        logdensity=ld,
        logdensity_grad=grad,
        guess_info=DEFAULT_GUESS_INFO,
    )


def simulate(
    key: jax.Array,
    params: dict,
    guess: jax.Array,
    error_sd: float,
) -> tuple[jax.Array, jax.Array]:
    sol, _ = solve_target(guess, params)
    return sol, sol + jax.random.normal(key, shape=sol.shape) * error_sd


def joint_logdensity(params, obs, guess_info, gfunc):
    last_solution, _, _, _ = guess_info
    use_default = jnp.isclose(last_solution, DEFAULT_GUESS).all()
    guess = jax.lax.cond(use_default, guess_static, gfunc, guess_info, params)
    solution, n_steps = solve_target(guess, params)
    log_prior = norm.logpdf(
        params["theta"],
        loc=jnp.zeros(DEFAULT_GUESS.shape),
        scale=PRIOR_SD,
    ).sum()
    log_likelihood = norm.logpdf(obs, loc=solution, scale=ERROR_SD).sum()
    return log_prior + log_likelihood, (
        solution,
        params,
        guess,
        n_steps,
    )


def test_trajectory(
    initial_position,
    lp_func: Callable,
    gfuncs: dict[str, Callable],
    num_integration_steps: int,
    step_size: float,
):
    out_list = []
    initial_state = get_initial_state(initial_position, lp_func)
    states = []
    step = grapevine_velocity_verlet(lp_func, metric.kinetic_energy)
    for i in range(num_integration_steps):
        new_state = jax.lax.fori_loop(
            0,
            i,
            lambda _, state: step(state, step_size),
            initial_state,
        )
        states.append(new_state)
    previous_state = initial_state
    for i, state in enumerate(states[1:]):
        for gfunc_name, gfunc in gfuncs.items():
            guess = (
                gfunc(previous_state.guess_info, state.position)
                if i > 0
                else DEFAULT_GUESS
            )
            sol, path_to_sol = solve_target_with_steps(guess, state.position)
            for j, intermediate_solution in enumerate(path_to_sol):
                out_list.append(
                    {
                        "i": i,
                        "j": j,
                        "gfunc": gfunc_name,
                        "sol_0": intermediate_solution[0].item(),
                        "sol_1": intermediate_solution[1].item(),
                        "guess_0": guess[0].item(),
                        "guess_1": guess[1].item(),
                        # "steps": n_steps,
                    }
                )
        previous_state = state
    steps_df = pl.from_records(out_list)
    return steps_df


def main():
    key = jax.random.key(SEED)
    sim_key, mcmc_key = jax.random.split(key)
    _, obs = simulate(sim_key, TRUE_PARAMS, DEFAULT_GUESS_INFO[0], ERROR_SD)
    log_posterior_guess_implicit = partial(
        joint_logdensity,
        obs=obs,
        gfunc=guess_implicit,
    )
    run_mcmc = partial(
        run_grapenuts,
        logdensity_fn=log_posterior_guess_implicit,
        init_parameters=TRUE_PARAMS,
        default_guess_info=DEFAULT_GUESS_INFO,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLE,
        initial_step_size=INIT_STEPSIZE,
        max_num_doublings=MAX_TREEDEPTH,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=TARGET_ACCEPT,
        progress_bar=False,
    )
    mcmc, info = run_mcmc(rng_key=mcmc_key)
    print(f"Number of divergent transitions: {info.is_divergent.sum()}")
    i = -1
    traj_steps = test_trajectory(
        initial_position={"theta": mcmc.position["theta"][i]},
        lp_func=log_posterior_guess_implicit,
        gfuncs={
            "guess_implicit": guess_implicit,
            "guess_static": guess_static,
            "guess_previous": guess_previous,
        },
        num_integration_steps=12,
        step_size=0.01,
    )
    traj_steps.write_csv("benchmarks/trajectory.csv")
    print(traj_steps.sort("i").head(10))


if __name__ == "__main__":
    main()
