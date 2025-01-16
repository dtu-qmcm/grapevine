"""Make data for a figure comparing grapeNUTS and NUTS on a single trajectory.

The target problem is a 2-D minimisation problem (probably Rosenbrock?).

The left pane of the figure shows the values of the target solution x at each
point in the trajectory, as well as the distance of the point from the default guess).

The right pane shows the histogram of total number of Newton steps taken by
each method.

"""

from functools import partial
import jax
import jax.numpy as jnp
import optimistix as optx
import polars as pl
from blackjax.mcmc.metrics import default_metric
from jax.scipy.stats import norm
from jaxtyping import Scalar
from grapevine.integrator import (
    grapevine_velocity_verlet,
    GrapevineIntegratorState,
)
from grapevine import run_grapenuts

SEED = 1234
N_WARMUP = 1000
N_SAMPLE = 1000
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.99
SD = 0.05

solver = optx.BFGS(rtol=1e-6, atol=1e-6)

initial_momentum = jnp.array([0.1, 0.1])
inverse_mass_matrix = jnp.array([1, 1])
metric = default_metric(inverse_mass_matrix)
TRUE_PARAMS = jnp.array([1.3, 0.8])
DEFAULT_GUESS = TRUE_PARAMS
INITIAL_POSITION = jnp.array([1.0404058, 1.0131096])


def rosenbrock2d(x: jax.Array, args: jax.Array) -> Scalar:
    xt = x * args
    return (1 - xt[0]) ** 2 + 100 * (xt[1] - xt[0] ** 2) ** 2


def camel(x: jax.Array, args: jax.Array) -> Scalar:
    xt = x + args
    return (
        2 * xt[0] ** 2
        - 1.05 * xt[0] ** 4
        + (xt[0] ** 6) / 6
        + xt[0] * xt[1]
        + xt[1] ** 2
    )


def posterior_logdensity(params, guess):
    sol = optx.minimise(
        rosenbrock2d,
        solver,
        guess,
        args=params,
        max_steps=int(1e6),
    )
    log_prior = norm.logpdf(params, loc=TRUE_PARAMS, scale=0.1).sum()
    log_likelihood = norm.logpdf(TRUE_PARAMS, loc=sol.value, scale=SD).sum()
    return log_prior + log_likelihood, sol.value


def get_initial_state(initial_position):
    """Get the initial integrator state."""
    (initial_logdensity, next_guess), logdensity_grad = jax.value_and_grad(
        posterior_logdensity, has_aux=True
    )(initial_position, guess=DEFAULT_GUESS)
    return GrapevineIntegratorState(
        position=initial_position,
        momentum=initial_momentum,
        logdensity=initial_logdensity,
        logdensity_grad=logdensity_grad,
        guess=next_guess,
    )


run_mcmc = partial(
    run_grapenuts,
    logdensity_fn=posterior_logdensity,
    init_parameters=TRUE_PARAMS,
    default_guess=DEFAULT_GUESS,  # type: ignore
    num_warmup=N_WARMUP,
    num_samples=N_SAMPLE,
    initial_step_size=INIT_STEPSIZE,
    max_num_doublings=MAX_TREEDEPTH,
    is_mass_matrix_diagonal=False,
    target_acceptance_rate=TARGET_ACCEPT,
    progress_bar=False,
)


def test_trajectory(initial_position):
    """Test a trajectory, comparing NUTS and grapeNUTS solver approaches."""
    initial_state = get_initial_state(initial_position)
    step = grapevine_velocity_verlet(
        posterior_logdensity,
        metric.kinetic_energy,
    )

    states = []

    for i in range(50):
        state = jax.lax.fori_loop(
            0,
            i,
            lambda _, state: step(state, 0.01),
            initial_state,
        )
        states.append(state)
    guess = DEFAULT_GUESS
    steps = []
    for state in states:
        sol_gn = optx.minimise(
            rosenbrock2d,
            solver,
            guess,
            args=state.position,
            max_steps=int(1e6),
        )
        sol_nuts = optx.minimise(
            rosenbrock2d,
            solver,
            DEFAULT_GUESS,
            args=state.position,
            max_steps=int(1e6),
        )
        steps.append(
            {
                "sol_0": sol_gn.value[0].item(),
                "sol_1": sol_gn.value[1].item(),
                "guess_0": guess[0].item(),
                "guess_1": guess[1].item(),
                "default_0": DEFAULT_GUESS[0],
                "default_1": DEFAULT_GUESS[1],
                "GN": sol_gn.stats["num_steps"],
                "NUTS": sol_nuts.stats["num_steps"],
            }
        )
        guess = state.guess

    steps_df = pl.from_records(steps)
    return states, steps_df


def main():
    key = jax.random.key(SEED)
    mcmc, info = run_mcmc(rng_key=key)
    traj_states, traj_steps = test_trajectory(initial_position=mcmc.position[50])
    traj_steps.write_csv("benchmarks/trajectory.csv")
    print(traj_steps)
    print(traj_steps.mean())


if __name__ == "__main__":
    main()
