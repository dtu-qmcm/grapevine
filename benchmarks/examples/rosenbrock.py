"""Example of a joint log density with an embedded minimisation problem."""

from functools import partial

from jax import numpy as jnp
from jax.scipy.stats import norm
import jax
from jaxtyping import Scalar
import optimistix as optx
from grapevine.benchmarking import with_step_count
from grapevine.heuristics import guess_default, guess_previous

jax.config.update("jax_enable_x64", True)

ERROR_SD = 0.05
PRIOR_SD = 0.3


def get_default_guess(shape):
    return jnp.full(shape, 1.0)


SOLVER = optx.BFGS(rtol=1e-9, atol=1e-9)


@jax.jit
def rosenbrock(x: jax.Array, args: dict) -> Scalar:
    """A function for benchmarking minimisation algorithms.

    For 2 < dim < 8 the solution is x + args = 1.

    See <https://doi.org/10.1093/comjnl/3.3.175> and <https://doi.org/10.1162/evco.2009.17.3.437> for more.

    Example usage:

    ```python
    from jax import numpy as jnp
    import optimistix as optx
    from grapevine.example_functions import rosenbckro

    solver = optx.BFGS(rtol=1e-9, atol=1e-9)
    guess = jnp.array([0.1, 0.2, 0.3])
    theta = jnp.array([1.1, 1.2, 1.3])
    sol = optx.minimise(rosenbrock, solver, guess, args=theta)
    ```

    """

    xt = x + args["theta"]
    return (
        100.0 * (xt[1:] - xt[:-1] ** 2.0) ** 2.0 + (1 - xt[:-1]) ** 2.0
    ).sum()


@jax.jit
def solve(guess, theta):
    sol = optx.minimise(
        rosenbrock,
        SOLVER,
        guess,
        args=theta,
        max_steps=int(1e5),
    )
    return sol.value, jnp.array(sol.stats["num_steps"])


@jax.jit
def joint_logdensity(params, obs, guess):
    guess, previous_steps = guess
    solution, steps_here = solve(guess, params)
    log_prior = norm.logpdf(
        params["theta"], loc=jnp.zeros(params["theta"].shape), scale=PRIOR_SD
    ).sum()
    log_likelihood = norm.logpdf(obs, loc=solution, scale=ERROR_SD).sum()
    steps = previous_steps + steps_here
    return log_prior + log_likelihood, (solution, steps)


def simulate(
    key: jax.Array, params: dict, guess: jax.Array
) -> tuple[jax.Array, jax.Array]:
    sol, _ = solve(guess, params)
    return sol, sol + jax.random.normal(key, shape=sol.shape) * ERROR_SD


def get_default_guess_info(shape):
    """The value grapevine starts each trajectory from."""
    return get_default_guess(shape), jnp.array(0)


def get_guess_fns(shape):
    return {
        "guess_static": with_step_count(
            partial(guess_default, default_guess=get_default_guess(shape))
        ),
        "guess_previous": with_step_count(guess_previous),
    }
