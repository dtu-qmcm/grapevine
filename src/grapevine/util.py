"""Provides utility function `run_grapenuts`."""

import functools
from typing import Callable, TypedDict, Unpack

import blackjax
import jax

from blackjax.types import ArrayTree
from jax._src.random import KeyArray

from grapevine import grapenuts_sampler, grapevine_velocity_verlet
from grapevine.adaptation import grapenuts_window_adaptation


class AdaptationKwargs(TypedDict):
    """Keyword arguments to the blackjax function window_adaptation."""

    initial_step_size: float
    max_num_doublings: int
    is_mass_matrix_diagonal: bool
    target_acceptance_rate: float


@functools.partial(jax.jit, static_argnames=["kernel", "num_samples"])
def _inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run MCMC with blackjax."""

    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)
    return states, info


def run_grapenuts(
    logdensity_fn: Callable,
    rng_key: KeyArray,
    init_parameters: ArrayTree,
    num_warmup: int,
    num_samples: int,
    default_guess: ArrayTree,
    **adapt_kwargs: Unpack[AdaptationKwargs],
):
    """Run the default NUTS algorithm with blackjax."""
    warmup = grapenuts_window_adaptation(
        grapenuts_sampler,
        logdensity_fn,
        default_guess=default_guess,
        progress_bar=True,
        integrator=grapevine_velocity_verlet,
        **adapt_kwargs,
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    (initial_state, tuned_parameters), (_, info, _) = warmup.run(
        warmup_key,
        init_parameters,
        num_steps=num_warmup,  #  type: ignore
    )
    rng_key, sample_key = jax.random.split(rng_key)
    kernel = grapenuts_sampler(
        logdensity_fn,
        default_guess=default_guess,
        **tuned_parameters,
    ).step
    states, info = _inference_loop(
        sample_key,
        kernel=kernel,
        initial_state=initial_state,
        num_samples=num_samples,
    )
    return states, info
