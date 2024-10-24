"""Provides utility function `run_grapenuts`."""

from typing import Callable, TypedDict, Unpack

import equinox as eqx
import jax

from blackjax.types import ArrayTree
from blackjax.util import run_inference_algorithm
from jax._src.random import KeyArray

from grapevine import grapenuts_sampler, grapevine_velocity_verlet
from grapevine.adaptation import grapenuts_window_adaptation


class AdaptationKwargs(TypedDict):
    """Keyword arguments to the blackjax function window_adaptation."""

    initial_step_size: float
    max_num_doublings: int
    is_mass_matrix_diagonal: bool
    target_acceptance_rate: float


@eqx.filter_jit
def run_grapenuts(
    logdensity_fn: Callable,
    rng_key: KeyArray,
    init_parameters: ArrayTree,
    num_warmup: int,
    num_samples: int,
    default_guess: ArrayTree,
    progress_bar: bool = True,
    **adapt_kwargs: Unpack[AdaptationKwargs],
):
    """Run the default NUTS algorithm with blackjax."""
    warmup = grapenuts_window_adaptation(
        grapenuts_sampler,
        logdensity_fn,
        default_guess=default_guess,
        progress_bar=progress_bar,
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
    )
    _, (states, info) = run_inference_algorithm(
        sample_key,
        kernel,
        num_steps=num_samples,
        initial_state=initial_state,
        progress_bar=progress_bar,
    )
    return states, info
