from typing import Callable

import jax

from blackjax.base import AdaptationAlgorithm
from blackjax.adaptation.base import AdaptationInfo, AdaptationResults
from blackjax.adaptation.window_adaptation import base, build_schedule
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from jax import numpy as jnp

from grapevine.integrator import grapevine_velocity_verlet


def return_all_adapt_info(state, info, adaptation_state):
    """Return fully populated AdaptationInfo.

    Used for adaptation_info_fn parameters of the adaptation algorithms.
    """
    return AdaptationInfo(state, info, adaptation_state)


def grapenuts_window_adaptation(
    algorithm,
    logdensity_fn: Callable,
    default_guess_info: ArrayTree,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    adaptation_info_fn: Callable = return_all_adapt_info,
    integrator=grapevine_velocity_verlet,
    **extra_parameters,
) -> AdaptationAlgorithm:
    mcmc_kernel = algorithm.build_kernel(default_guess_info, integrator)

    adapt_init, adapt_step, adapt_final = base(
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )

    def one_step(carry, xs):
        _, rng_key, adaptation_stage = xs
        state, adaptation_state = carry

        new_state, info = mcmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            adaptation_state.step_size,
            adaptation_state.inverse_mass_matrix,
            **extra_parameters,
        )
        new_adaptation_state = adapt_step(
            adaptation_state,
            adaptation_stage,
            new_state.position,
            info.acceptance_rate,
        )

        return (
            (new_state, new_adaptation_state),
            adaptation_info_fn(new_state, info, new_adaptation_state),
        )

    def run(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        num_steps: int = 1000,
    ):
        init_state = algorithm.init(position, logdensity_fn, default_guess_info)
        init_adaptation_state = adapt_init(position, initial_step_size)

        if progress_bar:
            print("Running window adaptation")
        scan_fn = gen_scan_fn(num_steps, progress_bar=progress_bar)
        start_state = (init_state, init_adaptation_state)
        keys = jax.random.split(rng_key, num_steps)
        schedule = build_schedule(num_steps)
        last_state, info = scan_fn(
            one_step,
            start_state,
            (jnp.arange(num_steps), keys, schedule),
        )

        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            "step_size": step_size,
            "inverse_mass_matrix": inverse_mass_matrix,
            **extra_parameters,
        }

        return (
            AdaptationResults(
                last_chain_state,
                parameters,
            ),
            info,
        )

    return AdaptationAlgorithm(run)
