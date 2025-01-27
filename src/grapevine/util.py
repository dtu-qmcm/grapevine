"""Provides utility functions run_grapenuts, get_idata and time_run."""

import time
from typing import Callable, TypedDict, Unpack

import arviz as az
import jax
import numpy as np
from blackjax import nuts
from blackjax import window_adaptation as nuts_window_adaptation
from blackjax.types import ArrayTree
from blackjax.util import run_inference_algorithm
from jax import numpy as jnp

from grapevine import grapenuts_sampler, grapevine_velocity_verlet
from grapevine.adaptation import grapenuts_window_adaptation


class AdaptationKwargs(TypedDict):
    """Keyword arguments to the blackjax function window_adaptation."""

    initial_step_size: float
    max_num_doublings: int
    is_mass_matrix_diagonal: bool
    target_acceptance_rate: float


def run_grapenuts(
    logdensity_fn: Callable,
    rng_key: jax.Array,
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


def run_nuts(
    logdensity_fn: Callable,
    rng_key: jax.Array,
    init_parameters: ArrayTree,
    num_warmup: int,
    num_samples: int,
    progress_bar: bool = True,
    **adapt_kwargs: Unpack[AdaptationKwargs],
):
    """Run the default NUTS algorithm with blackjax."""
    warmup = nuts_window_adaptation(
        nuts,
        logdensity_fn,
        progress_bar=progress_bar,
        **adapt_kwargs,
    )
    warmup_key, sample_key = jax.random.split(rng_key, 2)
    (initial_state, tuned_parameters), _ = warmup.run(
        warmup_key,
        init_parameters,
        num_steps=num_warmup,  #  type: ignore
    )
    kernel = nuts(logdensity_fn, **tuned_parameters)
    (_, out) = run_inference_algorithm(
        sample_key,
        kernel,
        num_samples,
        initial_state,
    )
    return out


def get_idata(samples, info, coords=None, dims=None) -> az.InferenceData:
    """Get an arviz InferenceData from a grapeNUTS output."""
    sample_dict = {k: jnp.expand_dims(v, 0) for k, v in samples.position.items()}
    posterior = az.convert_to_inference_data(
        sample_dict,
        group="posterior",
        coords=coords,
        dims=dims,
    )
    sample_stats = az.convert_to_inference_data(
        {
            "diverging": info.is_divergent,
            "energy": info.energy,
        },
        group="sample_stats",
    )
    idata = az.concat(posterior, sample_stats)
    assert idata is not None, "idata should not be None!"
    return idata


def time_run(run_fn):
    """Time run_fn and check how many effective samples it generates."""
    _ = run_fn()  # dummy run for jit compiling
    start = time.time()
    out = run_fn()
    _ = next(iter(out[0].position.values())).block_until_ready()
    end = time.time()
    idata = get_idata(*out)
    runtime = end - start
    ess = az.ess(idata.posterior)  # type: ignore
    neff = np.sum([ess[v].values.sum() for v in ess.data_vars]).item()
    return {"time": runtime, "neff": neff}
