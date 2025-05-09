"""Tests for functions in the util module."""

import jax

from jax import numpy as jnp

from grapevine.util import run_grapenuts

from grapevine.examples.simple_example_problem import (
    posterior_logdensity,
    default_guess,
)

SEED = 12345
initial_position = jnp.array(0.0)
inverse_mass_matrix = jnp.array([1.0])


def test_run_grapenuts():
    key = jax.random.key(SEED)
    _ = run_grapenuts(
        logdensity_fn=posterior_logdensity,
        rng_key=key,
        init_parameters=initial_position,
        num_warmup=10,
        num_samples=10,
        default_guess=default_guess,
        progress_bar=False,
        initial_step_size=0.01,
        max_num_doublings=4,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=0.8,
    )
