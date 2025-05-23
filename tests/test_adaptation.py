"""Tests for the adaptation module."""

import jax

from jax import numpy as jnp

from grapevine.adaptation import grapenuts_window_adaptation
from grapevine.grapenuts import grapenuts_sampler
from grapevine.integrator import grapevine_velocity_verlet

from grapevine.examples.simple_example_problem import (
    posterior_logdensity,
    default_guess_info,
)

SEED = 12345
INITIAL_PARAM_VALUE = jnp.array(0.3)


def test_window_adaptation():
    """Check that window adaptation runs."""
    key = jax.random.key(SEED)
    adaptation = grapenuts_window_adaptation(
        grapenuts_sampler,
        posterior_logdensity,
        default_guess_info=default_guess_info,
        progress_bar=False,
        integrator=grapevine_velocity_verlet,
    )
    (initial_state, tuned_parameters), (_, info, _) = adaptation.run(
        key,
        INITIAL_PARAM_VALUE,
        num_steps=5,  #  type: ignore
    )
