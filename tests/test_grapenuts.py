import jax

from blackjax.util import run_inference_algorithm
from jax import numpy as jnp

from grapevine.grapenuts import init, grapenuts_sampler
from tests.simple_example_problem import posterior_logdensity, default_guess

SEED = 12345
initial_position = jnp.array(0.0)
inverse_mass_matrix = jnp.array([1.0])


def test_sampler():
    """Test that the grapenuts sampler runs."""
    key = jax.random.key(SEED)
    init_state = init(initial_position, posterior_logdensity, default_guess)
    kernel = grapenuts_sampler(
        posterior_logdensity,
        default_guess=default_guess,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=0.01,
    )
    _, (states, info) = run_inference_algorithm(
        key,
        kernel,
        num_steps=10,
        initial_state=init_state,
        progress_bar=False,
    )
