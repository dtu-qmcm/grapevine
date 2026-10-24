import jax

from blackjax.util import run_inference_algorithm
from jax import numpy as jnp

from grapevine.grapenuts import init, grapenuts_sampler
from grapevine.examples.simple_example_problem import (
    posterior_logdensity,
    default_guess,
)

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
    )
    assert states.position.shape == (10,)
    assert jnp.isfinite(states.logdensity).all()


def test_solver_info_default_is_empty():
    """By default nothing about the solver is recorded."""
    state = init(initial_position, posterior_logdensity, default_guess)
    assert jax.tree.leaves(state.solver_info) == []


def test_solver_info_fn_is_used():
    """A solver_info_fn records something about each iteration's solve."""
    key = jax.random.key(SEED)
    solver_info_fn = jnp.square
    init_state = init(
        initial_position,
        posterior_logdensity,
        default_guess,
        solver_info_fn=solver_info_fn,
    )
    kernel = grapenuts_sampler(
        posterior_logdensity,
        default_guess=default_guess,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=0.01,
        solver_info_fn=solver_info_fn,
    )
    _, (states, _) = run_inference_algorithm(
        key,
        kernel,
        num_steps=10,
        initial_state=init_state,
    )
    assert states.solver_info.shape == (10,)
    assert (states.solver_info > 0).all()
