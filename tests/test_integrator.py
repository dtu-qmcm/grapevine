"""Tests for the function grapevine_velocity_verlet"""

from functools import partial

import chex
import jax

from blackjax.mcmc.integrators import IntegratorState, velocity_verlet
from blackjax.mcmc.metrics import default_metric
from jax import numpy as jnp
from grapevine.integrator import (
    grapevine_velocity_verlet,
    GrapevineIntegratorState,
    GuessInputs,
)
from grapevine.examples.simple_example_problem import (
    posterior_logdensity,
    joint_logdensity,
    default_guess,
    obs,
)

initial_position = jnp.array(0.0)
initial_momentum = jnp.array(0.5)
inverse_mass_matrix = jnp.array([1.0])
metric = default_metric(inverse_mass_matrix)


def get_initial_state():
    """Get the initial integrator state."""
    (initial_logdensity, solution), logdensity_grad = jax.value_and_grad(
        posterior_logdensity, has_aux=True
    )(initial_position, guess=default_guess)
    return GrapevineIntegratorState(
        position=initial_position,
        momentum=initial_momentum,
        logdensity=initial_logdensity,
        logdensity_grad=logdensity_grad,
        guess_inputs=GuessInputs(solution, initial_position, jnp.bool_(False)),
    )


def get_final_state():
    """Get the final integrator state."""
    initial_state = get_initial_state()
    step = grapevine_velocity_verlet(
        posterior_logdensity, metric.kinetic_energy
    )
    return jax.lax.fori_loop(
        0,
        50,
        lambda _, state: step(state, 0.001),
        initial_state,
    )


def test_evolution():
    """Check that the final position is as expected."""
    expected_final_position = jnp.array(0.02488716)
    final_state = get_final_state()
    chex.assert_trees_all_close(
        final_state.position,
        expected_final_position,
        atol=1e-2,
    )


def test_conservation_of_energy():
    """Check that energy is conserved."""
    initial_state = get_initial_state()
    final_state = get_final_state()
    initial_energy = -initial_state.logdensity + metric.kinetic_energy(
        initial_momentum
    )
    final_energy = -final_state.logdensity + metric.kinetic_energy(
        final_state.momentum
    )
    chex.assert_trees_all_close(initial_energy, final_energy, atol=1e-3)


def test_same_as_non_grapevine():
    """Check that grapevine gives result is same as plain velocity_verlet."""

    def joint_logdensity_vv(a, obs):
        return joint_logdensity(a, obs, default_guess)[0]

    final_state_gvvv = get_final_state()
    posterior_logdensity_vv = partial(joint_logdensity_vv, obs=obs)
    initial_logdensity_vv, logdensity_grad_vv = jax.value_and_grad(
        posterior_logdensity_vv
    )(initial_position)
    initial_state_vv = IntegratorState(
        position=initial_position,
        momentum=initial_momentum,
        logdensity=initial_logdensity_vv,
        logdensity_grad=logdensity_grad_vv,
    )
    step_vv = velocity_verlet(posterior_logdensity_vv, metric.kinetic_energy)
    final_state_vv = jax.lax.fori_loop(
        0,
        50,
        lambda _, state: step_vv(state, 0.001),
        initial_state_vv,
    )
    chex.assert_trees_all_close(
        final_state_gvvv.position,
        final_state_vv.position,
        atol=1e-3,
    )


def test_guess_fn_is_used():
    """The integrator consults guess_fn rather than reusing the solution."""
    calls = []

    def guess_fn(inputs, position):
        calls.append(position)
        return inputs.solution

    step = grapevine_velocity_verlet(
        posterior_logdensity, metric.kinetic_energy, guess_fn
    )
    step(get_initial_state(), 0.001)
    assert calls


def test_guess_fn_does_not_change_the_trajectory():
    """Whichever guess the solver starts from, it finds the same root."""
    always_default = lambda inputs, position: default_guess  # noqa: E731
    step = grapevine_velocity_verlet(
        posterior_logdensity, metric.kinetic_energy, always_default
    )
    final_state = jax.lax.fori_loop(
        0, 50, lambda _, state: step(state, 0.001), get_initial_state()
    )
    chex.assert_trees_all_close(
        final_state.position, get_final_state().position, atol=1e-6
    )
