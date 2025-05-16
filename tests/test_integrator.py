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
)
from grapevine.examples.simple_example_problem import (
    posterior_logdensity,
    joint_logdensity,
    default_guess_info,
    obs,
)

initial_position = jnp.array(0.0)
initial_momentum = jnp.array(0.5)
inverse_mass_matrix = jnp.array([1.0])
metric = default_metric(inverse_mass_matrix)


def get_initial_state():
    """Get the initial integrator state."""
    (initial_logdensity, next_guess_info), logdensity_grad = jax.value_and_grad(
        posterior_logdensity, has_aux=True
    )(initial_position, guess_info=default_guess_info)
    return GrapevineIntegratorState(
        position=initial_position,
        momentum=initial_momentum,
        logdensity=initial_logdensity,
        logdensity_grad=logdensity_grad,
        guess_info=next_guess_info,
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
        return joint_logdensity(a, obs, default_guess_info)[0]

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
