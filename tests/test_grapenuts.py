import jax

from blackjax.util import run_inference_algorithm
from jax import numpy as jnp

import optimistix as optx
import pytest

from grapevine.grapenuts import init, grapenuts_sampler
from tests.simple_example_problem import (
    fn,
    posterior_logdensity,
    default_guess,
    solver,
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


def test_the_state_carries_the_last_solution():
    """`guess` holds the solution found at the state's own position."""
    state = init(initial_position, posterior_logdensity, default_guess)
    _, solution = posterior_logdensity(initial_position, guess=default_guess)
    assert jnp.array_equal(state.guess, solution)
    assert not jnp.array_equal(state.guess, default_guess)


def test_the_carried_guess_is_the_solution_at_the_new_position():
    """After a step, `guess` is the root at the position it came with.

    This is the contract the next trajectory relies on: it starts its solve
    from this value, at this position, so the two have to correspond. The
    first step of a trajectory used to ignore both and start from
    `default_guess` instead, which is the step furthest from any solution and
    so the most expensive one.
    """
    kernel = grapenuts_sampler(
        posterior_logdensity,
        default_guess=default_guess,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=0.01,
    )
    state = init(initial_position, posterior_logdensity, default_guess)
    for key in jax.random.split(jax.random.key(SEED), 3):
        state, _ = kernel.step(key, state)
        _, solution = posterior_logdensity(state.position, guess=state.guess)
        assert jnp.allclose(state.guess, solution, rtol=1e-6)
    assert not jnp.array_equal(state.guess, default_guess)


def test_a_non_finite_guess_falls_back_to_the_default():
    """One failed solve must not poison every trajectory after it.

    Resetting each trajectory to `default_guess` used to guarantee this for
    free. Carrying the solution over removes that guarantee, so the fallback
    below replaces it: a guess that is not finite is swapped for the default
    rather than being handed to the solver again.
    """
    kernel = grapenuts_sampler(
        posterior_logdensity,
        default_guess=default_guess,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=0.01,
    )
    state = init(initial_position, posterior_logdensity, default_guess)
    poisoned = state._replace(guess=jnp.array(jnp.nan))
    recovered, _ = kernel.step(jax.random.key(SEED), poisoned)
    assert jnp.isfinite(recovered.guess)
    assert jnp.isfinite(recovered.logdensity)


def step_counting_logdensity(a, guess):
    start, previous_steps = guess
    solution = optx.root_find(fn, solver, start, args=a)
    logdensity, _ = posterior_logdensity(a, guess=start)
    steps = previous_steps + solution.stats["num_steps"]
    return logdensity, (solution.value, steps)


def reset_step_count(solution):
    return solution[0], jnp.zeros_like(solution[1])


@pytest.mark.parametrize("resets", [True, False])
def test_carry_fn_decides_what_crosses_a_trajectory_boundary(resets):
    counting_guess = (default_guess, jnp.array(0))
    earlier_steps = 1000
    kwargs = {"carry_fn": reset_step_count} if resets else {}
    kernel = grapenuts_sampler(
        step_counting_logdensity,
        default_guess=counting_guess,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=0.01,
        solver_info_fn=lambda solution: solution[1],
        **kwargs,
    )
    state = init(
        initial_position,
        step_counting_logdensity,
        counting_guess,
        solver_info_fn=lambda solution: solution[1],
    )
    root, _ = state.guess
    state = state._replace(guess=(root, jnp.array(earlier_steps)))
    for key in jax.random.split(jax.random.key(SEED), 3):
        state, _ = kernel.step(key, state)
        assert jnp.array_equal(state.solver_info, state.guess[1])
        if resets:
            assert state.solver_info < earlier_steps
        else:
            assert state.solver_info >= earlier_steps
