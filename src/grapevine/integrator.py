"""Blackjax implementation of grapevine integrator."""

from typing import Callable, NamedTuple
import jax
from blackjax.types import ArrayTree, ArrayLikeTree

from blackjax.mcmc.metrics import KineticEnergy
from blackjax.mcmc.integrators import euclidean_momentum_update_fn


class GrapevineIntegratorState(NamedTuple):
    """State of the integrator."""

    position: ArrayTree
    momentum: ArrayLikeTree
    logdensity: float
    logdensity_grad: ArrayTree
    guess: ArrayTree


def grapevine_euclidean_position_update_fn(logdensity_fn: Callable):
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn, has_aux=True)

    def update(
        position: ArrayTree,
        kinetic_grad: ArrayTree,
        step_size: float,
        coef: float,
        guess: ArrayTree,
    ):
        new_position = jax.tree_util.tree_map(
            lambda x, grad: x + step_size * coef * grad,
            position,
            kinetic_grad,
        )
        (
            (logdensity, new_guess),
            logdensity_grad,
        ) = logdensity_and_grad_fn(new_position, guess=guess)
        del guess
        return new_position, logdensity, logdensity_grad, new_guess

    return update


def _format_grapevine_euclidean_state_output(
    position,
    momentum,
    logdensity,
    logdensity_grad,
    kinetic_grad,
    position_update_info,
    momentum_update_info,
):
    """Get a GrapevineIntegratorState from the required info, and clean up."""
    del kinetic_grad, momentum_update_info
    return GrapevineIntegratorState(
        position,
        momentum,
        logdensity,
        logdensity_grad,
        position_update_info,
    )


def grapevine_generalized_two_stage_integrator(
    operator1: Callable,
    operator2: Callable,
    coefficients: list[float],
    format_output_fn: Callable = lambda x: x,
):
    def one_step(state: GrapevineIntegratorState, step_size: float):
        position, momentum, _, logdensity_grad, guess = state
        # auxiliary infomation generated during integration for diagnostics.
        # It is updated by the operator1 and operator2 at each call.
        momentum_update_info = None
        position_update_info = guess
        for i, coef in enumerate(coefficients[:-1]):
            if i % 2 == 0:
                momentum, kinetic_grad, momentum_update_info = operator1(
                    momentum,
                    logdensity_grad,
                    step_size,
                    coef,
                    auxiliary_info=momentum_update_info,
                    is_last_call=False,
                )
            else:
                (
                    position,
                    logdensity,
                    logdensity_grad,
                    position_update_info,
                ) = operator2(
                    position,
                    kinetic_grad,
                    step_size,
                    coef,
                    guess=position_update_info,
                )
        # Separate the last steps to short circuit the computation of the
        # kinetic_grad.
        momentum, kinetic_grad, momentum_update_info = operator1(
            momentum,
            logdensity_grad,
            step_size,
            coefficients[-1],
            momentum_update_info,
            is_last_call=True,
        )
        return format_output_fn(
            position,
            momentum,
            logdensity,
            logdensity_grad,
            kinetic_grad,
            position_update_info,
            momentum_update_info,
        )

    return one_step


def generate_grapevine_euclidean_integrator(coefficients):
    def euclidean_integrator(
        logdensity_fn: Callable, kinetic_energy_fn: KineticEnergy
    ) -> Callable:
        position_update_fn = grapevine_euclidean_position_update_fn(logdensity_fn)
        momentum_update_fn = euclidean_momentum_update_fn(kinetic_energy_fn)
        one_step = grapevine_generalized_two_stage_integrator(
            momentum_update_fn,
            position_update_fn,
            coefficients,
            format_output_fn=_format_grapevine_euclidean_state_output,
        )
        return one_step

    return euclidean_integrator


velocity_verlet_coefficients = [0.5, 1.0, 0.5]
grapevine_velocity_verlet = generate_grapevine_euclidean_integrator(
    velocity_verlet_coefficients
)
