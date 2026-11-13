"""The NUTS algorithm sped up for guess problems using the grapevine method."""

from typing import Callable, NamedTuple
import jax
from jax import numpy as jnp
from blackjax.types import Array, ArrayTree, ArrayLikeTree, PRNGKey
from blackjax import GenerateSamplingAPI
from blackjax.base import SamplingAlgorithm

from blackjax.mcmc.nuts import NUTSInfo, iterative_nuts_proposal
from blackjax.mcmc.metrics import MetricTypes, default_metric

from grapevine.heuristics import guess_previous
from grapevine.integrator import (
    grapevine_velocity_verlet,
    GrapevineIntegratorState,
    GuessInputs,
)


class GrapeNUTSState(NamedTuple):
    """State of the grapevine sampler.

    `guess` is the solution found at `position`, carried so that the next
    trajectory can start from it rather than from the default guess. See
    `build_kernel`.
    """

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    solver_info: ArrayTree
    guess: ArrayTree


def no_solver_info(solution: ArrayTree) -> None:
    """Record nothing about the solver. The default `solver_info_fn`."""
    del solution
    return None


def carry_solution(solution: ArrayTree) -> ArrayTree:
    return solution


def _all_finite(tree: ArrayTree) -> Array:
    """Whether every leaf of a tree is finite, as a traced boolean."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.bool_(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def init(
    position: ArrayTree,
    logdensity_fn: Callable,
    default_guess: ArrayTree,
    *,
    solver_info_fn: Callable = no_solver_info,
):
    """Initialise the GrapeNUTS sampler."""
    (logdensity, solution), logdensity_grad = jax.value_and_grad(
        logdensity_fn, has_aux=True
    )(position, guess=default_guess)
    return GrapeNUTSState(
        position,
        logdensity,
        logdensity_grad,
        solver_info_fn(solution),
        solution,
    )


def build_kernel(
    default_guess: ArrayTree,
    integrator: Callable = grapevine_velocity_verlet,
    divergence_threshold: int = 1000,
    *,
    guess_fn: Callable = guess_previous,
    solver_info_fn: Callable = no_solver_info,
    carry_fn: Callable = carry_solution,
):
    """Get a GrapeNUTS kernel.

    Inspired by [blackjax.mcmc.nuts.build_kernel](https://github.com/blackjax-devs/blackjax/blob/b107f9fd60cfc1261a5ce35690b1d0f141041c07/blackjax/mcmc/nuts.py#L77).

    :param default_guess: the guess to fall back on when the previous
        trajectory's solution cannot be used

    :param integrator: a grapevine-style symplectic integrator, e.g. grapevine.grapevine_velocity_verlet

    :param divergence_threshold: A number that defines what counts as a divergent transition

    :param guess_fn: how to turn the previous solution into the next guess

    :param solver_info_fn: what to record about each iteration's solve

    :param carry_fn: how to turn a draw's solution into the guess that starts
        the next trajectory. The default passes the solution on unchanged.

    """

    def kernel(
        rng_key: PRNGKey,
        state: GrapeNUTSState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: MetricTypes,
        max_num_doublings: int = 10,
    ) -> tuple[GrapeNUTSState, NUTSInfo]:
        """Generate a new sample with the GrapeNUTS kernel."""

        metric = default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(
            logdensity_fn, metric.kinetic_energy, guess_fn
        )
        proposal_generator = iterative_nuts_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            metric.check_turning,
            max_num_doublings,
            divergence_threshold,
        )
        key_momentum, key_integrator = jax.random.split(rng_key, 2)
        position, logdensity, logdensity_grad, _, guess = state
        momentum = metric.sample_momentum(key_momentum, position)
        # Start the trajectory from the last solution rather than from
        # `default_guess`. The first step of a trajectory is otherwise the
        # only one that throws the grapevine away, and it is the one that
        # costs the most: on enzax's glycolysis model it took 100.7 ms against
        # 6.2 ms for every other step, because the fixed default was 276% away
        # from the root at a posterior draw.
        #
        # The fallback is not optional. A solver that fails returns something
        # non-finite, and starting the next solve there fails again, so
        # without it one bad solve would poison every trajectory that follows.
        # Resetting each trajectory used to provide that guarantee for free.
        usable = _all_finite(guess)
        guess = jax.tree.map(
            lambda carried, fallback: jnp.where(usable, carried, fallback),
            carry_fn(guess),
            default_guess,
        )
        integrator_state = GrapevineIntegratorState(
            position,
            momentum,
            logdensity,
            logdensity_grad,
            # `is_default` skips `guess_fn`, which needs a position to
            # extrapolate from. The carried solution has one -- this state's
            # own position -- so the heuristic applies across the trajectory
            # boundary too, and only the fallback goes in unextrapolated.
            GuessInputs(guess, position, ~usable),
        )
        proposal, info = proposal_generator(
            key_integrator,
            integrator_state,
            step_size,
        )
        proposal = GrapeNUTSState(
            proposal.position,
            proposal.logdensity,
            proposal.logdensity_grad,
            solver_info_fn(proposal.guess_inputs.solution),
            proposal.guess_inputs.solution,
        )
        return proposal, info

    return kernel


def get_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: MetricTypes,
    default_guess: Array,
    *,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
    integrator: Callable = grapevine_velocity_verlet,
    guess_fn: Callable = guess_previous,
    solver_info_fn: Callable = no_solver_info,
    carry_fn: Callable = carry_solution,
) -> SamplingAlgorithm:
    kernel = build_kernel(
        default_guess,
        integrator,
        divergence_threshold,
        guess_fn=guess_fn,
        solver_info_fn=solver_info_fn,
        carry_fn=carry_fn,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(
            position,
            logdensity_fn,
            default_guess,
            solver_info_fn=solver_info_fn,
        )

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            max_num_doublings,
        )

    return SamplingAlgorithm(init_fn, step_fn)


grapenuts_sampler = GenerateSamplingAPI(get_api, init, build_kernel)
