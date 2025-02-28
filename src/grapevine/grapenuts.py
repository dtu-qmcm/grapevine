"""The NUTS algorithm sped up for guess problems using the grapevine method."""

from typing import Callable, NamedTuple
import jax
from blackjax.types import Array, ArrayTree, ArrayLikeTree, PRNGKey
from blackjax import GenerateSamplingAPI
from blackjax.base import SamplingAlgorithm

from blackjax.mcmc.nuts import NUTSInfo, iterative_nuts_proposal
from blackjax.mcmc.metrics import MetricTypes, default_metric

from grapevine.integrator import (
    grapevine_velocity_verlet,
    GrapevineIntegratorState,
)


class GrapeNUTSState(NamedTuple):
    """State of the grapevine sampler."""

    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    guess: ArrayTree
    n_newton_steps: int


def init(
    position: ArrayTree,
    logdensity_fn: Callable,
    default_guess: ArrayTree,
):
    """Initialise the GrapeNUTS sampler."""
    (logdensity, _), logdensity_grad = jax.value_and_grad(logdensity_fn, has_aux=True)(
        position, guess=default_guess
    )
    return GrapeNUTSState(position, logdensity, logdensity_grad, default_guess, 0)


def build_kernel(
    default_guess: ArrayTree,
    integrator: Callable = grapevine_velocity_verlet,
    divergence_threshold: int = 1000,
):
    """Get a GrapeNUTS kernel.

    Inspired by [blackjax.mcmc.nuts.build_kernel](https://github.com/blackjax-devs/blackjax/blob/b107f9fd60cfc1261a5ce35690b1d0f141041c07/blackjax/mcmc/nuts.py#L77).

    :param default_guess: a default guess for the solving problem, used at the start of each trajectory

    :param integrator: a grapevine-style symplectic integrator, e.g. grapevine.grapevine_velocity_verlet

    :param divergence_threshold: A number that defines what counts as a divergent transition

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
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator = iterative_nuts_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            metric.check_turning,
            max_num_doublings,
            divergence_threshold,
        )
        key_momentum, key_integrator = jax.random.split(rng_key, 2)
        position, logdensity, logdensity_grad, guess, _ = state
        momentum = metric.sample_momentum(key_momentum, position)
        integrator_state = GrapevineIntegratorState(
            position,
            momentum,
            logdensity,
            logdensity_grad,
            guess,
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
            default_guess,
            proposal.guess[-1],
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
) -> SamplingAlgorithm:
    kernel = build_kernel(
        default_guess,
        integrator=grapevine_velocity_verlet,
        divergence_threshold=divergence_threshold,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn, default_guess)

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
