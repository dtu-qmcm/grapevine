"""Window adaptation for GrapeNUTS."""

from types import SimpleNamespace
from typing import Callable

import blackjax
from blackjax.base import AdaptationAlgorithm
from blackjax.types import ArrayTree

from grapevine.grapenuts import no_solver_info
from grapevine.heuristics import guess_previous
from grapevine.integrator import grapevine_velocity_verlet


def grapenuts_window_adaptation(
    algorithm,
    logdensity_fn: Callable,
    default_guess: ArrayTree,
    *,
    integrator: Callable = grapevine_velocity_verlet,
    guess_fn: Callable = guess_previous,
    solver_info_fn: Callable = no_solver_info,
    **kwargs,
) -> AdaptationAlgorithm:
    """Adapt GrapeNUTS's step size and mass matrix.

    Blackjax's window adaptation reaches the algorithm only through
    `algorithm.init(position, logdensity_fn)` and
    `algorithm.build_kernel(integrator)`, so binding grapevine's extra
    arguments into a stand-in is enough to reuse it whole.
    """
    bound = SimpleNamespace(
        init=lambda position, ldf: algorithm.init(
            position, ldf, default_guess, solver_info_fn=solver_info_fn
        ),
        build_kernel=lambda integrator=integrator: algorithm.build_kernel(
            default_guess,
            integrator,
            guess_fn=guess_fn,
            solver_info_fn=solver_info_fn,
        ),
    )
    return blackjax.window_adaptation(
        bound, logdensity_fn, integrator=integrator, **kwargs
    )
