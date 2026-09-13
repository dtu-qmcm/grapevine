"""Hooks for running GrapeNUTS with a generic sampler runner."""

from functools import partial
from typing import Callable, NamedTuple

from blackjax.types import ArrayTree

from grapevine.adaptation import grapenuts_window_adaptation
from grapevine.grapenuts import grapenuts_sampler, no_solver_info
from grapevine.heuristics import guess_previous
from grapevine.integrator import grapevine_velocity_verlet


class Sampler(NamedTuple):
    """How to build a sampler's warmup and its sampling step."""

    make_warmup: Callable
    make_kernel: Callable


def grapenuts_warmup(density: Callable, *, default_guess, **kwargs):
    return grapenuts_window_adaptation(
        grapenuts_sampler, density, default_guess, **kwargs
    )


def grapenuts_kernel(density: Callable, *, default_guess, **kwargs):
    return grapenuts_sampler(
        density, default_guess=default_guess, **kwargs
    ).step


def grapenuts(
    default_guess: ArrayTree,
    *,
    integrator: Callable = grapevine_velocity_verlet,
    guess_fn: Callable = guess_previous,
    solver_info_fn: Callable = no_solver_info,
) -> Sampler:
    """Get a GrapeNUTS sampler for a runner such as blackjax_utils.

    :param default_guess: the guess used at the start of each trajectory

    :param guess_fn: how to turn the previous solution into the next guess

    :param solver_info_fn: what to record about each iteration's solve
    """
    bound = dict(
        default_guess=default_guess,
        integrator=integrator,
        guess_fn=guess_fn,
        solver_info_fn=solver_info_fn,
    )
    return Sampler(
        partial(grapenuts_warmup, **bound),
        partial(grapenuts_kernel, **bound),
    )
