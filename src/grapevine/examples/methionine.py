"""A large-ish steady kinetic model."""

from functools import partial

import diffrax
import jax
from jax.flatten_util import ravel_pytree
import lineax as lx
from enzax.examples import methionine
from enzax.statistical_modelling import (
    enzax_log_likelihood,
    enzax_prior_logdensity,
    prior_from_truth,
)
from jax import numpy as jnp

from grapevine.heuristics import (
    guess_implicit,
    guess_implicit_cg,
    guess_previous,
)
import optimistix as optx

jax.config.update("jax_enable_x64", True)

DEFAULT_GUESS = jnp.full(5, 0.01)
ROOT_FINDER = optx.Newton(rtol=1e-9, atol=1e-9)
ODE_SOLVER = diffrax.Kvaerno5()
ERROR_SD = 0.05
PARAM_SD = 0.02
TRUE_PARAMS = methionine.parameters
PRIOR = prior_from_truth(TRUE_PARAMS, PARAM_SD)  # pyright: ignore


def get_default_guess(*args):
    return DEFAULT_GUESS


def ode_solve(guess, params):
    term = diffrax.ODETerm(methionine.model)
    controller = diffrax.PIDController(
        pcoeff=0.1,
        icoeff=0.3,
        rtol=1e-11,
        atol=1e-11,
    )
    cond_fn = diffrax.steady_state_event()
    event = diffrax.Event(cond_fn)
    adjoint = diffrax.ImplicitAdjoint(
        linear_solver=lx.AutoLinearSolver(well_posed=False)
    )
    sol = diffrax.diffeqsolve(
        terms=term,
        solver=ODE_SOLVER,
        t0=jnp.array(0.0),
        t1=jnp.array(1000.0),
        dt0=jnp.array(0.000001),
        y0=guess,
        max_steps=None,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
        args=params,
    )
    if sol.ys is not None:
        return sol.ys[0], sol.stats["num_steps"]
    else:
        raise ValueError("No steady state found.")


@partial(jax.jit, static_argnames="gfunc")
def joint_logdensity(params, obs, guess_info, gfunc):
    log_prior = enzax_prior_logdensity(params, PRIOR)
    last_solution, _, previous_steps = guess_info
    use_default = jnp.isclose(last_solution, DEFAULT_GUESS).all()
    guess = jax.lax.cond(
        use_default, get_default_guess, gfunc, guess_info, params
    )
    steady, steps_here = ode_solve(guess, params)
    steps = previous_steps + steps_here
    conc_hat = methionine.model.get_conc(steady, params["log_conc_unbalanced"])
    flat_log_enzyme, _ = ravel_pytree(params["log_enzyme"])
    enz_hat = jnp.exp(jnp.array(flat_log_enzyme))
    flux_hat = methionine.model.flux(steady, params)
    conc_msts, enz_msts, flux_msts = obs
    conc_err = jnp.full_like(conc_hat, ERROR_SD)
    flux_err = jnp.full_like(flux_hat, ERROR_SD)
    enz_err = jnp.full_like(enz_hat, ERROR_SD)
    log_likelihood = enzax_log_likelihood(
        (conc_hat, conc_msts, conc_err),
        (enz_hat, enz_msts, enz_err),
        (flux_hat, flux_msts, flux_err),
    )
    return log_prior + log_likelihood, (steady, params, steps)


def simulate(key, params, guess):
    key_conc, key_enz, key_flux = jax.random.split(key, num=3)
    sol, _ = ode_solve(guess, params)
    conc = methionine.model.get_conc(sol, params["log_conc_unbalanced"])
    true_flux = methionine.model.flux(sol, methionine.parameters)
    true_log_enz_flat, _ = ravel_pytree(params["log_enzyme"])
    conc_err = jnp.full_like(conc, ERROR_SD)
    flux_err = jnp.full_like(true_flux, ERROR_SD)
    enz_err = jnp.full_like(true_log_enz_flat, ERROR_SD)
    return sol, (
        jnp.exp(jnp.log(conc) + jax.random.normal(key_conc) * conc_err),
        jnp.exp(true_log_enz_flat + jax.random.normal(key_enz) * enz_err),
        true_flux + jax.random.normal(key_flux) * flux_err,
    )


joint_logdensity_guess_default = partial(
    joint_logdensity, gfunc=lambda g, p: DEFAULT_GUESS
)
joint_logdensity_guess_previous = partial(
    joint_logdensity, gfunc=guess_previous
)
joint_logdensity_guess_implicit = partial(
    joint_logdensity,
    gfunc=partial(guess_implicit, target_function=methionine.model.dcdt),
)
joint_logdensity_guess_implicit_cg = partial(
    joint_logdensity,
    gfunc=partial(guess_implicit_cg, target_function=methionine.model.dcdt),
)
