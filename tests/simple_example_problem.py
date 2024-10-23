"""Simple example problems for other tests to use."""
from functools import partial

from jax.scipy.stats import norm
from jax import numpy as jnp

import optimistix as optx

solver = optx.Newton(rtol=1e-8, atol=1e-8)
default_guess = jnp.array(0.01)
obs = jnp.array(0.7)


def fn(y, args):
    """Equation defining a root-finding problem."""
    a = args
    return y - jnp.tanh(y * a + 1)


def joint_logdensity(a, obs, guess):
    """An example log density."""
    sol = optx.root_find(fn, solver, guess, args=a)
    log_prior = norm.logpdf(a, loc=0.0, scale=1.0)
    log_likelihood = norm.logpdf(obs, loc=sol.value, scale=0.5)
    return log_prior + log_likelihood, sol.value


posterior_logdensity = partial(joint_logdensity, obs=obs)
