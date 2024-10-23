# grapevine

JAX/Blackjax implementation of the grapevine method for reusing the solutions of guessing problems embedded in Hamiltonian trajectories.

The grapevine method can dramatically speed up MCMC for statistical with embedded equation solving problems.

## Installation

```sh
pip install grapevine-mcmc
```

## Usage

First make a suitable log density function.

This function should have two arguments: a set of parameters (a [Pytree](https://jax.readthedocs.io/en/latest/pytrees.html)) and a guess (also a Pytree). It should return the log density of these parameters (a number) and a new guess. It should also be generally compatible with JAX, and will probalbly involve some differentiable numerical solving, for example using [optimistix](https://docs.kidger.site/optimistix/).

Here is a simple example of such a function:

```python
from functools import partial

import jax

from jax.scipy.stats import norm
from jax.scipy.special import expit
from jax import numpy as jnp

import optimistix as optx

# equation solving problems often need 64 bit floats
jax.config.update("jax_enable_x64", True)

solver = optx.Newton(rtol=1e-8, atol=1e-8)
obs = jnp.array(0.7)


def fn(y, args):
    """Equation defining a root-finding problem."""
    a = args
    return y - jnp.tanh(y * expit(a) + 1)


def joint_logdensity(a, obs, guess):
    """An example log density."""
    sol = optx.root_find(fn, solver, guess, args=a)
    log_prior = norm.logpdf(a, loc=0.0, scale=1.0)
    log_likelihood = norm.logpdf(obs, loc=sol.value, scale=0.5)
    return log_prior + log_likelihood, sol.value


posterior_logdensity = partial(joint_logdensity, obs=obs)
posterior_logdensity(a=0.0, guess=0.01)
# (Array(-1.22095095, dtype=float64), Array(0.8952192, dtype=float64))
```

Now you can run MCMC on your model using GrapeNUTS, the grapevine version of the [NUTS](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf) sampler!

```python
from grapevine import run_grapenuts

INITIAL_POSITION = jnp.array(0.0)
DEFAULT_GUESS = jnp.array(0.01)
SEED = 1234

key = jax.random.key(SEED)
samples, info = run_grapenuts(
    logdensity_fn=posterior_logdensity,
    rng_key=key,
    init_parameters=INITIAL_POSITION,
    num_warmup=10,
    num_samples=10,
    default_guess=DEFAULT_GUESS,
    progress_bar=False,
    initial_step_size=0.01,
    max_num_doublings=4,
    is_mass_matrix_diagonal=True,
    target_acceptance_rate=0.8,
)
jnp.quantile(samples.position, jnp.array([0.01, 0.5, 0.99]))
# Array([-1.26712677,  0.12950684,  0.93903677], dtype=float64)
```
