# grapevine
[![Tests](https://github.com/dtu-qmcm/grapevine/actions/workflows/run_tests.yml/badge.svg)](https://github.com/dtu-qmcm/grapevine/actions/workflows/run_tests.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Supported Python versions: 3.12 and newer](https://img.shields.io/badge/python->=3.12-blue.svg)](https://www.python.org/)

JAX/Blackjax implementation of the grapevine method for reusing the solutions of guessing problems embedded in Hamiltonian trajectories.

The grapevine method can dramatically speed up MCMC for statistical with embedded equation solving problems.

## Installation

```sh
pip install grapevine-mcmc
```

## Usage

First make a suitable log density function.

This function should have two arguments: a set of parameters (a [Pytree](https://jax.readthedocs.io/en/latest/pytrees.html)) and a guess (also a Pytree). It should return the log density of these parameters (a number) and the solution it found, which grapevine feeds back in as the next guess. It should also be generally compatible with JAX, and will probalbly involve some differentiable numerical solving, for example using [optimistix](https://docs.kidger.site/optimistix/).

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

grapevine provides the sampler; [blackjax-utils](https://github.com/teddygroves/blackjax-utils) runs it, handling chains, initial jitter and position flattening. `grapenuts` returns the two hooks it needs.

```python
from blackjax_utils import run_sampler
from grapevine import grapenuts

states, info = run_sampler(
    key=jax.random.key(1234),
    log_posterior=posterior_logdensity,
    init_params=jnp.array(0.0),
    init_sd=0.01,
    n_chain=4,
    n_warmup=200,
    n_sample=200,
    warmup_options=dict(initial_step_size=0.01),
    sampler=grapenuts(default_guess=jnp.array(0.01)),
)
jnp.quantile(states.position, jnp.array([0.01, 0.5, 0.99]))
# Array([-1.26712677,  0.12950684,  0.93903677], dtype=float64)
```

## Choosing a guess

`grapenuts` takes a `guess_fn`, which turns the previous problem's solution into
the next guess. The default, `guess_previous`, reuses the solution as it is.
`guess_implicit` instead takes an Euler step from it, using the solution's
jacobian with respect to the parameters, and `guess_implicit_cg` does the same
without materialising any jacobians:

```python
from functools import partial
from grapevine import grapenuts, guess_implicit

sampler = grapenuts(
    default_guess=jnp.array(0.01),
    guess_fn=partial(guess_implicit, target_function=fn),
)
```

A `guess_fn` is called as `guess_fn(guess_inputs, position)`, where
`guess_inputs` has the previous solution, the position at which it was found,
and a flag marking the start of a trajectory. Whatever it returns is what the
log density gets as its `guess`, so a log density that accumulates diagnostics
across a trajectory can carry them alongside the solution.

Pass `solver_info_fn` to record something about each iteration's solve, such as
the number of steps the solver took. It defaults to recording nothing.

# How to run the benchmarks

1. Install [uv](https://docs.astral.sh/uv/)
2. Run these commands

```sh
uv run benchmarks/methionine.py
uv run benchmarks/linear.py
uv run benchmarks/test_functions.py
uv run benchmarks/adversarial.py
uv run benchmarks/trajectory.py
uv run benchmarks/analyse_results.py
```

Alternatively, run this convenient shell script:

```sh
bash run_all_benchmarks.sh
```
