"""Utility functions for benchmarking."""

from functools import partial
import time
from typing import Callable

import arviz as az
import equinox as eqx
import jax
import numpy as np
import polars as pl
from tqdm import tqdm
from dataclasses import dataclass

import jax.numpy as jnp


# from grapevine.examples import linear_pathway
from grapevine.util import run_grapenuts, get_idata


def time_run(run_fn):
    """Time run_fn and check how many effective samples it generates."""
    try:
        _ = run_fn()  # dummy run for jit compiling
        start = time.time()
        out = run_fn()
        _ = next(iter(out[0].position.values())).block_until_ready()
        end = time.time()
        idata = get_idata(*out)
        ess = az.ess(idata.posterior)  # type: ignore
        neff = np.sum([ess[v].values.sum() for v in ess.data_vars]).item()
        n_newton_steps = idata.sample_stats["n_newton_steps"].values.sum()
        runtime = end - start
        divergences = idata.sample_stats["diverging"].sum()
    except Exception as err:
        print(err)
        neff = 0.0
        n_newton_steps = 0
        runtime = 0.0
        divergences = 1
    return {
        "time": runtime,
        "neff": neff,
        "n_newton_steps": n_newton_steps,
        "divergences": divergences,
    }


def compare_single(
    rng_key: jax.Array,
    true_params: dict,
    joint_logdensity_funcs: dict,
    run_grapenuts_kwargs: dict,
    sim_func: Callable,
    default_guess_info: tuple,
) -> pl.DataFrame:
    """Run a single comparison of the different guessing heuristics."""
    sim_key, sample_key = jax.random.split(rng_key)
    # simulate
    _, sim = sim_func(sim_key, true_params, default_guess_info[0])
    # posteriors
    posterior_logdensity_funcs = {
        k: partial(v, obs=sim) for k, v in joint_logdensity_funcs.items()
    }
    results = []
    for k, posterior in posterior_logdensity_funcs.items():
        run_fn = eqx.filter_jit(
            partial(
                run_grapenuts,
                logdensity_fn=posterior,
                rng_key=sample_key,
                init_parameters=true_params,
                default_guess_info=default_guess_info,
                **run_grapenuts_kwargs,
            )
        )
        result = time_run(run_fn)
        result["heuristic"] = k
        results.append(result)
    return pl.from_dicts(results)


def randomise_params(key, baseline_params, sd):
    leaves, treedef = jax.tree.flatten(baseline_params)
    keys = jax.random.split(key, len(leaves))
    keytree = jax.tree.unflatten(treedef, keys)

    def randomise_leaf(leaf, leaf_key):
        return leaf + jax.random.normal(leaf_key, leaf.shape) * sd

    return jax.tree.map(randomise_leaf, baseline_params, keytree)


def run_benchmark(
    random_seed,
    joint_logdensity_funcs,
    baseline_params: dict,
    param_sd: float,
    n_test: int,
    run_grapenuts_kwargs,
    sim_func: Callable,
    default_guess_info: tuple,
):
    key = jax.random.key(random_seed)
    keys = jax.random.split(key, n_test)
    results = []
    for i, keyi in tqdm(enumerate(keys), total=len(keys)):
        compare_key, param_key = jax.random.split(keyi)
        true_params = randomise_params(param_key, baseline_params, param_sd)
        result = compare_single(
            compare_key,
            true_params,
            joint_logdensity_funcs,
            run_grapenuts_kwargs,
            sim_func,
            default_guess_info,
        )
        result = result.with_columns(rep=i)
        results.append(result)
    return pl.concat(results).sort("heuristic", "rep")


@dataclass
class Rosenbrock:
    """Rosenbrock function.

    f(x) = sum_{i=1}^{n-1} [100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Global minimum at (1, 1, ..., 1) with value 0.
    """

    n_dimensions: int
    a: float = 1.0
    b: float = 100.0

    def __call__(self, x):
        x = jnp.atleast_1d(x)
        return jnp.sum(
            self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2
        )


@dataclass
class Sphere:
    """Sphere function (n-dimensional).

    f(x) = sum_{i=1}^{n} x_i^2

    Global minimum at (0, 0, ..., 0) with value 0.
    """

    n_dimensions: int

    def __call__(self, x):
        return jnp.sum(x**2)


@dataclass
class Rastrigin:
    """Rastrigin function (n-dimensional).

    f(x) = 10n + sum_{i=1}^{n} [x_i^2 - 10 * cos(2π * x_i)]

    Global minimum at (0, 0, ..., 0) with value 0.
    Highly multimodal with many local minima.
    """

    n_dimensions: int
    A: float = 10.0

    def __call__(self, x):
        n = self.n_dimensions
        return self.A * n + jnp.sum(x**2 - self.A * jnp.cos(2 * jnp.pi * x))


@dataclass
class Beale:
    """Beale function (2-dimensional).

    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Global minimum at (3, 0.5) with value 0.
    """

    n_dimensions: int = 2  # Fixed at 2D

    def __call__(self, x):
        if len(x) != 2:
            raise ValueError(
                f"Beale function is 2D but got {len(x)} dimensions"
            )
        return (
            (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        )


@dataclass
class Ackley:
    """Ackley function (n-dimensional).

    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum_{i=1}^{n} x_i^2))
           - exp(1/n * sum_{i=1}^{n} cos(2π * x_i)) + 20 + e

    Global minimum at (0, 0, ..., 0) with value 0.
    """

    n_dimensions: int
    a: float = 20.0
    b: float = 0.2
    c: float = 2 * jnp.pi

    def __call__(self, x):
        n = self.n_dimensions
        sum1 = jnp.sum(x**2)
        sum2 = jnp.sum(jnp.cos(self.c * x))

        term1 = -self.a * jnp.exp(-self.b * jnp.sqrt(sum1 / n))
        term2 = -jnp.exp(sum2 / n)

        return term1 + term2 + self.a + jnp.exp(1.0)


@dataclass
class StyblinskiTang:
    """Styblinski-Tang function (n-dimensional).

    f(x) = 0.5 * sum_{i=1}^{n} [x_i^4 - 16x_i^2 + 5x_i]

    Global minimum at (-2.903534, ..., -2.903534) with value -39.16617n.
    """

    n_dimensions: int

    def __call__(self, x):
        return 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x)


@dataclass
class Levy:
    """Levy function (n-dimensional).

    f(x) = sin^2(πw_1) + sum_{i=1}^{n-1} [(w_i-1)^2 * (1+10sin^2(πw_i+1))]
           + (w_n-1)^2 * (1+sin^2(2πw_n))

    where w_i = 1 + (x_i - 1)/4

    Global minimum at (1, 1, ..., 1) with value 0.
    """

    n_dimensions: int

    def __call__(self, x):
        w = 1 + (x - 1) / 4

        term1 = jnp.sin(jnp.pi * w[0]) ** 2

        term2 = jnp.sum(
            (w[:-1] - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * w[:-1] + 1) ** 2)
        )

        term3 = (w[-1] - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * w[-1]) ** 2)

        return term1 + term2 + term3


@dataclass
class Easom:
    """Easom function (2-dimensional).

    f(x,y) = -cos(x) * cos(y) * exp(-((x-π)^2 + (y-π)^2))

    Global minimum at (π, π) with value -1.
    """

    n_dimensions: int = 2  # Fixed at 2D

    def __call__(self, x):
        if len(x) != 2:
            raise ValueError(
                f"Easom function is 2D but got {len(x)} dimensions"
            )
        return (
            -jnp.cos(x[0])
            * jnp.cos(x[1])
            * jnp.exp(-((x[0] - jnp.pi) ** 2 + (x[1] - jnp.pi) ** 2))
        )
