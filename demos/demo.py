from functools import partial

import jax
import jax.numpy as jnp
import optimistix as optx

from jax.scipy.stats import norm

from grapevine import run_grapenuts

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)

SEED = 1234
TRUE_PARAMS = jnp.array([-1.0, -1.0])


def fn(y, args):
    """Function defining a root finding problem.

    We want to find y such that c and d are zero.
    """
    a, b = y
    c = jnp.tanh(jnp.sum(b)) - (a * args)
    d = (a * args) ** 2 - jnp.sinh(b + 1)
    return c, d


solver = optx.Newton(rtol=1e-8, atol=1e-8)
default_guess = (jnp.array(0.01), jnp.full((2, 2), 0.01))


def joint_logdensity(params, obs, guess):
    sd = jnp.exp(params[0])
    a = jnp.exp(params[1])
    sol = optx.root_find(fn, solver, guess, args=a)
    log_prior = norm.logpdf(params[0], -1.0, 1.0) + norm.logpdf(params[1], 0.0, 1.0)
    log_likelihood = (
        norm.logpdf(obs[0], loc=sol.value[0], scale=sd)
        + norm.logpdf(obs[1], loc=sol.value[1], scale=sd).sum()
    )
    return log_prior + log_likelihood, sol.value


def simulate(params, guess, key):
    sd = jnp.exp(params[0])
    a = jnp.exp(params[1])
    sol = optx.root_find(fn, solver, guess, args=a)
    key_0, key = jax.random.split(key)
    key_1, key = jax.random.split(key)
    return (
        solval + jax.random.normal(k, shape=solval.shape) * sd
        for solval, k in zip(sol.value, [key_0, key_1])
    )


def main():
    key = jax.random.key(SEED)
    sim_0, sim_1 = simulate(TRUE_PARAMS, default_guess, key)
    posterior_logdensity = partial(joint_logdensity, obs=(sim_0, sim_1))
    samples, info = run_grapenuts(
        posterior_logdensity,
        key,
        init_parameters=TRUE_PARAMS,
        default_guess=default_guess,
        num_warmup=200,
        num_samples=200,
        initial_step_size=0.0001,
        max_num_doublings=10,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=0.95,
    )
    print("1%, 50% and 99% posterior quantiles:")
    print(jnp.quantile(samples.position, jnp.array([0.01, 0.5, 0.99]), axis=0))


if __name__ == "__main__":
    main()
