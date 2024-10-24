"""An example comparing GrapeNUTS and NUTS on a simple problem.

The problem is taken from the Stan documentation: <https://mc-stan.org/docs/stan-users-guide/algebraic-equations.html#coding-an-algebraic-system>

To formulate this situation as a statistical modelling problem, there is a function `fn` that takes in a state (`y`) and some parameters (`args`) and returns the quantities that should be zero.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by first choosing some true parameter values (see dictionary `TRUE_PARAMS`), and then simulating some measurements of c_m1_int and c_m2_int using these parameters: see function `simulate` for how this works. Then the log posterior is sampled using NUTS and GrapeNUTS, and the times are printed.

"""

from collections import OrderedDict
from functools import partial
import timeit

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from blackjax import nuts
from blackjax import window_adaptation as nuts_window_adaptation
from blackjax.util import run_inference_algorithm
from jax.scipy.stats import norm

from grapevine import run_grapenuts

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)

SEED = 1234
SD = 0.05
TRUE_PARAMS = OrderedDict(theta=jnp.array([3.0, 6.0]))
DEFAULT_GUESS = jnp.array([1.0, 1.0])


@eqx.filter_jit
def fn(y, args):
    y1, y2 = y
    theta1, theta2 = args
    return jnp.array([y1 - theta1, y1 * y2 - theta2])


solver = optx.Newton(rtol=1e-9, atol=1e-9)


@eqx.filter_jit
def joint_logdensity_grapenuts(params, obs, guess):
    theta = params["theta"]
    sol = optx.root_find(fn, solver, guess, args=theta)
    log_prior = jnp.sum(norm.logpdf(theta, loc=TRUE_PARAMS["theta"], scale=0.1).sum())
    log_likelihood = norm.logpdf(obs, loc=sol.value, scale=SD).sum()
    return log_prior + log_likelihood, sol.value


@eqx.filter_jit
def joint_logdensity_nuts(params, obs):
    ld, _ = joint_logdensity_grapenuts(params, obs, DEFAULT_GUESS)
    return ld


@eqx.filter_jit
def simulate(key, params, guess):
    theta = params["theta"]
    sol = optx.root_find(fn, solver, guess, args=theta)
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * SD
    )


def main():
    key = jax.random.key(SEED)
    key, sim_key = jax.random.split(key)
    _, sim = simulate(sim_key, TRUE_PARAMS, DEFAULT_GUESS)
    posterior_logdensity_gn = partial(joint_logdensity_grapenuts, obs=sim)
    posterior_logdensity_nuts = partial(joint_logdensity_nuts, obs=sim)
    key, grapenuts_key = jax.random.split(key)
    key, nuts_key_warmup = jax.random.split(key)
    key, nuts_key_sampling = jax.random.split(key)

    def run_grapenuts_example():
        return run_grapenuts(
            posterior_logdensity_gn,
            grapenuts_key,
            init_parameters=TRUE_PARAMS,
            default_guess=DEFAULT_GUESS,
            num_warmup=1000,
            num_samples=1000,
            initial_step_size=0.0001,
            max_num_doublings=10,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=0.95,
            progress_bar=False,
        )

    def run_nuts_example():
        warmup = nuts_window_adaptation(
            nuts,
            posterior_logdensity_nuts,
            progress_bar=False,
            initial_step_size=0.0001,
            max_num_doublings=10,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=0.95,
        )
        (initial_state, tuned_parameters), _ = warmup.run(
            nuts_key_warmup,
            TRUE_PARAMS,
            num_steps=1000,  #  type: ignore
        )
        kernel = nuts(posterior_logdensity_nuts, **tuned_parameters)
        return run_inference_algorithm(
            nuts_key_sampling,
            kernel,
            1000,
            initial_state,
        )

    # timers
    _ = run_grapenuts_example()  # run once for jitting
    time_grapenuts = timeit.timeit(run_grapenuts_example, number=5)  #  type: ignore
    _ = run_nuts_example()  # run once for jitting
    time_nuts = timeit.timeit(run_nuts_example, number=5)  #  type: ignore

    # print results
    print(f"Runtime for grapenuts: {round(time_grapenuts, 4)}")
    print(f"Runtime for nuts: {round(time_nuts, 4)}")


if __name__ == "__main__":
    main()
