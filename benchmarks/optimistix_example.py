"""An example comparing GrapeNUTS and NUTS on a representative problem.

This is supposed to be a complete example, mirroring how the grapevine method is used in practice.  


"""

from functools import partial
import timeit

from blackjax.mcmc.nuts import NUTSInfo
from blackjax.util import run_inference_algorithm
from grapevine.grapenuts import GrapeNUTSState
import jax
import jax.numpy as jnp
import optimistix as optx

from blackjax import nuts
from blackjax import window_adaptation as nuts_window_adaptation
from jax.scipy.stats import norm

from grapevine import run_grapenuts, grapenuts_sampler

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)

SEED = 1234
SD = 0.05
TRUE_PARAMS = {
    "log_km_s": 2.0,
    "log_km_p": 3.0,
    "log_vmax": -1.0,
    "log_k_eq": 5.0,
    "log_s1": 2.0,
    "log_s2": 2.9,
    "log_s3": 0.9,
    "log_s4": 0.1,
}
TRUE_PARAMS_ARR = jnp.array(list(TRUE_PARAMS.values()))
DEFAULT_GUESS = jnp.array([0.01, 0.01, 0.01, 0.01])

# hack the timeit module to not destroy the timed function's return value
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


def rmm(p, km_s, km_p, vmax, k_eq, s):
    num = vmax * (s - p / k_eq) / km_s
    denom = 1 + s / km_s + p / km_p
    return num / denom


def fn(y, args):
    p1, p2, p3, p4 = y
    km_s, km_p, vmax, k_eq, s1, s2, s3, s4 = args
    v1 = rmm(p1, km_s, km_p, vmax, k_eq, s1)
    v2 = rmm(p2, km_s, km_p, vmax, k_eq, s2)
    v3 = rmm(p3, km_s, km_p, vmax, k_eq, s3)
    v4 = rmm(p4, km_s, km_p, vmax, k_eq, s4)
    return jnp.array([v1, v2, v3, v4])


solver = optx.Newton(rtol=1e-8, atol=1e-8)


def grapenuts_state_from_nuts_state(nuts_state, guess):
    position, logdensity, logdensity_grad = nuts_state
    return GrapeNUTSState(position, logdensity, logdensity_grad, guess)


def joint_logdensity_grapenuts(params, obs, guess):
    sol = optx.root_find(fn, solver, guess, args=jnp.exp(params))
    log_prior = norm.logpdf(
        params,
        loc=TRUE_PARAMS_ARR,
        scale=jnp.full(params.shape, 1),
    ).sum()
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(sol.value), scale=jnp.full(obs.shape, SD)
    ).sum()
    return log_prior + log_likelihood, sol.value


def joint_logdensity_nuts(params, obs):
    sol = optx.root_find(fn, solver, DEFAULT_GUESS, args=jnp.exp(params))
    log_prior = norm.logpdf(
        params,
        loc=TRUE_PARAMS_ARR,
        scale=jnp.full(params.shape, 1),
    ).sum()
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(sol.value), scale=jnp.full(obs.shape, SD)
    ).sum()
    return log_prior + log_likelihood


def simulate(key, params, guess):
    sol = optx.root_find(fn, solver, guess, args=jnp.exp(params))
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * SD
    )


def main():
    key = jax.random.key(SEED)
    key, sim_key = jax.random.split(key)
    true_p, sim = simulate(sim_key, TRUE_PARAMS_ARR, DEFAULT_GUESS)
    print("True substrate concs: " + str(true_p))
    print("Simulated measurements: " + str(sim))
    posterior_logdensity_gn = partial(joint_logdensity_grapenuts, obs=sim)
    posterior_logdensity_nuts = partial(joint_logdensity_nuts, obs=sim)
    key, grapenuts_key = jax.random.split(key)
    key, nuts_key_warmup = jax.random.split(key)
    key, nuts_key_sampling = jax.random.split(key)

    def run_grapenuts_example():
        return run_grapenuts(
            posterior_logdensity_gn,
            key,
            init_parameters=(TRUE_PARAMS_ARR),
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
            TRUE_PARAMS_ARR,
            num_steps=1000,  #  type: ignore
        )
        kernel = nuts(posterior_logdensity_nuts, **tuned_parameters)
        return run_inference_algorithm(
            nuts_key_sampling,
            kernel,
            1000,
            initial_state,
        )

    time_grapenuts, (state_grapenuts, _) = timeit.timeit(
        run_grapenuts_example,
        number=1,
    )  #  type: ignore
    time_nuts, (_, (state_nuts, _)) = timeit.timeit(
        run_nuts_example,
        number=1,
    )  #  type: ignore
    __import__("pdb").set_trace()
    print("True param vals: " + str(TRUE_PARAMS_ARR))
    print("GrapeNUTS quantiles:")
    print(
        jnp.quantile(
            state_grapenuts.position,
            jnp.array([0.01, 0.5, 0.99]),
            axis=0,
        ).round(4)
    )
    print("NUTS quantiles:")
    print(
        jnp.quantile(
            state_nuts.position,
            jnp.array([0.01, 0.5, 0.99]),
            axis=0,
        ).round(4)
    )
    print(f"Runtime for grapenuts: {round(time_grapenuts, 4)}")
    print(f"Runtime for nuts: {round(time_nuts, 4)}")


if __name__ == "__main__":
    main()
