"""An example comparing GrapeNUTS and NUTS on a model of the methionine pathway.

See here for details about the model:
<https://pubs.acs.org/doi/10.1021/acssynbio.3c00662>

"""

import time
from functools import partial
from pathlib import Path

import arviz as az
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import polars as pl
from blackjax import nuts
from blackjax import window_adaptation as nuts_window_adaptation
from blackjax.util import run_inference_algorithm
from jax.scipy.stats import norm
from jax.flatten_util import ravel_pytree
from enzax.examples import methionine
from enzax.kinetic_model import RateEquationModel, get_conc
from enzax.mcmc import (
    ObservationSet,
    get_idata,
)
from enzax.steady_state import get_kinetic_model_steady_state

from grapevine import run_grapenuts

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
SD = 0.05
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "methionine_pathway.csv"
DEFAULT_GUESS = jnp.full((5,), 0.01)
N_WARMUP = 100
N_SAMPLE = 100
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.95
TRUE_PARAMS = methionine.parameters
TRUE_MODEL = methionine.model
PRIOR_SD = 0.1


ode_solver = diffrax.Tsit5()
steady_state_cond = diffrax.steady_state_event()
steady_state_event = diffrax.Event(steady_state_cond)
adjoint = diffrax.ImplicitAdjoint(
    linear_solver=lineax.AutoLinearSolver(well_posed=False)
)
controller = diffrax.PIDController(pcoeff=0.1, icoeff=0.3, rtol=1e-9, atol=1e-9)


@eqx.filter_jit
def joint_logdensity_grapenuts(params, obs, prior_mean, prior_sd, guess):
    # find the steady state concentration and flux
    model = RateEquationModel(params, methionine.structure)
    steady = get_kinetic_model_steady_state(model, guess)
    conc = get_conc(steady, params.log_conc_unbalanced, methionine.structure)
    flux = model.flux(steady)
    # prior
    flat_params, _ = ravel_pytree(params)
    log_prior = norm.logpdf(flat_params, loc=prior_mean, scale=prior_sd).sum()
    # likelihood
    flat_log_enzyme, _ = ravel_pytree(params.log_enzyme)
    log_likelihood = (
        norm.logpdf(jnp.log(obs.conc), jnp.log(conc), obs.conc_scale).sum()
        + norm.logpdf(jnp.log(obs.enzyme), flat_log_enzyme, obs.enzyme_scale).sum()
        + norm.logpdf(obs.flux, flux, obs.flux_scale).sum()
    )
    ## integrate above and bottom
    return log_prior + log_likelihood, steady


@eqx.filter_jit
def joint_logdensity_nuts(params, obs, prior_mean, prior_sd):
    ld, _ = joint_logdensity_grapenuts(params, obs, prior_mean, prior_sd, DEFAULT_GUESS)
    return ld


def time_grapenuts_run(key, posterior_logdensity, true_params, default_guess):
    print("here")
    run_fn = partial(
        run_grapenuts,
        logdensity_fn=posterior_logdensity,
        rng_key=key,
        init_parameters=true_params,
        default_guess=default_guess,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLE,
        initial_step_size=INIT_STEPSIZE,
        max_num_doublings=MAX_TREEDEPTH,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=TARGET_ACCEPT,
        progress_bar=True,
    )
    start = time.time()
    out = run_fn()
    end = time.time()
    runtime = end - start
    idata = get_idata(*out)
    neff = az.ess(idata.sample_stats)["energy"].item()  # type: ignore
    return {
        "algorithm": "grapeNUTS",
        "time": runtime,
        "neff": neff,
    }


def time_nuts_run(key, posterior_logdensity, true_params):
    print("here")
    key_warmup, key_sampling = jax.random.split(key)
    warmup = nuts_window_adaptation(
        nuts,
        posterior_logdensity,
        progress_bar=True,
        initial_step_size=INIT_STEPSIZE,
        max_num_doublings=MAX_TREEDEPTH,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=TARGET_ACCEPT,
    )

    def run_fn():
        (initial_state, tuned_parameters), _ = warmup.run(
            key_warmup,
            true_params,
            num_steps=N_WARMUP,  #  type: ignore
        )
        kernel = nuts(posterior_logdensity, **tuned_parameters)
        (_, out) = run_inference_algorithm(
            key_sampling,
            kernel,
            N_SAMPLE,
            initial_state,
        )
        return out

    start = time.time()
    out = run_fn()
    end = time.time()
    runtime = end - start
    idata = get_idata(*out)
    neff = az.ess(idata.sample_stats)["energy"].item()  # type: ignore
    return {
        "algorithm": "NUTS",
        "time": runtime,
        "neff": neff,
    }


@eqx.filter_jit
def simulate(key, model, guess):
    true_steady = get_kinetic_model_steady_state(model, guess)
    true_conc = get_conc(
        true_steady,
        TRUE_PARAMS.log_conc_unbalanced,
        methionine.structure,
    )
    true_flux = model.flux(true_steady)
    # simulate observations
    error_conc = 0.03
    error_flux = 0.05
    error_enzyme = 0.03
    key = jax.random.key(SEED)
    true_log_enz_flat, _ = ravel_pytree(TRUE_PARAMS.log_enzyme)
    key_conc, key_enz, key_flux, key_nuts = jax.random.split(key, num=4)
    obs_conc = jnp.exp(jnp.log(true_conc) + jax.random.normal(key_conc) * error_conc)
    obs_enzyme = jnp.exp(true_log_enz_flat + jax.random.normal(key_enz) * error_enzyme)
    obs_flux = true_flux + jax.random.normal(key_flux) * error_conc
    obs = ObservationSet(
        conc=obs_conc,
        flux=obs_flux,
        enzyme=obs_enzyme,
        conc_scale=error_conc,
        flux_scale=error_flux,
        enzyme_scale=error_enzyme,
    )
    return obs


def run_single_comparison(
    key: jax.Array, true_params: dict
) -> tuple[jax.Array, dict, dict]:
    key, sim_key = jax.random.split(key)
    key, grapenuts_key = jax.random.split(key)
    # key, grapenuts_key_jac = jax.random.split(key)
    key, nuts_key = jax.random.split(key)
    default_guess = DEFAULT_GUESS
    # simulate
    sim = simulate(sim_key, TRUE_MODEL, default_guess)
    # parameters
    flat_true_params, _ = ravel_pytree(TRUE_PARAMS)
    # posteriors
    posterior_logdensity_gn = partial(
        joint_logdensity_grapenuts,
        obs=sim,
        prior_mean=flat_true_params,
        prior_sd=PRIOR_SD,
    )
    posterior_logdensity_nuts = partial(
        joint_logdensity_nuts, obs=sim, prior_mean=flat_true_params, prior_sd=PRIOR_SD
    )
    # results
    result_gn = time_grapenuts_run(
        grapenuts_key,
        posterior_logdensity_gn,
        TRUE_PARAMS,
        default_guess,
    )
    result_nuts = time_nuts_run(
        nuts_key,
        posterior_logdensity_nuts,
        TRUE_PARAMS,
    )
    return key, result_gn, result_nuts


def run_comparison(n_test: int):
    key = jax.random.key(SEED)
    results = []
    for i in range(n_test):
        key, result_gn, result_nuts = run_single_comparison(
            key,
            TRUE_PARAMS,
        )
        result_gn["repeat"] = i
        # result_gn_jac["repeat"] = i
        result_nuts["repeat"] = i
        results += [
            result_gn,
            # result_gn_jac,
            result_nuts,
        ]
    return pl.from_records(results).with_columns(
        (pl.col("neff") / pl.col("time")).alias("neff/s")
    )


def main():
    results = run_comparison(n_test=1)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    print("Mean results:")
    results.write_csv(CSV_OUTPUT_FILE)
    print(results)


if __name__ == "__main__":
    main()
