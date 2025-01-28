"""An example comparing GrapeNUTS and NUTS on a model of the methionine pathway.

See here for details about the model:
<https://pubs.acs.org/doi/10.1021/acssynbio.3c00662>

"""

from functools import partial
from pathlib import Path
import time

import arviz as az
import diffrax
import equinox as eqx
from grapevine.util import run_nuts
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import polars as pl
from jax.scipy.stats import norm
from jax.flatten_util import ravel_pytree
from enzax.examples import methionine
from enzax.kinetic_model import RateEquationModel, get_conc
from enzax.mcmc import ObservationSet
from enzax.steady_state import get_kinetic_model_steady_state

from grapevine import run_grapenuts

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
PARAM_SD = 0.01
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "methionine_pathway.csv"
DEFAULT_GUESS = jnp.full((5,), 0.01)
N_WARMUP = 5
N_SAMPLE = 5
N_TEST = 2
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.95
TRUE_PARAMS = methionine.parameters
TRUE_MODEL = methionine.model
PRIOR_SD = 0.1
ERROR_CONC = 0.03
ERROR_FLUX = 0.05
ERROR_ENZYME = 0.03


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


@eqx.filter_jit
def simulate(key, params, guess):
    model = RateEquationModel(params, methionine.structure)
    true_steady = get_kinetic_model_steady_state(model, guess)
    true_conc = get_conc(
        true_steady,
        TRUE_PARAMS.log_conc_unbalanced,
        methionine.structure,
    )
    true_flux = model.flux(true_steady)
    # simulate observations
    key = jax.random.key(SEED)
    true_log_enz_flat, _ = ravel_pytree(TRUE_PARAMS.log_enzyme)
    key_conc, key_enz, key_flux = jax.random.split(key, num=3)
    obs_conc = jnp.exp(jnp.log(true_conc) + jax.random.normal(key_conc) * ERROR_CONC)
    obs_enzyme = jnp.exp(true_log_enz_flat + jax.random.normal(key_enz) * ERROR_ENZYME)
    obs_flux = true_flux + jax.random.normal(key_flux) * ERROR_FLUX
    obs = ObservationSet(
        conc=obs_conc,
        flux=obs_flux,
        enzyme=obs_enzyme,
        conc_scale=ERROR_CONC,
        flux_scale=ERROR_FLUX,
        enzyme_scale=ERROR_ENZYME,
    )
    return obs


def get_idata(samples, info, coords=None, dims=None) -> az.InferenceData:
    """Get an arviz InferenceData from a grapeNUTS output."""
    sample_dict = {"dgf": jnp.expand_dims(samples.position.dgf, 0)}
    posterior = az.convert_to_inference_data(
        sample_dict,
        group="posterior",
        coords=coords,
        dims=dims,
    )
    sample_stats = az.convert_to_inference_data(
        {
            "diverging": info.is_divergent,
            "energy": info.energy,
        },
        group="sample_stats",
    )
    idata = az.concat(posterior, sample_stats)
    assert idata is not None, "idata should not be None!"
    return idata


def time_run(run_fn):
    """Time run_fn and check how many effective samples it generates."""
    _ = run_fn()  # dummy run for jit compiling
    start = time.time()
    out = run_fn()
    _ = out[0].position.log_substrate_km["AHC1"].block_until_ready()
    end = time.time()
    idata = get_idata(*out)
    runtime = end - start
    ess = az.ess(idata.sample_stats["energy"])  # type: ignore
    neff = np.sum([ess[v].values.sum() for v in ess.data_vars]).item()
    return {"time": runtime, "neff": neff}


def compare_single(key: jax.Array, params) -> dict:
    sim_key, grapenuts_key, nuts_key = jax.random.split(key, 3)
    # simulate
    sim = simulate(sim_key, params, DEFAULT_GUESS)
    flat_true_params, _ = ravel_pytree(params)
    # posteriors
    posterior_logdensity_gn = partial(
        joint_logdensity_grapenuts,
        obs=sim,
        prior_mean=flat_true_params,
        prior_sd=PRIOR_SD,
    )
    posterior_logdensity_nuts = partial(
        joint_logdensity_nuts,
        obs=sim,
        prior_mean=flat_true_params,
        prior_sd=PRIOR_SD,
    )
    run_fn_gn = eqx.filter_jit(
        partial(
            run_grapenuts,
            logdensity_fn=posterior_logdensity_gn,
            rng_key=grapenuts_key,
            init_parameters=params,
            default_guess=DEFAULT_GUESS,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLE,
            initial_step_size=INIT_STEPSIZE,
            max_num_doublings=MAX_TREEDEPTH,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=TARGET_ACCEPT,
            progress_bar=False,
        )
    )
    run_fn_nuts = eqx.filter_jit(
        partial(
            run_nuts,
            logdensity_fn=posterior_logdensity_nuts,
            rng_key=nuts_key,
            init_parameters=params,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLE,
            initial_step_size=INIT_STEPSIZE,
            max_num_doublings=MAX_TREEDEPTH,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=TARGET_ACCEPT,
            progress_bar=False,
        )
    )
    # results
    result_gn = time_run(run_fn_gn)
    result_nuts = time_run(run_fn_nuts)
    perf_gn = result_gn["neff"] / result_gn["time"]
    perf_nuts = result_nuts["neff"] / result_nuts["time"]
    perf_ratio = perf_gn / perf_nuts
    return {
        "neff_n": result_nuts["neff"],
        "neff_gn": result_gn["neff"],
        "time_n": result_nuts["time"],
        "time_gn": result_gn["time"],
        "perf_n": perf_nuts,
        "perf_gn": perf_gn,
        "perf_ratio": perf_ratio,
    }


def generate_random_params(key, params_in, sd):
    flat, unravel_func = ravel_pytree(params_in)
    return unravel_func(flat + jax.random.normal(key, flat.shape) * sd)


def run_comparison(n_test: int):
    key = jax.random.key(SEED)
    keys = jax.random.split(key, n_test)
    results = []
    for i, keyi in enumerate(keys):
        compare_key, param_key = jax.random.split(keyi)
        params = generate_random_params(param_key, TRUE_PARAMS, PARAM_SD)
        result = compare_single(compare_key, params)
        result["rep"] = i
        results.append(result)
        print(results)
    return pl.from_records(results)


def main():
    results = run_comparison(n_test=N_TEST)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    print("Mean results:")
    results.write_csv(CSV_OUTPUT_FILE)
    print(results)


if __name__ == "__main__":
    main()
