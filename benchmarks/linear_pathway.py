"""An example comparing GrapeNUTS and NUTS on a representative problem.

The problem is a steady kinetic model of a linear pathway with this structure:

    Aext <-r1-> Aint <-r2-> Bint <-r3-> Bext

Reactions r1 and r3 behave according to the law of mass action, and reaction r2 according to the Michaelis Menten rate law. We assume we have measurements of Aint and Bint, as well as plenty of information about all the kinetic parameters and boundary conditions, and that the pathway is in a steady state, so that the concentrations c_m1_int and c_m2_int are not changing.

To formulate this situation as a statistical modelling problem, there are two functions `rmm` and `ma` that specify rate laws, and another function `fn` that specifies a steady state problem, i.e. finding values for c_m1_int and c_m2_int that put the system in a steady state.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by first choosing some true parameter values (see dictionary `TRUE_PARAMS`), and then simulating some measurements of c_m1_int and c_m2_int using these parameters: see function `simulate` for how this works. Then the log posterior is sampled using NUTS and GrapeNUTS, and the times are printed.

"""

from collections import OrderedDict
from functools import partial
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import optimistix as optx
import polars as pl
from jax.scipy.stats import norm

from grapevine.example_functions import linear_pathway_steady_state
from grapevine.util import run_grapenuts, run_nuts, time_run

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
ERROR_SD = 0.05
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "linear_pathway.csv"
TRUE_PARAMS = OrderedDict(
    log_km=jnp.array([2.0, 3.0]),
    log_vmax=jnp.array(0.0),
    log_keq=jnp.array([1.0, 1.0, 1.0]),
    log_kf=jnp.array([1.0, -1.0]),
    log_conc_ext=jnp.array([1.0, 0.0]),
)
DEFAULT_GUESS = jnp.array([0.1, 0.1])
N_WARMUP = 1000
N_SAMPLE = 1000
INIT_STEPSIZE = 0.0001
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.99
N_TEST = 6

ode_solver = diffrax.Tsit5()
steady_state_cond = diffrax.steady_state_event()
steady_state_event = diffrax.Event(steady_state_cond)
adjoint = diffrax.ImplicitAdjoint(
    linear_solver=lineax.AutoLinearSolver(well_posed=False)
)
controller = diffrax.PIDController(pcoeff=0.1, icoeff=0.3, rtol=1e-9, atol=1e-9)


solver = optx.Newton(rtol=1e-9, atol=1e-9)


def joint_logdensity_grapenuts(params, obs, guess):
    sol = optx.root_find(
        linear_pathway_steady_state,
        solver,
        guess,
        args=params,
        max_steps=int(1e5),
    )
    log_km, log_vmax, log_keq, log_kf, log_conc_ext = params.values()
    log_prior = jnp.sum(
        norm.logpdf(log_km, loc=TRUE_PARAMS["log_km"], scale=0.1).sum()
        + norm.logpdf(log_vmax, loc=TRUE_PARAMS["log_vmax"], scale=0.1).sum()
        + norm.logpdf(log_keq, loc=TRUE_PARAMS["log_keq"], scale=0.1).sum()
        + norm.logpdf(log_kf, loc=TRUE_PARAMS["log_kf"], scale=0.1).sum()
        + norm.logpdf(log_conc_ext, loc=TRUE_PARAMS["log_conc_ext"], scale=0.1).sum()
    )
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(sol.value), scale=jnp.full(obs.shape, ERROR_SD)
    ).sum()
    return log_prior + log_likelihood, sol.value


@eqx.filter_jit
def joint_logdensity_grapenuts_jac(params, obs, guess):
    jac = jax.jacfwd(linear_pathway_steady_state, argnums=0)(guess, params)
    inv_jac = jnp.linalg.inv(jac)

    def f_aux(t, x, args):
        inv_jac, params = args
        return (
            -inv_jac
            @ linear_pathway_steady_state(x, params)
            * jnp.log(0.2)
            / jnp.log(0.8)
            / (1 - t)
        )

    term = diffrax.ODETerm(f_aux)

    sol = diffrax.diffeqsolve(
        terms=term,
        solver=ode_solver,
        t0=0.0,
        t1=0.99999,
        dt0=0.01,
        y0=guess,
        max_steps=None,
        args=(inv_jac, params),
        stepsize_controller=controller,
        adjoint=adjoint,
    )
    if sol.ys is None:
        raise ValueError("No steady state found!")
    log_km, log_vmax, log_keq, log_kf, log_conc_ext = params.values()
    log_prior = jnp.sum(
        norm.logpdf(log_km, loc=TRUE_PARAMS["log_km"], scale=0.1).sum()
        + norm.logpdf(log_vmax, loc=TRUE_PARAMS["log_vmax"], scale=0.1).sum()
        + norm.logpdf(log_keq, loc=TRUE_PARAMS["log_keq"], scale=0.1).sum()
        + norm.logpdf(log_kf, loc=TRUE_PARAMS["log_kf"], scale=0.1).sum()
        + norm.logpdf(log_conc_ext, loc=TRUE_PARAMS["log_conc_ext"], scale=0.1).sum()
    )
    log_likelihood = norm.logpdf(
        jnp.log(obs), loc=jnp.log(sol.ys[0]), scale=jnp.full(obs.shape, ERROR_SD)
    ).sum()
    return log_prior + log_likelihood, sol.ys[0]


def joint_logdensity_nuts(params, obs):
    ld, _ = joint_logdensity_grapenuts(params, obs, DEFAULT_GUESS)
    return ld


def simulate(key, params, guess):
    sol = optx.root_find(
        linear_pathway_steady_state,
        solver,
        guess,
        args=params,
    )
    return sol.value, jnp.exp(
        jnp.log(sol.value) + jax.random.normal(key, shape=sol.value.shape) * ERROR_SD
    )


def compare(key: jax.Array) -> dict:
    sim_key, grapenuts_key, nuts_key = jax.random.split(key, 3)
    # simulate
    _, sim = simulate(sim_key, TRUE_PARAMS, DEFAULT_GUESS)
    # posteriors
    posterior_logdensity_gn = partial(joint_logdensity_grapenuts, obs=sim)
    posterior_logdensity_nuts = partial(joint_logdensity_nuts, obs=sim)
    run_fn_gn = eqx.filter_jit(
        partial(
            run_grapenuts,
            logdensity_fn=posterior_logdensity_gn,
            rng_key=grapenuts_key,
            init_parameters=TRUE_PARAMS,
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
            init_parameters=TRUE_PARAMS,
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
    result_gn = time_run(run_fn_gn, test_var="log_km")
    result_nuts = time_run(run_fn_nuts, test_var="log_km")
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


def run_comparison(n_test: int):
    key = jax.random.key(SEED)
    keys = jax.random.split(key, n_test)
    results = []
    for i, keyi in enumerate(keys):
        result = compare(keyi)
        result["rep"] = i
        results.append(result)
    return pl.from_records(results)


def main():
    results = run_comparison(n_test=N_TEST)
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    results.write_csv(CSV_OUTPUT_FILE)
    print("Results:")
    print(results)


if __name__ == "__main__":
    main()
