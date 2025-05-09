"""An example comparing GrapeNUTS and NUTS on a representative problem.

The problem is a steady kinetic model of a linear pathway with this structure:

    Aext <-r1-> Aint <-r2-> Bint <-r3-> Bext

Reactions r1 and r3 behave according to the law of mass action, and reaction r2 according to the Michaelis Menten rate law. We assume we have measurements of Aint and Bint, as well as plenty of information about all the kinetic parameters and boundary conditions, and that the pathway is in a steady state, so that the concentrations c_m1_int and c_m2_int are not changing.

To formulate this situation as a statistical modelling problem, there is a function `linear_pathway_steady_state` that specifies a steady state problem, i.e. finding values for c_m1_int and c_m2_int that put the system in a steady state.

We can then specify joint and posterior log density functions in terms of log scale parameters, which we can sample using GrapeNUTS.

The benchmark proceeds by repeatedly choosing some true parameter values at random by perturbing the dictionary `TRUE_PARAMS`, then using these parameters to simulate some measurements of c_m1_int and c_m2_int. Then the log posterior is sampled using NUTS and GrapeNUTS, and the relative ess/second valeus are printed.

"""

from pathlib import Path

import jax

from grapevine.examples import linear_pathway
from grapevine.benchmarking import run_benchmark

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "linear.csv"
N_TEST = 6
RUN_GRAPENUTS_KWARGS = dict(
    num_warmup=2000,
    num_samples=1000,
    initial_step_size=0.0001,
    max_num_doublings=10,
    is_mass_matrix_diagonal=False,
    target_acceptance_rate=0.9,
    progress_bar=False,
)
DEFAULT_GUESS_INFO = (
    linear_pathway.DEFAULT_GUESS,
    linear_pathway.TRUE_PARAMS,
    0,
)


def main():
    results = run_benchmark(
        random_seed=SEED,
        joint_logdensity_funcs={
            "NUTS": linear_pathway.joint_logdensity_guess_default,
            "guess_previous": linear_pathway.joint_logdensity_guess_previous,
            "ift": linear_pathway.joint_logdensity_guess_implicit,
            "ift_cg": linear_pathway.joint_logdensity_guess_implicit_cg,
        },
        baseline_params=linear_pathway.TRUE_PARAMS,
        param_sd=linear_pathway.PARAM_SD,
        n_test=N_TEST,
        run_grapenuts_kwargs=RUN_GRAPENUTS_KWARGS,
        sim_func=linear_pathway.simulate,
        default_guess_info=DEFAULT_GUESS_INFO,
    )
    print(f"Benchmark results saved to {CSV_OUTPUT_FILE}")
    results.write_csv(CSV_OUTPUT_FILE)
    print("Runtimes:")
    print(results.pivot("heuristic", index="rep", values="time"))
    print("Effective sample sizes:")
    print(results.pivot("heuristic", index="rep", values="neff"))
    print("Newton steps:")
    print(results.pivot("heuristic", index="rep", values="n_newton_steps"))


if __name__ == "__main__":
    main()
