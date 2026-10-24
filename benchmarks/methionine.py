"""An example comparing GrapeNUTS and NUTS on a representative problem."""

from pathlib import Path

import jax

from grapevine.examples import methionine
from grapevine.benchmarking import run_benchmark

# Use 64 bit floats
jax.config.update("jax_enable_x64", True)


SEED = 1234
HERE = Path(__file__).parent
CSV_OUTPUT_FILE = HERE / "methionine.csv"
N_TEST = 20
RUN_GRAPENUTS_KWARGS = dict(
    num_warmup=500,
    num_samples=500,
    initial_step_size=0.0001,
    max_num_doublings=10,
    is_mass_matrix_diagonal=False,
    target_acceptance_rate=0.9,
)


def main():
    results = run_benchmark(
        random_seed=SEED,
        joint_logdensity=methionine.joint_logdensity,
        # cg fails - singular jacobian?
        guess_fns={
            k: v
            for k, v in methionine.GUESS_FNS.items()
            if k != "guess_implicit_cg"
        },
        baseline_params=methionine.TRUE_PARAMS,
        param_sd=methionine.PARAM_SD,
        n_test=N_TEST,
        run_grapenuts_kwargs=RUN_GRAPENUTS_KWARGS,
        sim_func=methionine.simulate,
        default_guess=methionine.DEFAULT_GUESS,
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
