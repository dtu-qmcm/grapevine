from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import polars as pl

HERE = Path(__file__).parent
CSV_FILE_ROSENBROCK = HERE / "rosenbrock.csv"
CSV_FILE_LINEAR = HERE / "linear_pathway.csv"


def plot_comparison(ax_title: str, df: pl.DataFrame, ax: plt.Axes):
    xlow, xhigh = ax.get_xlim()
    groups = df.group_by(["algorithm"])
    n_group = df["algorithm"].n_unique()
    x = np.linspace(xlow, xhigh, n_group)
    algs = []
    for xi, ((algorithm_name,), subdf) in zip(x, groups):
        xs = np.linspace(xi - 0.01, xi + 0.01, len(subdf))
        ax.scatter(xs, subdf["neff_per_s"])
        algs.append(algorithm_name)
    ax.set(
        title=ax_title,
        ylabel="Effective samples per second\n(more is better)",
        xlabel="Algorithm",
    )
    ax.set_xticks(x, algs)
    ax.set_ylim(ymin=0)
    return ax


def mm_fig(results_df: pl.DataFrame):
    f, ax = plt.subplots(figsize=[8, 5])
    ax = plot_comparison(
        ax_title="Michaelis Menten root-finding benchmark",
        df=results_df.filter(pl.col("algorithm") != "Stan"),
        ax=ax,
    )
    return f, ax


def rosenbrock_fig(results: pl.DataFrame):
    improvement = (
        results[["dim", "algorithm", "repeat", "neff/s"]]
        .pivot(values="neff/s", index=["dim", "repeat"], on="algorithm")
        .with_columns(rel_improvement=pl.col("grapeNUTS") / pl.col("NUTS"))
    )
    f, ax = plt.subplots(figsize=[8, 5])
    ax.set_ylim(ymin=0.0, ymax=3.0)
    ax.scatter(improvement["dim"], improvement["rel_improvement"])
    ax.axhline(1.0, linestyle="--", color="black", label="y=1")
    ax.text(3.0, 1.05, "↑ grapeNUTS did better")
    ax.text(3.0, 0.95, "↓ NUTS did better", verticalalignment="top")

    ax.set_xticks(improvement["dim"].unique().sort())
    ax.set(
        title="Rosenbrock minimisation benchmark",
        xlabel="Dimension of embedded problem",
        ylabel="neff per second ratio grapeNUTS:NUTS",
    )
    return f, ax


def main():
    matplotlib.rcParams["savefig.dpi"] = 300
    df_simple = pl.read_csv(CSV_FILE_ROSENBROCK)
    df_linear = pl.read_csv(CSV_FILE_LINEAR)

    f, _ = mm_fig(df_linear)
    f.savefig(HERE / "mm.png", bbox_inches="tight")

    f, _ = rosenbrock_fig(df_simple)
    f.savefig(HERE / "rosenbrock.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
