from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import polars as pl

HERE = Path(__file__).parent
CSV_FILE_ROSENBROCK = HERE / "rosenbrock.csv"
CSV_FILE_LINEAR = HERE / "linear_pathway.csv"
CSV_FILE_TRAJECTORY = HERE / "trajectory.csv"


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


def trajectory_fig(result: pl.DataFrame):
    min_steps = result[["GN", "NUTS"]].to_numpy().min()
    max_steps = result[["GN", "NUTS"]].to_numpy().max()
    bins = np.linspace(min_steps - 1.5, max_steps + 1.5, 20)
    f, axes = plt.subplots(1, 2, figsize=[15, 5])
    axes[0].plot(
        result["sol_0"],
        result["sol_1"],
        "^",
        color="black",
        label="solution",
    )
    axes[0].plot(
        result["guess_0"],
        result["guess_1"],
        "o",
        color="tab:blue",
        label="Grapevine guess",
        markersize=9,
        zorder=0,
    )
    axes[0].plot(
        result["default_0"][0],
        result["default_1"][0],
        "x",
        color="tab:orange",
        label="Default guess",
    )
    for sx, sy, gx, gy in result[["sol_0", "sol_1", "guess_0", "guess_1"]].iter_rows():
        dx = sx - gx
        dy = sy - gy
        axes[0].arrow(
            gx,
            gy,
            dx,
            dy,
            alpha=0.3,
            color="tab:blue",
            fill=True,
            linewidth=0,
            length_includes_head=True,
        )
    axes[1].hist(result["GN"], bins=bins, alpha=0.8, label="Grapevine")
    axes[1].hist(result["NUTS"], bins=bins, alpha=0.8, label="Default guess")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[0].set(
        title="Trajectory through solution space",
        xlabel="Solution component 1",
        ylabel="Solution component 2",
    )
    axes[1].set(
        title="Distribution of solver costs",
        xlabel="Number of Newton steps\n(fewer is better)",
        ylabel="Frequency",
    )
    f.suptitle("Grapevine vs a default guess on a single HMC trajectory")
    return f, axes


def main():
    matplotlib.rcParams["savefig.dpi"] = 300
    df_simple = pl.read_csv(CSV_FILE_ROSENBROCK)
    df_linear = pl.read_csv(CSV_FILE_LINEAR)
    df_trajectory = pl.read_csv(CSV_FILE_TRAJECTORY)

    f, _ = mm_fig(df_linear)
    f.savefig(HERE / "mm.png", bbox_inches="tight")

    f, _ = rosenbrock_fig(df_simple)
    f.savefig(HERE / "rosenbrock.png", bbox_inches="tight")

    f, _ = trajectory_fig(df_trajectory)
    f.savefig(HERE / "trajectory.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
