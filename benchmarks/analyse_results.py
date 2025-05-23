from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch

HERE = Path(__file__).parent
CSV_FILE_METHIONINE = HERE / "methionine.csv"
CSV_FILE_ROSENBROCK = HERE / "rosenbrock.csv"
CSV_FILE_LINEAR = HERE / "linear.csv"
CSV_FILE_TRAJECTORY = HERE / "trajectory.csv"
CSV_FILE_TEST_FUNCTIONS = HERE / "test_functions.csv"


class HandlerArrow(HandlerPatch):
    """Custom legend handler for arrows.

    Copied from <https://stackoverflow.com/questions/60781312/plotting-arrow-in-front-of-legend>
    """

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        p = mpatches.FancyArrow(
            0,
            0.5 * height,
            width,
            0,
            length_includes_head=True,
            head_width=0.7 * height,
            fill=True,
            linewidth=0,
            color="tab:blue",
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


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


def performance_fig(results: pl.DataFrame):
    f, ax = plt.subplots(figsize=[8, 5])
    cases = pl.DataFrame(
        {
            "case": results["case"].unique(maintain_order=True),
            "x": np.linspace(*ax.get_xlim(), results["case"].n_unique()),
        }
    )
    plot_df = (
        results.join(cases, on="case")
        .with_columns(
            x_jitter=np.random.normal(scale=0.01, size=results.shape[0]),
            steps_per_neff=pl.col("n_newton_steps") / pl.col("neff"),
        )
        # guess_implicit_cg should behave the same as guess_implicit
        .filter(pl.col("heuristic") != "guess_implicit_cg")
    )
    for (heuristic,), subdf in plot_df.group_by(
        "heuristic", maintain_order=True
    ):
        ax.scatter(
            subdf["x"] + subdf["x_jitter"],
            subdf["steps_per_neff"],
            label=str(heuristic).replace("_", "-").capitalize(),
            alpha=0.8,
        )
        fail = subdf.filter(pl.col("n_newton_steps") == 0)
        ax.scatter(
            fail["x"] + fail["x_jitter"],
            fail["n_newton_steps"] + 0.1,
            marker="x",
            color="red",
            label="Unsuccessful MCMC run",
        )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(
        sorted(dict(zip(labels, handles)).items())
    )  # Remove duplicate labels
    ax.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
    )
    # ax.grid(visible=True, which="major", axis="y")
    ax.set_xticks(cases["x"], list(cases["case"]), fontsize="xx-small")
    ax.set(
        ylabel="Newton steps per effective sample",
        xlabel="Test case",
    )
    ax.semilogy()
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    return f, ax


def trajectory_fig(result: pl.DataFrame):
    min_steps = result[["GN", "NUTS"]].to_numpy().min()
    max_steps = result[["GN", "NUTS"]].to_numpy().max()
    bins = np.linspace(min_steps - 1.5, max_steps + 1.5, 20)
    f, axes = plt.subplots(1, 2, figsize=[15, 5])
    axes[0].plot(
        result["sol_0"],
        result["sol_1"],
        "o",
        color="black",
        label="solution",
    )
    axes[0].plot(
        result["default_0"][0],
        result["default_1"][0],
        "s",
        color="tab:orange",
        label="Default guess",
    )
    axes[0].plot(
        result["guess_0"],
        result["guess_1"],
        "|",
        color="tab:blue",
        label="Grapevine guess",
        markersize=10,
        zorder=-1,
    )
    for sx, sy, gx, gy in result[
        ["sol_0", "sol_1", "guess_0", "guess_1"]
    ].iter_rows():
        dx = sx - gx
        dy = sy - gy
        arrow = axes[0].arrow(
            gx,
            gy,
            dx,
            dy,
            # alpha=0.3,
            color="olivedrab",
            fill=True,
            linewidth=0.1,
            length_includes_head=True,
        )
    h, labels = axes[0].get_legend_handles_labels()
    h.append(arrow)
    labels.append("Grapevine guess to solution")
    axes[1].hist(result["GN"], bins=bins, alpha=0.8, label="Grapevine")
    axes[1].hist(result["NUTS"], bins=bins, alpha=0.8, label="Default guess")
    axes[0].legend(
        h,
        labels,
        handler_map={mpatches.FancyArrow: HandlerArrow()},
        frameon=False,
    )
    axes[1].legend(frameon=False)
    axes[0].set(
        xlabel="Solution component 1",
        ylabel="Solution component 2",
    )
    axes[1].set(
        xlabel="Number of Newton steps\n(fewer is better)",
        ylabel="Frequency",
    )
    return f, axes


def main():
    matplotlib.rcParams["savefig.dpi"] = 300
    df_methionine = pl.read_csv(CSV_FILE_METHIONINE).with_columns(
        case=pl.lit("Methionine cycle"), dim=0
    )
    df_linear = pl.read_csv(CSV_FILE_LINEAR).with_columns(
        case=pl.lit("Linear network"), dim=0
    )
    df_trajectory = pl.read_csv(CSV_FILE_TRAJECTORY)

    df_test_functions = pl.read_csv(CSV_FILE_TEST_FUNCTIONS)

    df_performance = pl.concat(
        [df_test_functions, df_linear, df_methionine], how="align"
    )

    f, _ = performance_fig(df_performance)
    f.savefig(HERE / "performance.png", bbox_inches="tight", dpi=300)

    f, _ = trajectory_fig(df_trajectory)
    f.savefig(HERE / "trajectory.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
