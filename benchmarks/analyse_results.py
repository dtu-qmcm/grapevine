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
HEURISTIC_COLORS = {
    "guess_static": "tab:blue",
    "guess_previous": "tab:orange",
    "guess_implicit": "tab:green",
}


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
            x_jitter=pl.Series(
                np.random.normal(scale=0.01, size=results.shape[0])
            ),
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
        ylabel="Solver steps per effective sample",
        xlabel="Test case",
    )
    ax.semilogy()
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    return f, ax


def trajectory_scatter(result: pl.DataFrame):
    f, axes = plt.subplots(3, 1, figsize=[8, 10], sharex=True)
    axes = axes.ravel()
    gfunc_to_ax = dict(zip(HEURISTIC_COLORS.keys(), axes))
    sol_x = (
        result.sort("i", "j")
        .group_by("i", maintain_order=True)
        .agg(pl.col("sol_0").last())["sol_0"]
    )
    sol_y = (
        result.sort("i", "j")
        .group_by("i", maintain_order=True)
        .agg(pl.col("sol_1").last())["sol_1"]
    )
    for (gfunc,), subdf in result.group_by("gfunc", maintain_order=True):
        assert isinstance(gfunc, str)
        ax = gfunc_to_ax[gfunc]
        total_steps = len(subdf) - 1
        legend_info = dict()
        sct = ax.scatter(sol_x, sol_y, color="black", label="solution")
        for i, subsubdf in subdf.group_by("i", maintain_order=True):
            x = [subsubdf["guess_0"].first()] + subsubdf["sol_0"].to_list()
            y = [subsubdf["guess_1"].first()] + subsubdf["sol_1"].to_list()
            line = ax.plot(
                x,
                y,
                marker="x",
                color=HEURISTIC_COLORS[gfunc],
                markevery=slice(1, None),
                zorder=-1,
                lw=1,
            )[0]
            line_color = line.get_color()
            marker = line.get_marker()
            line_handle = matplotlib.lines.Line2D(
                [],
                [],
                color=line_color,
                linestyle="-",
            )
            marker_handle = matplotlib.lines.Line2D(
                [],
                [],
                color=line_color,
                marker=marker,
                linestyle="None",
                markersize=8,
            )

        legend_info["Solution"] = sct
        legend_info["Solver path"] = line_handle
        legend_info["Newton step"] = marker_handle
        ax.legend(legend_info.values(), legend_info.keys(), frameon=False)
        ax.set(
            title="Heuristic: "
            + gfunc.replace("_", "-")
            + f" (total steps: {total_steps})",
            ylabel="Solution component 2",
        )
    axes[-1].set(xlabel="Solution component 1")
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

    f, _ = trajectory_scatter(df_trajectory)
    f.savefig(HERE / "trajectory.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
