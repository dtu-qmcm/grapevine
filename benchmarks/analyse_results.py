from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
    plot_df = (
        results.with_columns(
            x_jitter=pl.Series(
                np.random.normal(scale=0.01, size=results.shape[0])
            ),
            steps_per_neff=pl.col("n_newton_steps") / pl.col("neff"),
        )
        # guess_implicit_cg should behave the same as guess_implicit
        .filter(pl.col("heuristic") != "guess_implicit_cg")
        .sort(pl.col("steps_per_neff").mean().over("case"))
    )
    x = pl.DataFrame(
        {
            "case": plot_df["case"].unique(maintain_order=True),
            "x": np.linspace(*ax.get_xlim(), plot_df["case"].n_unique()),
        }
    )
    plot_df = plot_df.join(x, on="case")
    for (heuristic,), subdf in plot_df.group_by(
        "heuristic", maintain_order=True
    ):
        ax.scatter(
            subdf["x"] + subdf["x_jitter"],
            subdf["steps_per_neff"],
            label=str(heuristic).replace("_", "-").capitalize(),
            alpha=0.8,
            color=HEURISTIC_COLORS[heuristic],
        )
        fail = subdf.filter(pl.col("n_newton_steps") == 0)
        ax.scatter(
            fail["x"] + fail["x_jitter"],
            fail["n_newton_steps"] + 0.1,
            marker="|",
            color=HEURISTIC_COLORS[heuristic],
            label="Unsuccessful MCMC run",
        )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(
        sorted(dict(zip(labels, handles)).items())
    )  # Remove duplicate labels
    leg = ax.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
    )
    leg.legend_handles[-1].set_facecolor("black")

    # ax.grid(visible=True, which="major", axis="y")
    ax.set_xticks(x["x"], list(x["case"]), fontsize="xx-small")
    ax.set(
        ylabel="Solver steps per effective sample",
        xlabel="Benchmark",
    )
    ax.semilogy()
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    return f, ax


def create_trajectory_animation(result: pl.DataFrame):
    f, axes = plt.subplots(3, 1, figsize=[8, 10], sharex=True)

    all_x = result["guess_0"].to_list() + result["sol_0"].to_list()
    all_y = result["guess_1"].to_list() + result["sol_1"].to_list()
    xlim = (min(all_x) - 0.01, max(all_x) + 0.01)
    ylim = (min(all_y) - 0.01, max(all_y) + 0.01)
    df = result.pivot(on="gfunc", index=["i", "j"])
    sol_x, sol_y = (
        result.filter(gfunc="guess_implicit")
        .group_by("i", maintain_order=True)
        .last()[f"sol_{str(i)}"]
        for i in (0, 1)
    )

    def update(frame_idx):
        for ax, gfunc, color in zip(
            axes,
            HEURISTIC_COLORS.keys(),
            HEURISTIC_COLORS.values(),
        ):
            xcol = "sol_0_" + gfunc
            ycol = "sol_1_" + gfunc
            first_guess_x = df["guess_0_" + gfunc][0]
            first_guess_y = df["guess_1_" + gfunc][0]
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.scatter(
                sol_x, sol_y, color="black", marker="x", label="Solution"
            )
            df_so_far = (
                df.drop_nulls(subset=xcol)
                .with_row_index()
                .filter(pl.col("index") <= frame_idx)
            )
            x_so_far = [first_guess_x] + df_so_far[xcol].to_list()
            y_so_far = [first_guess_y] + df_so_far[ycol].to_list()
            i = df_so_far["i"].max()
            df_i_so_far = df_so_far.filter(i=i)
            guess_xi = df_i_so_far["guess_0_" + gfunc][0]
            guess_yi = df_i_so_far["guess_1_" + gfunc][0]
            xi_so_far = [guess_xi] + df_i_so_far[xcol].to_list()
            yi_so_far = [guess_yi] + df_i_so_far[ycol].to_list()
            ax.scatter(guess_xi, guess_yi, color=color, label="Guess")
            ax.plot(x_so_far, y_so_far, color=color, alpha=0.2)
            ax.plot(xi_so_far, yi_so_far, color=color)
            n_steps = df_so_far[xcol].is_not_null().sum()
            ax.set_title(f"Heuristic: {gfunc} (steps so far: {n_steps})")
            ax.legend(frameon=False)
        axes[-1].set(xlabel="Solution component 0")
        axes[1].set(ylabel="Solution component 1")
        return axes

    return FuncAnimation(
        f,
        update,
        frames=len(df),
        interval=200,
        repeat_delay=1000,
    )


def illustrative_figure(result: pl.DataFrame):
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
                color=HEURISTIC_COLORS[gfunc],
                zorder=-1,
                lw=1,
            )[0]
            crosses = ax.plot(
                [xi + 0.001 for xi in x],
                [yi - 0.005 for yi in y],
                color=HEURISTIC_COLORS[gfunc],
                marker="^",
                markevery=slice(1, None),
                zorder=-1,
                markersize=2,
                lw=0,
            )[0]
            line_color = line.get_color()
            marker = crosses.get_marker()
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

    f, _ = illustrative_figure(df_trajectory)
    f.savefig(HERE / "trajectory.png", bbox_inches="tight", dpi=300)

    anim = create_trajectory_animation(df_trajectory)
    anim.save(HERE / "trajectory.gif", writer=PillowWriter(fps=5))


if __name__ == "__main__":
    main()
