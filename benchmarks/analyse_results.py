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
    # Determine bounds
    all_x = result["guess_0"].to_list() + result["sol_0"].to_list()
    all_y = result["guess_1"].to_list() + result["sol_1"].to_list()
    xlim = (min(all_x) - 0.05, max(all_x) + 0.05)
    ylim = (min(all_y) - 0.05, max(all_y) + 0.05)

    data_by_gfunc = {
        gfunc: subdf
        for (gfunc,), subdf in result.group_by("gfunc", maintain_order=True)
    }

    # Build per-heuristic schedules
    schedules = {}
    max_frames = 0
    for gfunc, subdf in data_by_gfunc.items():
        sched = []
        unique_i = sorted(subdf["i"].unique())
        for i in unique_i:
            # Add initial guess frame for each MCMC step
            sched.append((i, -1))
            step_data = subdf.filter(pl.col("i") == i)
            max_j = step_data["j"].max()
            if max_j is not None:
                # Add frames for each Newton step
                for j in range(max_j + 1):
                    sched.append((i, j))

        schedules[gfunc] = sched
        max_frames = max(max_frames, len(sched))

    f, axes = plt.subplots(3, 1, figsize=[8, 10], sharex=True)
    axes = axes.ravel()
    gfunc_to_ax = dict(zip(HEURISTIC_COLORS.keys(), axes))

    def update(frame_idx):
        for ax in axes:
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        for gfunc, ax in gfunc_to_ax.items():
            subdf = data_by_gfunc.get(gfunc)
            if subdf is None:
                continue

            sched = schedules[gfunc]
            if frame_idx >= len(sched):
                curr_state = sched[-1]
                # Count frames that are NOT (i, -1) frames for total Newton steps
                step_count = sum(1 for s in sched if s[1] >= 0)
            else:
                curr_state = sched[frame_idx]
                # Count Newton steps up to current frame
                step_count = sum(1 for s in sched[: frame_idx + 1] if s[1] >= 0)

            curr_i, curr_j = curr_state

            # 1. History (Previous i): Plot full paths
            history = subdf.filter(pl.col("i") < curr_i)
            if not history.is_empty():
                for _, step_df in history.group_by("i"):
                    step_df = step_df.sort("j")
                    x_hist = [step_df["guess_0"][0]] + step_df[
                        "sol_0"
                    ].to_list()
                    y_hist = [step_df["guess_1"][0]] + step_df[
                        "sol_1"
                    ].to_list()

                    # Completed path
                    ax.plot(
                        x_hist,
                        y_hist,
                        color=HEURISTIC_COLORS[gfunc],
                        lw=0.5,
                        alpha=0.3,
                        zorder=-1,
                    )
                    # Final solution
                    ax.scatter(
                        x_hist[-1], y_hist[-1], color="black", s=10, zorder=0
                    )

            # 2. Current i
            current_step_full = subdf.filter(pl.col("i") == curr_i).sort("j")
            if not current_step_full.is_empty():
                guess_point = (
                    current_step_full["guess_0"][0],
                    current_step_full["guess_1"][0],
                )

                if curr_j == -1:
                    # Just the guess
                    x_path = [guess_point[0]]
                    y_path = [guess_point[1]]
                else:
                    # Draw path up to curr_j
                    current_partial = current_step_full.filter(
                        pl.col("j") <= curr_j
                    )
                    x_path = [guess_point[0]] + current_partial[
                        "sol_0"
                    ].to_list()
                    y_path = [guess_point[1]] + current_partial[
                        "sol_1"
                    ].to_list()

                # Draw current path
                ax.plot(
                    x_path,
                    y_path,
                    color=HEURISTIC_COLORS[gfunc],
                    lw=1.5,
                    zorder=1,
                )

                # Markers for intermediate steps
                if len(x_path) > 1:
                    ax.plot(
                        x_path[1:],
                        y_path[1:],
                        color=HEURISTIC_COLORS[gfunc],
                        marker="^",
                        linestyle="None",
                        markersize=4,
                        zorder=1,
                    )

                # Current position marker
                ax.plot(
                    x_path[-1],
                    y_path[-1],
                    color=HEURISTIC_COLORS[gfunc],
                    marker="o",
                    markersize=8,
                    markeredgecolor="black",
                    zorder=2,
                )

            gfunc_clean = gfunc.replace("_", "-")
            ax.set_title(
                f"Heuristic: {gfunc_clean} (steps so far: {step_count})"
            )
            ax.set_ylabel("Solution component 2")

        axes[-1].set(xlabel="Solution component 1")
        return axes

    anim = FuncAnimation(f, update, frames=max_frames, interval=200)
    return anim


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
