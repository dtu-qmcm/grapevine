from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
import polars as pl

HERE = Path(__file__).parent
CSV_FILE_ROSENBROCK = HERE / "rosenbrock.csv"
CSV_FILE_LINEAR = HERE / "linear_pathway.csv"
CSV_FILE_TRAJECTORY = HERE / "trajectory.csv"


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
    ax.set_ylim(ymin=0.0, ymax=3.0)

    models = results.group_by(["dim"]).agg(pl.col("model").first()).sort("dim")
    model_names = [models["model"].to_list()[0]] + [
        f"{m} dim {d}" for d, m in models.iter_rows() if m == "Rosenbrock"
    ]
    models = models.with_columns(
        xtick_loc=np.linspace(*ax.get_xlim(), len(model_names))  # type: ignore
    )
    results = results.join(models, on=["model", "dim"])
    ax.scatter(results["xtick_loc"], results["perf_ratio"])
    ax.axhline(1.0, linestyle="--", color="black", label="y=1")
    ax.text(0.1, 1.05, "↑ grapeNUTS did better")
    ax.text(0.1, 0.95, "↓ NUTS did better", verticalalignment="top")
    ax.set_xticks(models["xtick_loc"], model_names, rotation=90)
    ax.set(
        xlabel="Problem",
        ylabel="Performance ratio grapeNUTS:NUTS",
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
    for sx, sy, gx, gy in result[["sol_0", "sol_1", "guess_0", "guess_1"]].iter_rows():
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
    df_rb = pl.read_csv(CSV_FILE_ROSENBROCK)
    df_linear = pl.read_csv(CSV_FILE_LINEAR)
    df_trajectory = pl.read_csv(CSV_FILE_TRAJECTORY)
    df_rb = df_rb.with_columns(model=pl.lit("Rosenbrock"))
    df_linear = df_linear.with_columns(model=pl.lit("Small enzyme network"), dim=0)
    df_performance = pl.concat([df_rb, df_linear], how="align")

    f, _ = performance_fig(df_performance)
    f.savefig(HERE / "performance.png", bbox_inches="tight")

    f, _ = trajectory_fig(df_trajectory)
    f.savefig(HERE / "trajectory.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
