from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import polars as pl

HERE = Path(__file__).parent
CSV_FILE_SIMPLE = HERE / "simple_example.csv"
CSV_FILE_LINEAR = HERE / "linear_pathway.csv"


def plot_comparison(name: str, df: pl.DataFrame, ax: plt.Axes):
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
        title=name,
        ylabel="Effective samples per second\n(higher is better)",
        xlabel="Algorithm",
    )
    ax.set_xticks(x, algs)
    ax.set_ylim(ymin=0)
    return ax


def comparison_figure(df_dict):
    f, axes = plt.subplots(1, len(df_dict.keys()), figsize=[12, 5])
    f.suptitle("Benchmark comparison")
    for (name, df), ax in zip(df_dict.items(), axes):
        ax = plot_comparison(name, df.filter(pl.col("algorithm") != "Stan"), ax)
    return f, axes


def main():
    df_simple = pl.read_csv(CSV_FILE_SIMPLE)
    df_linear = pl.read_csv(CSV_FILE_LINEAR)
    df_dict = {"Simple model": df_simple, "Michaelis Menten model": df_linear}
    f, _ = comparison_figure(df_dict)
    f.savefig(HERE / "figure.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
