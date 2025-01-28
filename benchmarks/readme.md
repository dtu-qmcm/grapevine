# How to run the benchmarks

1. Install [uv](https://docs.astral.sh/uv/)
2. Install grapevine
3. Run the benchmarks:

```sh
uv run benchmarks/methionine.py
uv run benchmarks/linear.py
uv run benchmarks/rosenbrock.py
uv run benchmarks/trajectory.py
```

4. Make the graphs:

```sh
uv run benchmarks/analyse_results.py
```
