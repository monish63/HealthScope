"""
Multithreading Performance Benchmarking Module.

Evaluates parallel execution performance (ThreadPoolExecutor) versus sequential baseline
across independent epidemiological grouping tasks.
"""

from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_data_path(filename: str = "cleaned_health_data.csv") -> Path:
    """Find dataset file across standard candidate locations."""
    root = get_project_root()
    candidates = [
        root / "data" / "processed" / filename,
        root / "data" / filename,
        Path.cwd() / "data" / "processed" / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
        root / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset file '{filename}'. Searched in {[str(c) for c in candidates]}"
    )


def run_threading_benchmark(
    data_path: Path | None = None,
    max_workers: int = 3,
) -> dict:
    """
    Run sequential and multithreaded benchmarks on health data aggregation tasks.

    Parameters:
        data_path: Path to cleaned CSV.
        max_workers: Number of threads for parallel pool.

    Returns:
        dict: Performance metrics including runtimes and speedup.
    """
    if data_path is None:
        data_path = resolve_data_path("cleaned_health_data.csv")

    print(f"Loading cleaned dataset from: {data_path}")
    df = pd.read_csv(data_path)

    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["Cases"])

    # Three independent analysis tasks
    def disease_analysis():
        return df.groupby("Disease")["Cases"].sum()

    def state_analysis():
        return df.groupby("state_ut")["Cases"].sum()

    def yearly_analysis():
        return df.groupby("year")["Cases"].sum()

    tasks = [
        disease_analysis,
        state_analysis,
        yearly_analysis,
    ]

    # Sequential execution
    start = time.perf_counter()
    for task in tasks:
        _ = task()
    sequential_time = time.perf_counter() - start

    # Multithreaded execution
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda task: task(), tasks))
    parallel_time = time.perf_counter() - start

    speedup = (sequential_time / parallel_time) if parallel_time > 0 else 1.0

    print("\nPUBLIC HEALTH MULTITHREADING ANALYSIS")
    print("-------------------------------------")
    print("Disease groups:", len(results[0]))
    print("States analyzed:", len(results[1]))
    print("Years analyzed:", len(results[2]))
    print(f"Sequential time: {sequential_time:.6f} seconds")
    print(f"Multithreaded time ({max_workers} workers): {parallel_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print("Analysis completed successfully!")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "disease_groups": len(results[0]),
        "states_analyzed": len(results[1]),
        "years_analyzed": len(results[2]),
    }


if __name__ == "__main__":
    run_threading_benchmark()
