"""
Visualization Module for Public Health Analytics.

Generates high-resolution publication charts:
1. Year-wise Public Health Cases (Line plot)
2. Top 10 States by Total Cases (Horizontal bar chart)
3. Top 10 Diseases by Total Cases (Horizontal bar chart)
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def generate_graphs(
    data_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Generate and save all 3 public health summary graphs.

    Parameters:
        data_path: Path to cleaned CSV.
        output_dir: Destination directory for PNG plots.

    Returns:
        list[Path]: List of generated chart paths.
    """
    root = get_project_root()
    if data_path is None:
        data_path = resolve_data_path("cleaned_health_data.csv")
    if output_dir is None:
        output_dir = root / "outputs" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Prepare required columns
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["Cases"])

    generated_files = []

    # Graph 1: Year-wise cases
    yearly = df.dropna(subset=["year"]).groupby("year")["Cases"].sum().sort_index()
    yearly_path = output_dir / "yearly_cases.png"

    plt.figure(figsize=(9, 5))
    yearly.plot(kind="line", marker="o", color="#2563eb", linewidth=2.2, markersize=6)
    plt.title("Year-wise Public Health Cases", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Total Cases", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(yearly_path, dpi=300)
    plt.close()
    generated_files.append(yearly_path)

    # Graph 2: Top 10 states
    states = df.groupby("state_ut")["Cases"].sum()
    states = states.sort_values(ascending=False).head(10).sort_values()
    states_path = output_dir / "state_cases.png"

    plt.figure(figsize=(10, 6))
    states.plot(kind="barh", color="#0d9488")
    plt.title("Top 10 States by Total Cases", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Total Cases", fontsize=11)
    plt.ylabel("State / Union Territory", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(states_path, dpi=300)
    plt.close()
    generated_files.append(states_path)

    # Graph 3: Top 10 diseases
    diseases = df.groupby("Disease")["Cases"].sum()
    diseases = diseases.sort_values(ascending=False).head(10).sort_values()
    diseases_path = output_dir / "disease_cases.png"

    plt.figure(figsize=(10, 6))
    diseases.plot(kind="barh", color="#e11d48")
    plt.title("Top 10 Diseases by Total Cases", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Total Cases", fontsize=11)
    plt.ylabel("Disease", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(diseases_path, dpi=300)
    plt.close()
    generated_files.append(diseases_path)

    print("\nAll three graphs generated successfully!")
    for idx, path in enumerate(generated_files, start=1):
        print(f"{idx}. {path}")

    return generated_files


if __name__ == "__main__":
    generate_graphs()
