"""
Data Ingestion and Cleaning Module.

Loads raw epidemiological outbreak data, handles missing values, removes duplicates,
converts numeric columns safely, and saves the cleaned dataset.
"""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_input_path(filename: str = "Final_data.csv") -> Path:
    """Find input file across standard candidate locations."""
    root = get_project_root()
    candidates = [
        root / "data" / "raw" / filename,
        root / "data" / filename,
        Path.cwd() / "data" / "raw" / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
        root / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find input file '{filename}'. Searched in {[str(c) for c in candidates]}"
    )


def clean_health_data(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load raw health data, clean it, and export cleaned CSV.

    Parameters:
        input_path: Path to the raw CSV file.
        output_path: Path to write the cleaned CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    root = get_project_root()
    if input_path is None:
        input_path = resolve_input_path("Final_data.csv")
    if output_path is None:
        output_path = root / "data" / "processed" / "cleaned_health_data.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)

    print("\nORIGINAL DATASET")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Remove unnecessary index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove duplicate records
    df = df.drop_duplicates()

    # Convert numeric columns safely
    numeric_cols = [
        "Cases", "Deaths", "day", "mon", "year",
        "Latitude", "Longitude", "preci", "LAI", "Temp"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check missing values before cleaning
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Keep missing death counts as unknown
    # Fill missing climate values with their column median
    for col in ["preci", "LAI", "Temp"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("\nCLEANING COMPLETED")
    print("Rows after cleaning:", len(df))
    print("Columns after cleaning:", len(df.columns))
    print("\nRemaining missing values:")
    print(df.isnull().sum())
    print(f"\nSaved cleaned dataset as: {output_path}")

    return df


if __name__ == "__main__":
    clean_health_data()
