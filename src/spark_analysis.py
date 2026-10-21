"""
Distributed Health Outbreak Analysis using Apache Spark (PySpark).

Performs distributed aggregation across disease categories, geographic states/UTs,
and annual temporal trends, exporting summary CSVs for downstream analytics.
"""

import os
import sys
from pathlib import Path

# Resolve JAVA_HOME to DOS 8.3 short name on Windows to prevent cmd.exe batch syntax errors
if sys.platform == "win32" and "JAVA_HOME" in os.environ:
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(500)
        if ctypes.windll.kernel32.GetShortPathNameW(os.environ["JAVA_HOME"], buf, len(buf)):
            os.environ["JAVA_HOME"] = buf.value
    except Exception:
        pass

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count, avg, desc


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


def run_spark_analysis(
    data_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """
    Run PySpark aggregations on outbreak data and save summary CSVs.

    Parameters:
        data_path: Path to cleaned CSV file.
        output_dir: Directory where summary CSV files will be stored.

    Returns:
        tuple[Path, Path, Path]: Paths to disease, state, and year summary CSVs.
    """
    root = get_project_root()
    if data_path is None:
        data_path = resolve_data_path("cleaned_health_data.csv")
    if output_dir is None:
        output_dir = root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Spark session and reading from: {data_path}")
    spark = (
        SparkSession.builder
        .appName("PublicHealthOutbreakAnalysis")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        # Load cleaned dataset
        df = spark.read.csv(
            str(data_path),
            header=True,
            inferSchema=True
        )

        print("\n===== DATASET OVERVIEW =====")
        print("Total records:", df.count())
        df.printSchema()

        # 1. Disease-wise case analysis
        print("\n===== DISEASE-WISE CASE TOTALS =====")
        disease_summary = df.groupBy("Disease").agg(
            sum("Cases").alias("Total_Cases"),
            sum("Deaths").alias("Reported_Deaths"),
            count("*").alias("Records")
        ).orderBy(desc("Total_Cases"))

        disease_summary.show(10, truncate=False)

        # 2. State-wise case analysis
        print("\n===== STATE-WISE CASE TOTALS =====")
        state_summary = df.groupBy("state_ut").agg(
            sum("Cases").alias("Total_Cases"),
            count("*").alias("Records")
        ).orderBy(desc("Total_Cases"))

        state_summary.show(10, truncate=False)

        # 3. Year-wise case trends
        print("\n===== YEAR-WISE CASE TOTALS =====")
        year_summary = df.groupBy("year").agg(
            sum("Cases").alias("Total_Cases"),
            avg("Cases").alias("Average_Cases")
        ).orderBy("year")

        year_summary.show(truncate=False)

        # Save results
        disease_path = output_dir / "disease_summary.csv"
        state_path = output_dir / "state_summary.csv"
        year_path = output_dir / "year_summary.csv"

        disease_summary.toPandas().to_csv(disease_path, index=False)
        state_summary.toPandas().to_csv(state_path, index=False)
        year_summary.toPandas().to_csv(year_path, index=False)

        print("\nAnalysis completed. Summary CSV files saved:")
        print(f"1. {disease_path}")
        print(f"2. {state_path}")
        print(f"3. {year_path}")

        return disease_path, state_path, year_path

    finally:
        spark.stop()


if __name__ == "__main__":
    run_spark_analysis()
