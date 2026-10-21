"""
Public Health Outbreak Monitoring & Distributed Analytics Pipeline.

Main orchestrator CLI for the complete data engineering & analytics workflow:
1. Data Ingestion & Cleaning (Pandas)
2. Multithreading Benchmarking (ThreadPoolExecutor vs Sequential)
3. Distributed Aggregations (Apache Spark / PySpark)
4. Visualization & Publication Figures (Matplotlib)
5. Interactive Web Dashboard (Streamlit)
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.clean_data import clean_health_data
from src.threading_analysis import run_threading_benchmark
from src.spark_analysis import run_spark_analysis
from src.graphs import generate_graphs


def run_all():
    """Execute all pipeline stages sequentially."""
    print("\n" + "=" * 65)
    print(" [PUBLIC HEALTH OUTBREAK MONITORING & DISTRIBUTED ANALYTICS]")
    print("=" * 65)

    print("\n>>> STAGE 1: Data Cleaning & Preprocessing")
    clean_health_data()

    print("\n>>> STAGE 2: Multithreading Performance Benchmarking")
    run_threading_benchmark()

    print("\n>>> STAGE 3: Apache Spark Distributed Aggregations")
    try:
        run_spark_analysis()
    except Exception as exc:
        print(f"\n[WARNING] Spark analysis encountered an error: {exc}")
        print("Tip: Ensure Java JDK 11 or 17 is installed and JAVA_HOME is set.")

    print("\n>>> STAGE 4: Publication Visualization Generation")
    generate_graphs()

    print("\n" + "=" * 65)
    print(" [OK] PIPELINE EXECUTION COMPLETED")
    print(" Data outputs:    data/processed/")
    print(" Graphic figures: outputs/figures/")
    print(" To launch UI:    python main.py --dashboard")
    print("            or:   streamlit run dashboard.py")
    print("=" * 65 + "\n")


def launch_dashboard():
    """Launch the interactive Streamlit dashboard."""
    dashboard_path = PROJECT_ROOT / "dashboard.py"
    print(f"\nLaunching Streamlit Dashboard from {dashboard_path}...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard closed.")
    except Exception as exc:
        print(f"\nError launching dashboard: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Public Health Outbreak Monitoring & Distributed Analytics Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run entire end-to-end pipeline
  python main.py --clean      # Clean raw dataset
  python main.py --threading  # Run multithreading benchmark
  python main.py --spark      # Run PySpark distributed aggregations
  python main.py --graphs     # Generate visual plots
  python main.py --dashboard  # Launch interactive Streamlit dashboard
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline stages (cleaning, threading, spark, graphs)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run only data cleaning & preprocessing stage",
    )
    parser.add_argument(
        "--threading",
        action="store_true",
        help="Run only multithreading performance benchmark",
    )
    parser.add_argument(
        "--spark",
        action="store_true",
        help="Run only Apache Spark distributed analysis",
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Generate only visualization figures",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit interactive dashboard",
    )

    args = parser.parse_args()

    # If no flags are provided, run all stages
    if not (args.all or args.clean or args.threading or args.spark or args.graphs or args.dashboard):
        run_all()
        return

    if args.clean:
        clean_health_data()

    if args.threading:
        run_threading_benchmark()

    if args.spark:
        run_spark_analysis()

    if args.graphs:
        generate_graphs()

    if args.all:
        run_all()

    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
