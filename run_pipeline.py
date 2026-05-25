"""
GKFS Auto Switch - Full Pipeline Runner
========================================

Run this script to execute the entire ML pipeline:
1. Process CSVs Parquet
2. EDA Analysis (optional)
3. Model Training
4. Generate Performance Report

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --skip-processing  # Skip CSV processing if parquet exists
    python run_pipeline.py --data-only        # Only process data, no training

Maintainer: Project Team
Date: January 2026
"""

import subprocess
import sys
from pathlib import Path
import argparse

# Project paths
PROJECT_DIR = Path(__file__).parent
DATA_RAW = PROJECT_DIR / 'data' / 'raw'
DATA_PROCESSED = PROJECT_DIR / 'data' / 'processed'
NOTEBOOKS_DIR = PROJECT_DIR / 'notebooks'

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n {description}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"   Failed with exit code {result.returncode}")
        return False
    print(f"   Completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run GKFS Auto Switch Pipeline')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip CSV to Parquet if parquet exists')
    parser.add_argument('--data-only', action='store_true',
                       help='Only process data, skip model training')
    parser.add_argument('--station', default='SAMPLE_PSS',
                       help='Station to process (default: SAMPLE_PSS)')
    args = parser.parse_args()

    print_header("GKFS AUTO SWITCH - FULL PIPELINE")
    print(f"\nProject directory: {PROJECT_DIR}")
    print(f"Station: {args.station}")

    # =========================================================================
    # STEP 1: Check/Process Data
    # =========================================================================
    print_header("STEP 1: Data Processing (CSV  Parquet)")

    parquet_files = list(DATA_PROCESSED.glob('*.parquet'))

    if args.skip_processing and parquet_files:
        print(f"   Skipping - Found existing parquet files: {[f.name for f in parquet_files]}")
    else:
        # Check if raw CSVs exist
        csv_files = list(DATA_RAW.glob('*.csv'))
        if not csv_files:
            print(f"   No CSV files found in {DATA_RAW}")
            print("  Please add your CSV files to the data/raw folder:")
            print("    - actualdata.csv")
            print("    - scheduledata.csv")
            print("    - forecastdata.csv")
            sys.exit(1)

        print(f"  Found {len(csv_files)} CSV files")

        success = run_command([
            sys.executable,
            str(PROJECT_DIR / 'src' / 'data' / 'csv_to_parquet.py'),
            '--input', str(DATA_RAW),
            '--output', str(DATA_PROCESSED),
            '--station', args.station
        ], "Converting CSV to Parquet")

        if not success:
            print("\n Data processing failed. Check the error messages above.")
            sys.exit(1)

    if args.data_only:
        print_header("DATA PROCESSING COMPLETE")
        print("\nTo continue with model training, run:")
        print("  python run_pipeline.py --skip-processing")
        return

    # =========================================================================
    # STEP 2: Run Notebooks
    # =========================================================================
    print_header("STEP 2: Run Analysis & Training")

    notebooks = [
        ('01_EDA_Analysis.ipynb', 'Exploratory Data Analysis'),
        ('02_Model_Training.ipynb', 'Model Training & Evaluation'),
        ('03_Model_Performance.ipynb', 'Performance Showcase'),
    ]

    print("\n Notebooks to run:")
    for nb, desc in notebooks:
        print(f"   {nb}: {desc}")

    print("\n" + "-"*70)
    print("To run notebooks interactively:")
    print(f"  cd \"{PROJECT_DIR}\"")
    print(f"  jupyter notebook notebooks/")
    print("-"*70)

    # Option to run notebooks via nbconvert (headless)
    print("\nWould you like to run notebooks automatically? (headless mode)")
    print("This will execute all cells and save outputs to the notebooks.")

    try:
        response = input("\nRun automatically? [y/N]: ").strip().lower()
    except:
        response = 'n'

    if response == 'y':
        for nb_file, desc in notebooks:
            nb_path = NOTEBOOKS_DIR / nb_file
            if nb_path.exists():
                success = run_command([
                    sys.executable, '-m', 'jupyter', 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--inplace',
                    str(nb_path)
                ], f"Running {desc}")

                if not success:
                    print(f"\n Notebook {nb_file} had errors. Check the notebook for details.")
            else:
                print(f"   Notebook not found: {nb_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("PIPELINE COMPLETE")

    print("\n Output locations:")
    print(f"   Processed data: {DATA_PROCESSED}")
    print(f"   Trained models: {PROJECT_DIR / 'outputs' / 'models'}")
    print(f"   Predictions:    {PROJECT_DIR / 'outputs' / 'predictions'}")
    print(f"   Plots:          {PROJECT_DIR / 'outputs' / 'plots'}")

    print("\n Done! Check the outputs folder for results.")

if __name__ == '__main__':
    main()
