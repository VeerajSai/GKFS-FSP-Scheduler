"""
Quick Start Script for Streamlit App
=====================================

Run this script to check dependencies and launch the Streamlit application.

Usage:
    python launch_app.py
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn'
    ]

    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing

def main():
    print("=" * 70)
    print(" GKFS Auto Switch - Streamlit Application Launcher")
    print("=" * 70)

    # Check Python version
    print(f"\n Python version: {sys.version.split()[0]}")

    if sys.version_info < (3, 8):
        print(" Python 3.8+ required")
        sys.exit(1)

    # Check dependencies
    print("\n Checking dependencies...")
    missing = check_dependencies()

    if missing:
        print(f"\n Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print(" All core dependencies installed")

    # Check data
    data_dirs = [
        Path("data/processed"),
        Path("data/interim")
    ]

    data_found = False
    for data_dir in data_dirs:
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                print(f"\n Data found: {len(parquet_files)} parquet file(s) in {data_dir}")
                data_found = True
                break

    if not data_found:
        print("\n  Warning: No data files found in data/processed/ or data/interim/")
        print("   Please ensure your data is in the correct location.")
        print("   You can still launch the app, but data loading will fail.")

    # Launch app
    print("\n" + "=" * 70)
    print(" Launching Streamlit Application...")
    print("=" * 70)
    print("\nThe app will open in your default browser.")
    print("If it doesn't, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")

    try:
        subprocess.run([
            sys.executable,
            "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n Server stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
