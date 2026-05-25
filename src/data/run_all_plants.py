#!/usr/bin/env python
"""
Run csv_to_parquet.py for each plant in parallel.
"""

import subprocess
import polars as pl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get unique stations
def get_unique_stations():
    """Get all unique stations from the actual data."""
    csv_path = Path("./data/raw/actualdata.csv")
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return []

    try:
        df = pl.read_csv(csv_path, columns=["sscode"])
        unique_stations = sorted(df["sscode"].unique().to_list())
        return unique_stations
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return []


def process_plant(station: str) -> tuple[str, bool]:
    """Process a single plant/station."""
    try:
        logger.info(f"Starting processing for {station}...")

        cmd = [
            "python",
            "src/data/csv_to_parquet.py",
            "--input", "./data/raw",
            "--output", "./data/processed",
            "--station", station
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per plant
        )

        if result.returncode == 0:
            logger.info(f" Successfully processed {station}")
            return station, True
        else:
            logger.error(f" Failed to process {station}")
            if result.stderr:
                logger.error(f"  Error: {result.stderr[:500]}")
            return station, False

    except subprocess.TimeoutExpired:
        logger.error(f" Timeout processing {station} (exceeded 1 hour)")
        return station, False
    except Exception as e:
        logger.error(f" Exception processing {station}: {e}")
        return station, False


def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("CSV to Parquet: Processing All Plants")
    logger.info("=" * 70)

    # Get unique stations
    stations = get_unique_stations()
    if not stations:
        logger.error("No stations found. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(stations)} unique stations")

    # Process stations in parallel
    max_workers = min(4, len(stations))  # Use up to 4 parallel workers
    logger.info(f"Processing with {max_workers} parallel workers...\n")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_plant, station): station for station in stations}

        for i, future in enumerate(as_completed(futures), 1):
            station, success = future.result()
            results[station] = success
            logger.info(f"[{i}/{len(stations)}] Completed {station}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Processing Summary")
    logger.info("=" * 70)

    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    if failed > 0:
        logger.warning(f" Successful: {successful}/{len(results)}")
        logger.warning(f" Failed: {failed}/{len(results)}")

        logger.info("\nFailed stations:")
        for station, success in results.items():
            if not success:
                logger.info(f"  - {station}")
    else:
        logger.info(f" All {successful} stations processed successfully!")

    logger.info("\nOutput:")
    logger.info(f"  Parquet files: ./data/processed/parquet/")
    logger.info(f"  CSV files: ./data/processed/csv/")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
