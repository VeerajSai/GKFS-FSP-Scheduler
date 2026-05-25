"""
CSV to Parquet/CSV Data Processor (Polars Version)
===================================================

Converts raw Actual, Scheduled, and Forecasted CSV data into unified
parquet and CSV datasets for all FSPs (Forecasting Service Providers).

This script handles:
- Streaming/lazy evaluation for memory-efficient processing of large files
- Parsing nested JSON block data from CSV
- Merging data from all three sources on timestamp
- Saving as compressed parquet AND CSV for flexibility
- Processing all plants with consistent output format

Maintainer: Project Team
Date: January 2026
Updated: February 2026 - Migrated to Polars for performance
"""

import polars as pl
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import argparse
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FSPDataProcessor:
    """
    Process and merge Actual, Scheduled, and Forecast CSV data into parquet/CSV.

    The processor handles the complex nested JSON structure in the raw CSVs
    and transforms them into a clean timeseries format using Polars for
    high-performance data processing.
    """

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        sscode_filter: Optional[str] = None,
        batch_size: int = 50000,
        save_per_plant: bool = True
    ):
        """
        Initialize the processor.

        Parameters:
        -----------
        raw_data_dir : str
            Directory containing raw CSV files
        output_dir : str
            Directory to save processed files
        sscode_filter : str, optional
            Filter for specific station code (e.g., 'SAMPLE_PSS')
        batch_size : int
            Number of rows to process per batch (default: 10000)
        save_per_plant : bool
            If True, save separate files for each plant
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.sscode_filter = sscode_filter
        self.batch_size = batch_size
        self.save_per_plant = save_per_plant

        # Create output directories for both formats
        self.parquet_dir = self.output_dir / "parquet"
        self.csv_dir = self.output_dir / "csv"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # File mappings
        self.csv_files = {
            'actual': 'actualdata.csv',
            'schedule': 'scheduledata.csv',
            'forecast': 'forecastdata.csv'
        }

        logger.info(f"Initialized FSP Data Processor (Polars)")
        logger.info(f"  Raw data dir: {self.raw_data_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Parquet dir: {self.parquet_dir}")
        logger.info(f"  CSV dir: {self.csv_dir}")
        logger.info(f"  Batch size: {self.batch_size}")
        if sscode_filter:
            logger.info(f"  Filtering for: {sscode_filter}")

    def _parse_blocks_json(self, blocks_str: Any) -> List[Dict]:
        """Parse the nested JSON blocks column from CSV."""
        if blocks_str is None or blocks_str == '' or (isinstance(blocks_str, float)):
            return []

        try:
            # Handle different JSON formats
            if isinstance(blocks_str, list):
                return blocks_str

            # Replace NaN strings with null for JSON parsing
            blocks_str = str(blocks_str).replace('NaN', 'null')
            blocks_str = blocks_str.replace("'", '"')

            return json.loads(blocks_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse error: {e}")
            return []

    def _get_unique_stations(self, csv_file: Path) -> List[str]:
        """Get all unique station codes from a CSV file efficiently."""
        try:
            # Use lazy evaluation to only read the sscode column
            df = pl.scan_csv(csv_file, ignore_errors=True).select('sscode').unique().collect()
            return df['sscode'].to_list()
        except Exception as e:
            logger.error(f"Error reading stations from {csv_file}: {e}")
            return []

    def _process_forecast_data(self, csv_file: Path, stations_filter: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Process forecast data with Polars.

        The forecast CSV has a nested 'blocks' column containing JSON array
        of 96 blocks (15-min intervals for 24 hours).
        """
        logger.info(f"Processing forecast data from {csv_file.name}...")

        if not csv_file.exists():
            logger.error(f"File not found: {csv_file}")
            return pl.DataFrame()

        all_records = []

        # Read CSV in batches using Polars reader
        reader = pl.read_csv_batched(
            csv_file,
            batch_size=self.batch_size,
            ignore_errors=True,
            infer_schema_length=1000
        )

        batch_num = 0
        total_records = 0

        while True:
            batches = reader.next_batches(1)
            if batches is None:
                break

            chunk = batches[0]
            batch_num += 1

            # Filter by stations if specified
            if stations_filter:
                chunk = chunk.filter(pl.col('sscode').is_in(stations_filter))
            elif self.sscode_filter:
                chunk = chunk.filter(pl.col('sscode') == self.sscode_filter)

            if chunk.height == 0:
                continue

            # Process each row for blocks JSON
            for row in chunk.iter_rows(named=True):
                sscode = row.get('sscode', '')
                date = row.get('date', '')
                facode = row.get('facode', '')

                blocks = self._parse_blocks_json(row.get('blocks', ''))

                for block in blocks:
                    if not block:
                        continue

                    record = {
                        'date': date,
                        'sscode': sscode,
                        'forecast_facode': facode,
                        'block': block.get('block'),
                        'forecast_time': block.get('time'),
                        'forecast_avc': block.get('avc'),
                        'forecast_power': block.get('power'),
                        'forecast_windspeed': block.get('windspeed'),
                        'forecast_ghirr': block.get('ghirr'),
                        'forecast_flowrate': block.get('flowrate'),
                        'forecast_revno': block.get('revno'),
                        'forecast_source': 'forecast'
                    }
                    all_records.append(record)

            total_records = len(all_records)
            if batch_num % 50 == 0:
                logger.info(f"  Processed {batch_num} batches ({total_records:,} records)")

        if not all_records:
            logger.warning("No forecast data found")
            return pl.DataFrame()

        result = pl.DataFrame(all_records)
        logger.info(f"   Forecast: {result.height:,} records processed")

        # Clean up
        del all_records
        gc.collect()

        return result

    def _process_schedule_data(self, csv_file: Path, stations_filter: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Process schedule data with Polars.

        Schedule CSV has similar nested JSON structure.
        """
        logger.info(f"Processing schedule data from {csv_file.name}...")

        if not csv_file.exists():
            logger.error(f"File not found: {csv_file}")
            return pl.DataFrame()

        all_records = []

        reader = pl.read_csv_batched(
            csv_file,
            batch_size=self.batch_size,
            ignore_errors=True,
            infer_schema_length=1000
        )

        batch_num = 0
        total_records = 0

        while True:
            batches = reader.next_batches(1)
            if batches is None:
                break

            chunk = batches[0]
            batch_num += 1

            # Filter by stations if specified
            if stations_filter:
                chunk = chunk.filter(pl.col('sscode').is_in(stations_filter))
            elif self.sscode_filter:
                chunk = chunk.filter(pl.col('sscode') == self.sscode_filter)

            if chunk.height == 0:
                continue

            for row in chunk.iter_rows(named=True):
                sscode = row.get('sscode', '')
                date = row.get('date', '')

                blocks = self._parse_blocks_json(row.get('blocks', ''))

                for block in blocks:
                    if not block:
                        continue

                    record = {
                        'date': date,
                        'sscode': sscode,
                        'block': block.get('block'),
                        'schedule_time': block.get('time'),
                        'schedule_avc': block.get('avc'),
                        'schedule_power': block.get('power'),
                        'schedule_windspeed': block.get('windspeed'),
                        'schedule_ghirr': block.get('ghirr'),
                        'schedule_flowrate': block.get('flowrate'),
                        'schedule_revno': block.get('revno'),
                        'schedule_source': 'schedule'
                    }
                    all_records.append(record)

            total_records = len(all_records)
            if batch_num % 50 == 0:
                logger.info(f"  Processed {batch_num} batches ({total_records:,} records)")

        if not all_records:
            logger.warning("No schedule data found")
            return pl.DataFrame()

        result = pl.DataFrame(all_records)
        logger.info(f"   Schedule: {result.height:,} records processed")

        del all_records
        gc.collect()

        return result

    def _process_actual_data(self, csv_file: Path, stations_filter: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Process actual data with Polars.

        Actual data has a different structure with blocks as separate columns
        (e.g., blocks[0].power, blocks[1].power, etc.)
        """
        logger.info(f"Processing actual data from {csv_file.name}...")

        if not csv_file.exists():
            logger.error(f"File not found: {csv_file}")
            return pl.DataFrame()

        all_records = []

        reader = pl.read_csv_batched(
            csv_file,
            batch_size=self.batch_size,
            ignore_errors=True,
            infer_schema_length=1000
        )

        batch_num = 0
        total_records = 0

        while True:
            batches = reader.next_batches(1)
            if batches is None:
                break

            chunk = batches[0]
            batch_num += 1

            # Filter by stations if specified
            if stations_filter:
                chunk = chunk.filter(pl.col('sscode').is_in(stations_filter))
            elif self.sscode_filter:
                chunk = chunk.filter(pl.col('sscode') == self.sscode_filter)

            if chunk.height == 0:
                continue

            columns = chunk.columns

            for row in chunk.iter_rows(named=True):
                sscode = row.get('sscode', '')
                date = row.get('date', '')

                # Process 96 blocks (0-95)
                for block_num in range(96):
                    prefix = f'blocks[{block_num}].'
                    time_col = f'{prefix}time'

                    # Check if column exists
                    if time_col not in columns:
                        continue

                    block_time = row.get(time_col)

                    if block_time is None or block_time == '':
                        continue

                    record = {
                        'date': date,
                        'sscode': sscode,
                        'block': block_num + 1,  # 1-indexed
                        'actual_time': block_time,
                        'actual_avc': row.get(f'{prefix}avc'),
                        'actual_power': row.get(f'{prefix}power'),
                        'actual_windspeed': row.get(f'{prefix}windspeed'),
                        'actual_ghirr': row.get(f'{prefix}ghirr'),
                        'actual_flowrate': row.get(f'{prefix}flowrate'),
                        'actual_source': 'actual'
                    }
                    all_records.append(record)

            total_records = len(all_records)
            if batch_num % 50 == 0:
                logger.info(f"  Processed {batch_num} batches ({total_records:,} records)")

        if not all_records:
            logger.warning("No actual data found")
            return pl.DataFrame()

        result = pl.DataFrame(all_records)
        logger.info(f"   Actual: {result.height:,} records processed")

        del all_records
        gc.collect()

        return result

    def _normalize_date(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize date column to consistent format for merging."""
        if df.height == 0:
            return df

        if 'date' in df.columns:
            # Convert to datetime and extract date string
            df = df.with_columns(
                pl.col('date').str.slice(0, 10).alias('date')
            )
        return df

    def _create_timestamp(self, df: pl.DataFrame, time_col: str) -> pl.DataFrame:
        """Create timestamp column from time string."""
        if df.height == 0:
            return df

        if time_col in df.columns:
            try:
                # Handle ISO format timestamps
                df = df.with_columns(
                    pl.col(time_col)
                    .str.replace('T', ' ')
                    .str.replace('Z', '')
                    .str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f', strict=False)
                    .alias('timestamp')
                )
            except Exception as e:
                logger.debug(f"Timestamp parsing failed: {e}")
                # Try alternative format
                try:
                    df = df.with_columns(
                        pl.col(time_col).cast(pl.Datetime).alias('timestamp')
                    )
                except Exception:
                    logger.warning(f"Could not parse timestamps from {time_col}")

        return df

    def merge_datasets(
        self,
        forecast_df: pl.DataFrame,
        schedule_df: pl.DataFrame,
        actual_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Merge all three datasets on timestamp using Polars.

        The merge strategy:
        1. Forecast as base (has multiple FSPs per timestamp)
        2. Left join with Schedule on (date, sscode, block)
        3. Left join with Actual on (date, sscode, block)
        """
        logger.info("Merging datasets...")

        # Normalize dates for consistent merging
        if forecast_df.height > 0:
            forecast_df = self._normalize_date(forecast_df)
        if schedule_df.height > 0:
            schedule_df = self._normalize_date(schedule_df)
        if actual_df.height > 0:
            actual_df = self._normalize_date(actual_df)

        # Create timestamp columns
        if forecast_df.height > 0 and 'forecast_time' in forecast_df.columns:
            forecast_df = self._create_timestamp(forecast_df, 'forecast_time')

        if schedule_df.height > 0 and 'schedule_time' in schedule_df.columns:
            schedule_df = self._create_timestamp(schedule_df, 'schedule_time')

        if actual_df.height > 0 and 'actual_time' in actual_df.columns:
            actual_df = self._create_timestamp(actual_df, 'actual_time')

        # Define merge keys
        merge_keys = ['date', 'sscode', 'block']

        # Start with forecast as base
        if forecast_df.height == 0:
            logger.warning("No forecast data - using schedule as base")
            merged = schedule_df.clone() if schedule_df.height > 0 else pl.DataFrame()
        else:
            merged = forecast_df.clone()

        # Merge with schedule
        if schedule_df.height > 0:
            schedule_cols = [c for c in schedule_df.columns if c.startswith('schedule_') or c in merge_keys]
            schedule_subset = schedule_df.select(schedule_cols).unique(subset=merge_keys)

            merged = merged.join(
                schedule_subset,
                on=merge_keys,
                how='left'
            )
            logger.info(f"   Merged with schedule: {merged.height:,} records")

        # Merge with actual
        if actual_df.height > 0:
            actual_cols = [c for c in actual_df.columns if c.startswith('actual_') or c in merge_keys]
            actual_subset = actual_df.select(actual_cols).unique(subset=merge_keys)

            merged = merged.join(
                actual_subset,
                on=merge_keys,
                how='left'
            )
            logger.info(f"   Merged with actual: {merged.height:,} records")

        # Sort by timestamp if available
        if 'timestamp' in merged.columns:
            merged = merged.sort(['sscode', 'timestamp'])
        elif 'date' in merged.columns and 'block' in merged.columns:
            merged = merged.sort(['sscode', 'date', 'block'])

        return merged

    def save_data(self, df: pl.DataFrame, filename_base: str) -> Tuple[Path, Path]:
        """Save processed data as both parquet and CSV."""
        parquet_path = self.parquet_dir / f"{filename_base}.parquet"
        csv_path = self.csv_dir / f"{filename_base}.csv"

        # Save parquet (compressed with snappy)
        df.write_parquet(parquet_path, compression='snappy')
        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        logger.info(f"   Parquet: {parquet_path.name} ({parquet_size_mb:.2f} MB)")

        # Save CSV
        df.write_csv(csv_path)
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        logger.info(f"   CSV: {csv_path.name} ({csv_size_mb:.2f} MB)")

        return parquet_path, csv_path

    def _get_all_stations(self) -> List[str]:
        """Get all unique station codes from all CSV files."""
        all_stations = set()

        for data_type, filename in self.csv_files.items():
            csv_file = self.raw_data_dir / filename
            if csv_file.exists():
                stations = self._get_unique_stations(csv_file)
                all_stations.update(stations)

        stations_list = sorted([s for s in all_stations if s])  # Remove empty strings
        logger.info(f"Found {len(stations_list)} unique stations across all files")
        return stations_list

    def process_single_station(self, station: str) -> Optional[pl.DataFrame]:
        """Process data for a single station."""
        logger.info(f"\n--- Processing station: {station} ---")

        # Process each data source
        forecast_df = self._process_forecast_data(
            self.raw_data_dir / self.csv_files['forecast'],
            stations_filter=[station]
        )

        schedule_df = self._process_schedule_data(
            self.raw_data_dir / self.csv_files['schedule'],
            stations_filter=[station]
        )

        actual_df = self._process_actual_data(
            self.raw_data_dir / self.csv_files['actual'],
            stations_filter=[station]
        )

        # Merge datasets
        merged_df = self.merge_datasets(forecast_df, schedule_df, actual_df)

        # Clean up intermediate dataframes
        del forecast_df, schedule_df, actual_df
        gc.collect()

        return merged_df if merged_df.height > 0 else None

    def run(self) -> Dict[str, Tuple[Path, Path]]:
        """
        Execute the full processing pipeline.

        Returns:
        --------
        Dict[str, Tuple[Path, Path]] : Dictionary mapping station names to (parquet_path, csv_path)
        """
        logger.info("=" * 70)
        logger.info("FSP DATA PROCESSING PIPELINE (POLARS)")
        logger.info("=" * 70)

        start_time = datetime.now()
        output_files = {}

        if self.sscode_filter:
            # Process single station
            stations_to_process = [self.sscode_filter]
        else:
            # Get all stations
            stations_to_process = self._get_all_stations()

        if self.save_per_plant:
            # Process and save each station separately
            for idx, station in enumerate(stations_to_process, 1):
                logger.info(f"\n[{idx}/{len(stations_to_process)}] Processing: {station}")

                try:
                    merged_df = self.process_single_station(station)

                    if merged_df is not None and merged_df.height > 0:
                        filename_base = f"{station.lower()}_dataset"
                        parquet_path, csv_path = self.save_data(merged_df, filename_base)
                        output_files[station] = (parquet_path, csv_path)

                        logger.info(f"  Records: {merged_df.height:,}")
                        if 'date' in merged_df.columns:
                            dates = merged_df['date'].unique().sort()
                            logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
                    else:
                        logger.warning(f"  No data found for {station}")

                    # Clean up
                    del merged_df
                    gc.collect()

                except Exception as e:
                    logger.error(f"  Error processing {station}: {e}")
                    continue
        else:
            # Process all data together
            logger.info("Processing all stations together...")

            forecast_df = self._process_forecast_data(
                self.raw_data_dir / self.csv_files['forecast']
            )

            schedule_df = self._process_schedule_data(
                self.raw_data_dir / self.csv_files['schedule']
            )

            actual_df = self._process_actual_data(
                self.raw_data_dir / self.csv_files['actual']
            )

            merged_df = self.merge_datasets(forecast_df, schedule_df, actual_df)

            if merged_df.height > 0:
                filename_base = "all_stations_dataset"
                parquet_path, csv_path = self.save_data(merged_df, filename_base)
                output_files['all_stations'] = (parquet_path, csv_path)

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Stations processed: {len(output_files)}")
        logger.info(f"  Parquet output dir: {self.parquet_dir}")
        logger.info(f"  CSV output dir: {self.csv_dir}")
        logger.info(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        return output_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert FSP CSV data to Parquet and CSV format (Polars version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all stations (saves each separately)
  python csv_to_parquet.py --input ./data/raw --output ./data/processed

  # Process specific station
  python csv_to_parquet.py --input ./data/raw --output ./data/processed --station SAMPLE_PSS

  # Process all stations into single file
  python csv_to_parquet.py --input ./data/raw --output ./data/processed --merge-all

  # Adjust batch size for memory
  python csv_to_parquet.py --input ./data/raw --output ./data/processed --batch-size 5000
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='./data/raw',
        help='Directory containing raw CSV files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/processed',
        help='Directory to save processed files (parquet/ and csv/ subdirs will be created)'
    )

    parser.add_argument(
        '--station', '-s',
        type=str,
        default=None,
        help='Filter for specific station code (e.g., SAMPLE_PSS, GANI_PSS_01)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=50000,
        help='Batch size for processing (default: 10000)'
    )

    parser.add_argument(
        '--merge-all', '-m',
        action='store_true',
        default=False,
        help='Merge all stations into a single file instead of separate files'
    )

    args = parser.parse_args()

    # Initialize and run processor
    processor = FSPDataProcessor(
        raw_data_dir=args.input,
        output_dir=args.output,
        sscode_filter=args.station,
        batch_size=args.batch_size,
        save_per_plant=not args.merge_all
    )

    output_files = processor.run()

    if output_files:
        print(f"\n Success! Processed {len(output_files)} station(s)")
        print(f"  Parquet files: {processor.parquet_dir}")
        print(f"  CSV files: {processor.csv_dir}")
    else:
        print("\n Processing failed. Check logs for details.")
        exit(1)


if __name__ == '__main__':
    main()
