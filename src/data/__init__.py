"""Data processing utilities for FSP Auto Switch."""

from .csv_to_parquet import FSPDataProcessor

# Backward compatibility alias
PlantDataProcessor = FSPDataProcessor

__all__ = ['PlantDataProcessor', 'FSPDataProcessor']
