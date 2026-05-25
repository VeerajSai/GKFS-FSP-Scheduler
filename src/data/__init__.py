"""Data processing utilities for FSP Auto Switch."""

__all__ = ["PlantDataProcessor", "FSPDataProcessor"]


def __getattr__(name):
    if name in {"PlantDataProcessor", "FSPDataProcessor"}:
        from .csv_to_parquet import FSPDataProcessor

        return FSPDataProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
