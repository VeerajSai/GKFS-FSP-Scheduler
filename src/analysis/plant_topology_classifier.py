"""
Wind Plant Topology Classification Module
==========================================

Extracts wind plant locations from plant_details.xlsx and classifies
each plant/turbine into topological categories (Coastal Side, Plateau, Western Ghats)
based on latitude/longitude coordinates using geographic boundaries and elevation data.

Maintainer: Project Team
Date: January 2026
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic

warnings.filterwarnings("ignore")


@dataclass
class TopologyConfig:
    """Configuration for topology classification."""
    excel_path: str = "plant_details.xlsx"
    output_path: str = "data/processed/plant_topology_classifications.csv"
    elevation_cache_path: str = "data/interim/elevation_cache.json"

    # Elevation thresholds (meters)
    coastal_max_elevation: float = 100.0
    plateau_min_elevation: float = 100.0
    plateau_max_elevation: float = 500.0
    western_ghats_min_elevation: float = 500.0

    # Geographic boundaries (lat/long ranges)
    western_ghats_lat_range: Tuple[float, float] = (8.0, 21.0)  # 8N to 21N
    western_ghats_lon_range: Tuple[float, float] = (73.0, 77.0)  # 73E to 77E

    # Coastal distance threshold (km)
    coastal_distance_threshold: float = 50.0

    # API settings
    elevation_api_url: str = "https://api.open-elevation.com/api/v1/lookup"
    api_request_delay: float = 0.1  # seconds between API calls
    max_retries: int = 3


class PlantTopologyClassifier:
    """Classifies wind plants into topological categories."""

    def __init__(self, config: Optional[TopologyConfig] = None):
        """Initialize the classifier with configuration."""
        self.config = config or TopologyConfig()
        self.elevation_cache = self._load_elevation_cache()

        # Indian coastline approximate coordinates (key points)
        # Enhanced representation with more points for better accuracy
        self.coastline_points = [
            # Southern tip and Tamil Nadu
            (8.0, 77.0),   # Kanyakumari
            (9.0, 78.0),   # Tamil Nadu coast
            (10.0, 79.0), # Tamil Nadu coast
            # Kerala coast
            (9.0, 76.0),   # Kerala coast
            (10.0, 76.0),  # Kerala coast
            (11.0, 75.5),  # Kerala coast
            # Karnataka coast
            (12.0, 74.5),  # Karnataka coast
            (13.0, 74.0),  # Karnataka coast
            (14.0, 74.0),  # Karnataka coast
            # Goa and Maharashtra coast (critical for Sample Plant)
            (15.0, 73.8),  # Goa coast
            (15.5, 73.5),  # Maharashtra coast
            (16.0, 73.2),  # Maharashtra coast
            (16.5, 73.0),  # Maharashtra coast
            (17.0, 73.0),  # Maharashtra coast (Sample Plant area)
            (17.5, 72.8),  # Maharashtra coast
            (18.0, 72.8),  # Maharashtra coast
            (18.5, 72.7),  # Maharashtra coast
            (19.0, 72.7),  # Maharashtra coast
            # Gujarat coast
            (20.0, 72.5),  # Gujarat coast
            (21.0, 72.0),  # Gujarat coast
            (22.0, 69.0),  # Gujarat coast (Kutch)
        ]

    def _load_elevation_cache(self) -> Dict[str, float]:
        """Load elevation cache from file."""
        cache_path = Path(self.config.elevation_cache_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load elevation cache: {e}")
        return {}

    def _save_elevation_cache(self):
        """Save elevation cache to file."""
        cache_path = Path(self.config.elevation_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.elevation_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save elevation cache: {e}")

    def _get_cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for coordinates (rounded to 3 decimal places)."""
        return f"{lat:.3f},{lon:.3f}"

    def _fetch_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Fetch elevation for given coordinates using API."""
        cache_key = self._get_cache_key(lat, lon)

        # Check cache first
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]

        # Try API
        for attempt in range(self.config.max_retries):
            try:
                payload = {
                    "locations": [{"latitude": lat, "longitude": lon}]
                }
                response = requests.post(
                    self.config.elevation_api_url,
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                if 'results' in data and len(data['results']) > 0:
                    elevation = data['results'][0].get('elevation', None)
                    if elevation is not None:
                        # Cache the result
                        self.elevation_cache[cache_key] = elevation
                        time.sleep(self.config.api_request_delay)
                        return elevation

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                print(f"Warning: Could not fetch elevation for ({lat}, {lon}): {e}")

        return None

    def _distance_to_coast(self, lat: float, lon: float) -> float:
        """Calculate minimum distance to coastline in kilometers."""
        point = (lat, lon)
        min_distance = float('inf')

        # Check distance to all coastline points
        for coast_point in self.coastline_points:
            distance = geodesic(point, coast_point).kilometers
            min_distance = min(min_distance, distance)

        # For western coast regions, also calculate perpendicular distance
        # Western coast roughly follows longitude ~73-74 for latitudes 8-22
        if 8.0 <= lat <= 22.0 and 70.0 <= lon <= 78.0:
            # Approximate west coast longitude varies by latitude
            # More accurate: closer to 73.0-73.5 for Maharashtra (where Sample Plant is)
            if 15.0 <= lat <= 20.0:  # Maharashtra region
                west_coast_lon = 73.0  # Closer to actual coast
            else:
                west_coast_lon = 73.5
            coastal_point = (lat, west_coast_lon)
            distance = geodesic(point, coastal_point).kilometers
            min_distance = min(min_distance, distance)

        return min_distance

    def _is_in_western_ghats_region(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Western Ghats geographic boundary."""
        lat_min, lat_max = self.config.western_ghats_lat_range
        lon_min, lon_max = self.config.western_ghats_lon_range

        return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

    def get_elevation_subclassification(self, elevation: Optional[float]) -> str:
        """
        Get elevation-based sub-classification for turbines within a plant.
        This is a secondary classification that refines the primary zone classification.

        Returns:
            Elevation sub-classification string
        """
        if elevation is None:
            return 'Unknown Elevation'

        if elevation >= 800:
            return 'Very High Elevation'
        elif elevation >= 500:
            return 'High Elevation'
        elif elevation >= 300:
            return 'Moderate Elevation'
        elif elevation >= 100:
            return 'Low Elevation'
        else:
            return 'Very Low Elevation'

    def classify_location(
        self,
        lat: float,
        lon: float,
        use_elevation: bool = True
    ) -> Tuple[str, Dict[str, any]]:
        """
        Classify a location into topology category using professional meteorological
        and geographical analysis principles.

        Classification Priority (Professional Approach):
        1. Geographic location (coastal proximity, terrain type)
        2. Distance to coast (marine boundary layer effects)
        3. Topographic features (mountain ranges, plateaus)
        4. Regional wind patterns (monsoon zones)

        Elevation is used as a SUB-CLASSIFICATION factor, not a primary classifier.
        A coastal plant remains coastal regardless of elevation.

        Returns:
            Tuple of (category, metadata_dict)
        """
        metadata = {
            'elevation': None,
            'distance_to_coast': None,
            'in_western_ghats_region': False,
            'classification_method': 'boundary',
            'elevation_subclassification': None
        }

        # Check if coordinates are valid
        if pd.isna(lat) or pd.isna(lon):
            return 'Unknown', metadata

        # Calculate distance to coast
        distance_to_coast = self._distance_to_coast(lat, lon)
        metadata['distance_to_coast'] = round(distance_to_coast, 2)

        # Check Western Ghats region
        in_wg_region = self._is_in_western_ghats_region(lat, lon)
        metadata['in_western_ghats_region'] = in_wg_region

        # Try to get elevation
        elevation = None
        if use_elevation:
            elevation = self._fetch_elevation(lat, lon)
            metadata['elevation'] = elevation
            metadata['classification_method'] = 'hybrid'
            if elevation is not None:
                metadata['elevation_subclassification'] = self.get_elevation_subclassification(elevation)

        # ========================================================================
        # PROFESSIONAL METEOROLOGICAL CLASSIFICATION
        # Based on geographic location, coastal proximity, and terrain features
        # Elevation is a SUB-CLASSIFICATION, not a primary classifier
        # ========================================================================

        # ------------------------------------------------------------------------
        # 1. COASTAL WIND ZONES
        # Classification: Based on proximity to sea (marine boundary layer effects)
        # Criteria: Distance to coast OR geographic coastal boundaries
        # Key Principle: Coastal plants remain coastal regardless of elevation
        # ------------------------------------------------------------------------

        # Identify coastal geographic regions
        is_west_coastal = False
        is_east_coastal = False

        # West Coast (Arabian Sea) - Strong SW monsoon influence
        if (15.0 <= lat <= 20.0 and 72.5 <= lon <= 74.5):  # Maharashtra coast (includes Sample Plant)
            is_west_coastal = True
        elif (8.0 <= lat <= 15.0 and 73.0 <= lon <= 77.0):  # Karnataka/Kerala coast
            is_west_coastal = True
        elif (20.0 <= lat <= 22.0 and 72.0 <= lon <= 73.0):  # Gujarat coast
            is_west_coastal = True

        # East Coast (Bay of Bengal) - NE monsoon influence
        if (8.0 <= lat <= 20.0 and 77.0 <= lon <= 82.0):  # Tamil Nadu, Andhra coast
            is_east_coastal = True

        # COASTAL CLASSIFICATION (Priority 1)
        # If within coastal boundaries AND close to coast, classify as coastal
        # Elevation does NOT override coastal classification

        # COASTAL CLASSIFICATION (Priority 1)
        # Key Principle: Plants in coastal geographic regions remain coastal
        # regardless of elevation, as long as they're within reasonable distance

        # If in coastal geographic region AND within 200km of coast = Coastal
        if is_west_coastal and distance_to_coast <= 200.0:
            # West coast plant (e.g., Sample Plant) - always coastal if in region
            return 'Coastal Wind Zone (West Coast)', metadata
        elif is_east_coastal and distance_to_coast <= 200.0:
            # East coast plant - always coastal if in region
            return 'Coastal Wind Zone (East Coast)', metadata

        # Distance-based coastal classification (for locations not in specific coastal regions)
        if distance_to_coast <= 50.0:
            # True coastal zone (within 50km) - strong marine influence
            return 'Coastal Wind Zone', metadata
        elif distance_to_coast <= 150.0:
            # Coastal influence zone (50-150km) - moderate marine influence
            # Check if it's in Western Ghats with high elevation (could be mountain)
            if in_wg_region and elevation is not None and elevation >= 800 and distance_to_coast > 100:
                # Very high elevation in WG region, far from coast = mountain
                return 'Mountain Wind Zone', metadata
            else:
                return 'Coastal Influence Zone', metadata

        # ------------------------------------------------------------------------
        # 2. MOUNTAIN WIND ZONES (Western Ghats)
        # Classification: Based on being in mountain range AND far from coast
        # Criteria: In Western Ghats region + distance to coast > 100km
        # Key Principle: Must be far enough from coast to not have marine influence
        # ------------------------------------------------------------------------

        if in_wg_region:
            # Western Ghats region
            if distance_to_coast > 100.0:
                # Far from coast = true mountain zone (orographic effects dominate)
                if elevation is not None and elevation < 200:
                    # Low elevation in WG = valley/foothills
                    return 'Valley Wind Zone', metadata
                else:
                    return 'Mountain Wind Zone', metadata
            elif distance_to_coast > 50.0:
                # Moderate distance from coast in WG = transition zone
                if elevation is not None and elevation >= 500:
                    return 'Mountain Wind Zone', metadata
                elif elevation is not None and elevation >= 200:
                    return 'Valley Wind Zone', metadata
                else:
                    return 'Valley Wind Zone', metadata

        # ------------------------------------------------------------------------
        # 3. DESERT WIND ZONE (Thar Desert)
        # Classification: Based on specific geographic region
        # Criteria: Rajasthan/Gujarat desert region (24-30N, 69-76E)
        # Key Principle: Strong, consistent winds, low variability
        # ------------------------------------------------------------------------

        if (24.0 <= lat <= 30.0 and 69.0 <= lon <= 76.0):
            # Thar Desert region
            if elevation is not None and elevation >= 500:
                # High elevation desert edge = mountain
                return 'Mountain Wind Zone', metadata
            else:
                return 'Desert Wind Zone', metadata

        # ------------------------------------------------------------------------
        # 4. PLATEAU WIND ZONES (Deccan Plateau)
        # Classification: Interior regions, not coastal, not mountainous
        # Criteria: Distance to coast > 100km, not in Western Ghats
        # Key Principle: Moderate, variable winds typical of interior plateaus
        # ------------------------------------------------------------------------

        if (12.0 <= lat <= 20.0 and 74.0 <= lon <= 80.0):
            # Deccan Plateau region
            if distance_to_coast > 100.0 and not in_wg_region:
                # True interior plateau
                if elevation is not None:
                    if elevation >= 500:
                        return 'High Plateau Wind Zone', metadata
                    elif elevation >= 200:
                        return 'Plateau Wind Zone', metadata
                    else:
                        return 'Low Plateau Wind Zone', metadata
                else:
                    return 'Plateau Wind Zone', metadata

        # ------------------------------------------------------------------------
        # 5. NORTHERN REGIONS
        # Classification: Northern India (Himachal Pradesh, etc.)
        # Criteria: Latitude >= 26N
        # ------------------------------------------------------------------------

        if lat >= 26.0:
            if elevation is not None:
                if elevation >= 500:
                    return 'Mountain Wind Zone', metadata
                else:
                    return 'Plateau Wind Zone', metadata
            else:
                return 'Plateau Wind Zone', metadata

        # ------------------------------------------------------------------------
        # DEFAULT CLASSIFICATION (Fallback)
        # Based on available data (elevation, distance to coast, region)
        # ------------------------------------------------------------------------

        if elevation is not None:
            if elevation >= 500:
                # High elevation = likely mountain
                if distance_to_coast <= 150:
                    # But close to coast = coastal influence
                    return 'Coastal Influence Zone', metadata
                else:
                    return 'Mountain Wind Zone', metadata
            elif elevation >= 200:
                # Moderate elevation
                if distance_to_coast <= 150:
                    return 'Coastal Influence Zone', metadata
                else:
                    return 'Plateau Wind Zone', metadata
            else:
                # Low elevation
                if distance_to_coast <= 150:
                    return 'Coastal Wind Zone', metadata
                else:
                    return 'Low Plateau Wind Zone', metadata
        else:
            # Without elevation data
            if distance_to_coast <= 100:
                return 'Coastal Wind Zone', metadata
            elif in_wg_region:
                return 'Mountain Wind Zone', metadata
            else:
                return 'Plateau Wind Zone', metadata

    def load_plant_data(self) -> pd.DataFrame:
        """Load plant data from Excel file."""
        excel_path = Path(self.config.excel_path)

        if not excel_path.exists():
            raise FileNotFoundError(
                f"Excel file not found: {excel_path}. "
                f"Please ensure plant_details.xlsx is in the project root."
            )

        print(f"Loading plant data from {excel_path}...")
        df = pd.read_excel(excel_path)

        # Validate required columns
        required_cols = ['latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"Loaded {len(df)} records")
        return df

    def classify_all_plants(
        self,
        df: Optional[pd.DataFrame] = None,
        use_elevation: bool = True,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Classify all plants in the dataset.

        Args:
            df: DataFrame with plant data. If None, loads from Excel.
            use_elevation: Whether to use elevation data for classification.
            batch_size: Number of records to process before saving cache.

        Returns:
            DataFrame with topology_category column added.
        """
        if df is None:
            df = self.load_plant_data()

        print(f"\nClassifying {len(df)} locations...")
        print(f"Using elevation data: {use_elevation}")

        # Initialize result columns
        df = df.copy()
        df['topology_category'] = None
        df['elevation_m'] = None
        df['elevation_subclassification'] = None
        df['distance_to_coast_km'] = None
        df['in_western_ghats_region'] = False
        df['classification_method'] = 'boundary'

        # Classify each location
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing {idx + 1}/{len(df)}...")

            lat = row['latitude']
            lon = row['longitude']

            category, metadata = self.classify_location(lat, lon, use_elevation)

            df.at[idx, 'topology_category'] = category
            df.at[idx, 'elevation_m'] = metadata.get('elevation')
            df.at[idx, 'elevation_subclassification'] = metadata.get('elevation_subclassification')
            df.at[idx, 'distance_to_coast_km'] = metadata.get('distance_to_coast')
            df.at[idx, 'in_western_ghats_region'] = metadata.get('in_western_ghats_region', False)
            df.at[idx, 'classification_method'] = metadata.get('classification_method', 'boundary')

            # Save cache periodically
            if (idx + 1) % batch_size == 0:
                self._save_elevation_cache()

        # Final cache save
        self._save_elevation_cache()

        print(f"\nClassification complete!")
        return df

    def classify_turbines_by_elevation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify turbines within each plant by elevation bands.
        This creates sub-classifications for turbines based on their elevation.

        Returns:
            DataFrame with elevation_band column added
        """
        df = df.copy()
        df['elevation_band'] = None

        for plant_code in df['plant_code'].unique():
            plant_turbines = df[df['plant_code'] == plant_code].copy()

            if plant_turbines['elevation_m'].notna().sum() == 0:
                # No elevation data for this plant
                df.loc[df['plant_code'] == plant_code, 'elevation_band'] = 'No Elevation Data'
                continue

            elevations = plant_turbines['elevation_m'].dropna()
            if len(elevations) == 0:
                continue

            min_elev = elevations.min()
            max_elev = elevations.max()
            elev_range = max_elev - min_elev

            # Classify each turbine by elevation band
            for idx in plant_turbines.index:
                elev = plant_turbines.loc[idx, 'elevation_m']
                if pd.isna(elev):
                    df.at[idx, 'elevation_band'] = 'No Elevation Data'
                    continue

                # Relative classification within plant
                if elev_range > 100:  # Significant elevation variation
                    if elev >= max_elev - 0.2 * elev_range:
                        df.at[idx, 'elevation_band'] = 'Upper Elevation'
                    elif elev <= min_elev + 0.2 * elev_range:
                        df.at[idx, 'elevation_band'] = 'Lower Elevation'
                    else:
                        df.at[idx, 'elevation_band'] = 'Mid Elevation'
                else:
                    # Low variation - all similar elevation
                    df.at[idx, 'elevation_band'] = 'Uniform Elevation'

        return df

    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics by topology category."""
        if 'topology_category' not in df.columns:
            raise ValueError("DataFrame must have 'topology_category' column")

        summary = df.groupby('topology_category').agg({
            'plant_code': 'nunique',
            'plant_name': 'nunique',
            'turbine': 'count',
            'elevation_m': ['mean', 'min', 'max'],
            'distance_to_coast_km': ['mean', 'min', 'max']
        }).round(2)

        summary.columns = [
            'unique_plants', 'unique_plant_names', 'total_turbines',
            'avg_elevation_m', 'min_elevation_m', 'max_elevation_m',
            'avg_coast_distance_km', 'min_coast_distance_km', 'max_coast_distance_km'
        ]

        return summary

    def export_results(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """Export classification results to CSV."""
        output_path = output_path or self.config.output_path
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"\nResults exported to: {output_file}")

        # Also export summary
        summary = self.generate_summary_statistics(df)
        summary_path = output_file.parent / f"{output_file.stem}_summary.csv"
        summary.to_csv(summary_path)
        print(f"Summary statistics exported to: {summary_path}")

        return output_file, summary_path


def main():
    """Main function to run classification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify wind plants by topology"
    )
    parser.add_argument(
        '--excel-path',
        type=str,
        default='plant_details.xlsx',
        help='Path to Excel file with plant details'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='data/processed/plant_topology_classifications.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--no-elevation',
        action='store_true',
        help='Skip elevation API calls (use boundary-based classification only)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to topology config YAML file'
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TopologyConfig(**config_dict.get('topology', {}))
    else:
        config = TopologyConfig(
            excel_path=args.excel_path,
            output_path=args.output_path
        )

    # Create classifier
    classifier = PlantTopologyClassifier(config)

    # Classify all plants
    df = classifier.classify_all_plants(use_elevation=not args.no_elevation)

    # Export results
    classifier.export_results(df)

    # Print summary
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    summary = classifier.generate_summary_statistics(df)
    print(summary)
    print("\n" + "="*70)
    print("Category Distribution:")
    print(df['topology_category'].value_counts())
    print("="*70)


if __name__ == "__main__":
    main()
