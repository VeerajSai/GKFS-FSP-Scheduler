# Wind Plant Topology Classification Documentation

## Overview

This document describes the methodology and implementation for classifying wind power plants into topological categories based on their geographic location. The classification system extracts plant locations from `plant_details.xlsx` and categorizes each plant/turbine into one of three categories:

- **Coastal Side**: Plants located near the coastline
- **Plateau**: Plants located in interior plateau regions
- **Western Ghats**: Plants located in the Western Ghats mountain range

## Purpose and Scope

The topology classification system serves multiple purposes:

1. **Geographic Analysis**: Understand the distribution of wind plants across different topological regions
2. **Model Training**: Enable topology-based model training and analysis
3. **Performance Analysis**: Compare plant performance across different topological categories
4. **Resource Planning**: Support strategic planning for new plant locations

## Data Source

### Input File: `plant_details.xlsx`

The Excel file contains detailed information about wind and solar power plants, including:

- **plant_code**: Unique identifier for the plant
- **plant_name**: Name of the power plant
- **turbine**: Turbine identifier (individual turbines within a plant)
- **registered_name**: Registered name of the plant/turbine
- **latitude**: Latitude coordinate (decimal degrees)
- **longitude**: Longitude coordinate (decimal degrees)

**File Statistics:**
- Total records: ~1,267 rows
- Each row represents an individual turbine location
- Coordinates are in decimal degrees (WGS84)

### Data Validation

The classifier validates:
- Presence of required columns (latitude, longitude)
- Non-null coordinate values
- Valid coordinate ranges (latitude: -90 to 90, longitude: -180 to 180)

## Classification Methodology

### Hybrid Classification Approach

The system uses a **hybrid approach** combining:

1. **Geographic Boundary Analysis**: Predefined lat/long ranges for major regions
2. **Elevation Data**: API-based elevation lookup for precise classification
3. **Distance Calculations**: Distance to coastline for coastal classification

### Geographic Boundaries

#### Western Ghats Region

The Western Ghats is a mountain range running parallel to India's western coast. The geographic boundary is defined as:

- **Latitude Range**: 8N to 21N
- **Longitude Range**: 73E to 77E

This covers the approximate extent of the Western Ghats mountain range from the southern tip of India to Gujarat.

#### Coastal Regions

Coastal classification is based on distance to the coastline. The system uses a simplified coastline representation with key coastal points:

- Southern tip (Kanyakumari): 8N, 77E
- Kerala coast: Multiple points from 9N to 11N
- Karnataka coast: 13N to 14N
- Maharashtra coast: 15N to 19N (includes Sample Plant region)
- Gujarat coast: 20N to 21N

**Coastal Distance Threshold**: 50 km from the nearest coastline point

#### Plateau Regions

The Deccan Plateau is the default category for interior regions that:
- Are not within the Western Ghats geographic boundary
- Are more than 50 km from the coastline
- Do not meet elevation criteria for Western Ghats

### Elevation-Based Classification

Elevation data is obtained from the Open-Elevation API (https://api.open-elevation.com), a free, open-source elevation service.

#### Elevation Thresholds

| Category | Elevation Range | Notes |
|----------|----------------|-------|
| **Coastal Side** | 100 m | Low-lying coastal areas |
| **Plateau** | 100 m - 500 m | Interior plateau regions |
| **Western Ghats** | 500 m | High elevation mountain regions |

#### Elevation Data Caching

To improve performance and reduce API calls:
- Elevation data is cached locally in `data/interim/elevation_cache.json`
- Cache key format: `"lat,lon"` (rounded to 3 decimal places)
- Cache persists between runs to avoid redundant API calls

### Classification Logic Flow

```
For each plant location (lat, lon):
  1. Validate coordinates
  2. Calculate distance to nearest coastline point
  3. Check if within Western Ghats geographic boundary
  4. Fetch elevation (if enabled and not cached)
  5. Apply classification rules:

     IF elevation available:
       - IF in Western Ghats region AND elevation  500m  "Western Ghats"
       - ELIF distance_to_coast  50km AND elevation  100m  "Coastal Side"
       - ELIF elevation  100m  "Coastal Side"
       - ELIF 100m < elevation  500m  "Plateau"
       - ELIF elevation > 500m  "Western Ghats" (if in region) or "Plateau"

     ELSE (elevation unavailable):
       - IF in Western Ghats region AND distance_to_coast > 50km  "Western Ghats"
       - ELIF distance_to_coast  50km  "Coastal Side"
       - ELSE  "Plateau"
  6. Store classification result with metadata
```

### Classification Priority

1. **Coastal Side**: Takes precedence if within 50 km of coastline
2. **Western Ghats**: Requires both geographic region match AND elevation threshold (if elevation available)
3. **Plateau**: Default category for interior regions

## Implementation Details

### Module Structure

**File**: `src/analysis/plant_topology_classifier.py`

#### Main Classes

- **`TopologyConfig`**: Configuration dataclass with all classification parameters
- **`PlantTopologyClassifier`**: Main classifier class with methods:
  - `load_plant_data()`: Load Excel file
  - `classify_location()`: Classify single location
  - `classify_all_plants()`: Batch classification
  - `generate_summary_statistics()`: Generate summary by category
  - `export_results()`: Export to CSV

#### Key Methods

**`_fetch_elevation(lat, lon)`**
- Fetches elevation from Open-Elevation API
- Implements caching to avoid redundant calls
- Includes retry logic for failed requests
- Returns `None` if API unavailable (falls back to boundary-based classification)

**`_distance_to_coast(lat, lon)`**
- Calculates minimum distance to coastline using geodesic distance
- Uses simplified coastline representation
- Returns distance in kilometers

**`_is_in_western_ghats_region(lat, lon)`**
- Checks if coordinates fall within Western Ghats geographic boundary
- Simple rectangular boundary check

**`classify_location(lat, lon, use_elevation)`**
- Main classification logic
- Returns tuple: `(category, metadata_dict)`
- Metadata includes: elevation, distance_to_coast, in_western_ghats_region, classification_method

### Configuration File

**File**: `configs/topology_config.yaml`

Contains all configurable parameters:
- Input/output paths
- Elevation thresholds
- Geographic boundaries
- API settings
- Classification rules

### Output Format

#### Main Output: `plant_topology_classifications.csv`

Contains all original columns plus:
- **topology_category**: Classification result (Coastal Side, Plateau, Western Ghats)
- **elevation_m**: Elevation in meters (if available)
- **distance_to_coast_km**: Distance to nearest coastline in kilometers
- **in_western_ghats_region**: Boolean indicating if in Western Ghats geographic boundary
- **classification_method**: Method used (hybrid or boundary)

#### Summary Output: `plant_topology_classifications_summary.csv`

Aggregated statistics by category:
- Unique plant count
- Total turbine count
- Average/min/max elevation
- Average/min/max distance to coast

## Usage Examples

### Command Line Usage

```bash
# Basic usage (with elevation data)
python src/analysis/plant_topology_classifier.py

# Skip elevation API calls (boundary-based only)
python src/analysis/plant_topology_classifier.py --no-elevation

# Custom input/output paths
python src/analysis/plant_topology_classifier.py \
    --excel-path "path/to/plants.xlsx" \
    --output-path "path/to/output.csv"

# Use custom config file
python src/analysis/plant_topology_classifier.py --config configs/topology_config.yaml
```

### Python API Usage

```python
from src.analysis.plant_topology_classifier import (
    PlantTopologyClassifier,
    TopologyConfig
)

# Initialize with default config
classifier = PlantTopologyClassifier()

# Or with custom config
config = TopologyConfig(
    excel_path="plant_details.xlsx",
    output_path="results/classifications.csv"
)
classifier = PlantTopologyClassifier(config)

# Classify all plants
df = classifier.classify_all_plants(use_elevation=True)

# Export results
classifier.export_results(df)

# Generate summary
summary = classifier.generate_summary_statistics(df)
print(summary)

# Classify single location
category, metadata = classifier.classify_location(17.27, 73.84)
print(f"Category: {category}")
print(f"Elevation: {metadata['elevation']} m")
print(f"Distance to coast: {metadata['distance_to_coast']} km")
```

## Limitations and Assumptions

### Geographic Boundaries

1. **Simplified Coastline**: The coastline representation uses a simplified set of key points. For more precise coastal classification, a detailed coastline shapefile would be needed.

2. **Rectangular Western Ghats Boundary**: The Western Ghats boundary is approximated as a rectangle. The actual mountain range has irregular boundaries that may not be perfectly captured.

3. **Fixed Distance Threshold**: The 50 km coastal distance threshold is a fixed value. In reality, coastal influence may vary by region.

### Elevation Data

1. **API Dependency**: Classification quality depends on elevation API availability. The system falls back to boundary-based classification if the API is unavailable.

2. **API Rate Limiting**: The Open-Elevation API may have rate limits. The system includes delays between requests, but large datasets may take significant time.

3. **Elevation Accuracy**: API elevation data may have varying accuracy depending on location and data source.

### Classification Rules

1. **Binary Categories**: Plants are assigned to a single category. In reality, some locations may have characteristics of multiple categories.

2. **Elevation vs. Geography**: The system prioritizes elevation when available, but some high-elevation coastal areas may be misclassified.

3. **Regional Variations**: The classification rules are designed for Indian geography. Different thresholds may be needed for other regions.

### Data Quality

1. **Coordinate Accuracy**: Classification accuracy depends on the accuracy of input coordinates. Inaccurate coordinates will lead to incorrect classifications.

2. **Missing Data**: Plants with missing or invalid coordinates are classified as "Unknown".

## Future Enhancements

1. **Detailed Coastline Data**: Integrate detailed coastline shapefiles for more accurate coastal classification

2. **Elevation Database**: Use local elevation database (e.g., SRTM data) instead of API calls for faster processing

3. **Machine Learning Classification**: Train ML models on known plant locations to improve classification accuracy

4. **Visualization**: Add map visualization showing plant locations colored by topology category

5. **Regional Customization**: Allow region-specific classification rules and thresholds

6. **Uncertainty Quantification**: Provide confidence scores for classifications

## References

- Open-Elevation API: https://api.open-elevation.com
- Western Ghats: https://en.wikipedia.org/wiki/Western_Ghats
- Geopy library: https://geopy.readthedocs.io/

## Contact

For questions or issues related to topology classification, please contact the AI/ML Team.

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Author**: AI/ML Team
