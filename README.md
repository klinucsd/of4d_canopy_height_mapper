# Canopy Height Mapping Pipeline

A Python pipeline for generating wall-to-wall canopy height maps by combining satellite data from multiple sources using machine learning.

## Overview

This pipeline generates high-resolution canopy height maps by training a Random Forest model on GEDI lidar data and predicting heights using Sentinel-2, Sentinel-1, and SRTM data.

**Key Features:**
- No Google Earth Engine account required
- No computation limits
- Parallel processing support
- Comprehensive validation and visualization tools
- Model uncertainty quantification

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| **GEDI L2A** | Spaceborne lidar | Training labels (canopy height measurements) |
| **Sentinel-2** | Multispectral optical | Predictor variables (spectral bands, indices) |
| **Sentinel-1** | Synthetic aperture radar | Optional predictor (backscatter, texture) |
| **COP30 / SRTM** | Digital elevation model | Topographic features (elevation, slope) |

## Installation

```bash
pip install rasterio numpy pandas scikit-learn requests \
    pystac-client planetary-computer earthaccess \
    h5py pyproj scipy matplotlib seaborn joblib python-dotenv
```

### Credentials Setup

This pipeline requires a free NASA EarthData account to download GEDI L2A and SRTM data.

**Step 1: Get a NASA EarthData account**

Register at: https://urs.earthdata.nasa.gov/

**Step 2: Create your `.env` file**

```bash
cp .env.example .env
```

**Step 3: Edit `.env` with your credentials**

```bash
# Using a text editor
nano .env
# or
vim .env
```

Your `.env` file should look like this:

```bash
EARTHDATA_USERNAME=your_actual_username
EARTHDATA_PASSWORD=your_actual_password
OPENTOPOGRAPHY_API_KEY=your_opentopography_api_key
```

**Optional**: Get a free OpenTopography API key at https://portal.opentopography.org/ to avoid rate limits when downloading elevation data.

> **Note**: The `.env` file is ignored by git (see `.gitignore`) and will never be uploaded to GitHub, keeping your credentials safe.

## Quick Start

### Option 1: Command Line

```bash
python complete_canopy_height_pipeline.py
```

### Option 2: Python API

```python
from complete_canopy_height_pipeline import *
import visualizations as viz
import validation as val

# Configure
bbox = [-117.1, 32.7, -116.9, 32.9]  # [min_lon, min_lat, max_lon, max_lat]
start_date = '2022-01-01'
end_date = '2023-12-31'
output_dir = 'outputs'

# Download data
gedi_csv = download_gedi_earthaccess(bbox, start_date, end_date,
                                     f'{output_dir}/gedi.csv')
s2_path = download_sentinel2_mpc(bbox, start_date, end_date,
                                 f'{output_dir}/s2.tif')
s1_path = download_sentinel1_mpc(bbox, start_date, end_date,
                                 f'{output_dir}/s1.tif')  # Optional
topo_path = download_srtm_opentopography(bbox, f'{output_dir}/topo.tif')  # COP30 by default

# Train model
X, y, features = extract_features(gedi_csv, s2_path, s1_path, topo_path)
model = train_model(X, y, features)

# Generate map
predict_map(model, s2_path, s1_path, topo_path,
            f'{output_dir}/height_map.tif')
```

### Option 3: Run Examples

```bash
cd examples
python example_quick_demo.py        # Fast demo (5-15 min)
python example_full_pipeline.py     # Complete workflow (1-3 hrs)
```

## Project Structure

```
canopy_height_app/
├── complete_canopy_height_pipeline.py       # Main pipeline module
├── visualizations.py                        # Plotting and visualization tools
├── validation.py                            # Model validation metrics
├── .env.example                             # Environment variables template
├── .gitignore                               # Git ignore patterns
├── examples/
│   ├── example_quick_demo.py                # Minimal working example
│   ├── example_full_pipeline.py             # Complete workflow
│   └── README.md                            # Examples documentation
├── README.md                                # Main documentation (this file)
├── PIPELINE.md                              # Complete technical guide
└── API_REFERENCE.md                         # Detailed API documentation
```

## Core Functions

### Data Acquisition
- `download_gedi_earthaccess()` - Download GEDI L2A data via NASA EarthData
- `download_sentinel2_mpc()` - Download Sentinel-2 via Microsoft Planetary Computer
- `download_sentinel1_mpc()` - Download Sentinel-1 GRD data (optional, may return None)
- `download_srtm_opentopography()` - Download COP30 or SRTM elevation data

### Model Training
- `extract_features()` - Extract and align features from all data sources
- `train_model()` - Train Random Forest regressor with cross-validation

### Prediction
- `predict_map()` - Generate wall-to-wall canopy height prediction

### Validation & Visualization (visualizations.py)
- `plot_model_validation()` - Comprehensive validation plots (scatter, residuals, feature importance)
- `plot_canopy_height_map()` - Visualize predicted canopy height map (reprojected to WGS84)
- `plot_canopy_height_stats()` - Histogram and statistics panel
- `create_pipeline_summary()` - Multi-panel pipeline overview

### Validation Metrics (validation.py)
- `validate_predictions()` - Calculate accuracy metrics (R², RMSE, bias)

## Output Files

The pipeline generates the following outputs:

| File | Description |
|------|-------------|
| `height_map.tif` | Predicted canopy height (GeoTIFF) |
| `model.pkl` | Trained Random Forest model |
| `validation_report.json` | Model performance metrics |
| `feature_importance.png` | Feature importance plot |
| `prediction_scatter.png` | Validation scatter plot |
| `height_map_preview.png` | Preview of height map |

## Expected Model Performance

- **R²**: Typically 0.70 - 0.85
- **RMSE**: Typically 3 - 6 meters
- Performance varies with:
  - Forest type and structure complexity
  - Terrain ruggedness
  - GEDI data density in the area
  - Phenological match between GEDI and Sentinel-2

## Requirements

- Python 3.8+
- NASA EarthData account (for GEDI data) - register at https://urs.earthdata.nasa.gov/
- ~5-10 GB disk space per study area
- 8+ GB RAM recommended

## Common Issues

**Low model accuracy (R² < 0.70)**
- Increase GEDI data temporal range
- Ensure phenological match between datasets
- Add Sentinel-1 radar data

**Missing CRS information (Sentinel-1)**
- The pipeline includes automatic reprojection handling

**Rate limiting on OpenTopography**
- Add `OPENTOPOGRAPHY_API_KEY` to your `.env` file (get free key at https://portal.opentopography.org/)
- Pipeline automatically falls back to NASA EarthData SRTM if OpenTopography fails

**Shape mismatch (different UTM zones)**
- Pipeline automatically handles reprojection to a common grid

## Documentation

- `PIPELINE.md` - Complete technical guide with concepts and workflow details
- `API_REFERENCE.md` - Detailed API documentation for all functions
- `examples/README.md` - Example-specific documentation

## References

- [GEDI L2A User Guide (LPDAAC)](https://lpdaac.usgs.gov/documents/986/GEDI02_UserGuide_V2.pdf)
- [Sentinel-2 User Handbook (ESA/Copernicus)](https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_User_Handbook)
- [Sentinel-2 Documentation (Copernicus Data Space)](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html)
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)

## License

This project is provided as-is for research and educational purposes.
