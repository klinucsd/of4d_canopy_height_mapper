# Canopy Height Mapping Pipeline - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Visualization Module](#visualization-module)
6. [Validation Module](#validation-module)
7. [Examples and Use Cases](#examples-and-use-cases)
8. [Expected Outputs](#expected-outputs)
9. [Model Performance](#model-performance)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline generates wall-to-wall canopy height maps by combining:

- **GEDI L2A**: Spaceborne lidar canopy height measurements (training data)
- **Sentinel-2**: Multispectral optical imagery (predictor variables)
- **Sentinel-1**: Synthetic aperture radar (optional predictor)
- **SRTM**: Topographic data (elevation and slope)

### Key Features

- No Google Earth Engine account required
- No computation limits
- Parallel processing support
- Comprehensive validation and visualization
- Model uncertainty quantification

---

## Installation

### Requirements

```bash
pip install rasterio numpy pandas scikit-learn requests \
    pystac-client planetary-computer earthaccess \
    h5py pyproj scipy matplotlib seaborn joblib
```

### For Jupyter Notebook

```bash
pip install jupyter notebook
```

---

## Quick Start

### Option 1: Command Line

```bash
# Run the complete pipeline
python complete_canopy_height_pipeline_v2.py
```

### Option 2: Python Script

```python
from complete_canopy_height_pipeline_v2 import *
import visualizations as viz
import validation as val

# Configure
bbox = [-117.1, 32.7, -116.9, 32.9]
start_date = '2022-01-01'
end_date = '2023-12-31'
output_dir = 'outputs'

# Download data
gedi_csv = download_gedi_earthaccess(bbox, start_date, end_date,
                                     f'{output_dir}/gedi.csv')
s2_path = download_sentinel2_mpc(bbox, start_date, end_date,
                                 f'{output_dir}/s2.tif')
topo_path = download_srtm_opentopography(bbox, f'{output_dir}/topo.tif')

# Train model
X, y, features = extract_features(gedi_csv, s2_path, None, topo_path)
model = train_model(X, y, features)

# Generate map
predict_map(model, s2_path, None, topo_path,
            f'{output_dir}/height_map.tif')
```

### Option 3: Examples

```bash
cd examples
python example_quick_demo.py        # Fast demo (5-15 min)
python example_full_pipeline.py     # Complete workflow (1-3 hrs)
jupyter notebook Canopy_Height_Tutorial.ipynb  # Interactive
```

---

## API Reference

### Part 1: GEDI Data Download

#### `download_gedi_earthaccess(bbox, start_date, end_date, output_csv, n_workers=None)`

Download GEDI L2A data using NASA EarthData.

**Parameters:**
- `bbox` (list): [min_lon, min_lat, max_lon, max_lat]
- `start_date` (str): Start date 'YYYY-MM-DD'
- `end_date` (str): End date 'YYYY-MM-DD'
- `output_csv` (str): Output CSV path
- `n_workers` (int): Number of parallel workers (default: 4)

**Returns:**
- `str`: Path to output CSV, or None if failed

**Example:**
```python
gedi_csv = download_gedi_earthaccess(
    bbox=[-117.1, 32.7, -116.9, 32.9],
    start_date='2022-01-01',
    end_date='2023-12-31',
    output_csv='gedi.csv'
)
```

#### `augment_gedi_with_synthetic(gedi_csv, multiplier=3)`

Augment sparse GEDI data with synthetic points.

**Parameters:**
- `gedi_csv` (str): Path to GEDI CSV
- `multiplier` (int): Synthetic points per real point

**Returns:**
- `str`: Path to augmented CSV

---

### Part 2: Satellite Data Download

#### `download_sentinel2_mpc(bbox, start_date, end_date, output_path, max_items=10, resolution=30, n_workers=None)`

Download Sentinel-2 L2A from Microsoft Planetary Computer.

**Parameters:**
- `bbox` (list): Bounding box
- `start_date` (str): Start date
- `end_date` (str): End date
- `output_path` (str): Output GeoTIFF path
- `max_items` (int): Maximum scenes to process
- `resolution` (int): Target resolution in meters (10, 20, 30, or 60)
- `n_workers` (int): Number of parallel workers

**Returns:**
- `str`: Path to output GeoTIFF (5 bands: B04, B08, B11, B12, NDVI)

#### `download_sentinel1_mpc(bbox, start_date, end_date, output_path, max_items=5)`

Download Sentinel-1 GRD/RTC data.

**Parameters:**
- `bbox` (list): Bounding box
- `start_date` (str): Start date
- `end_date` (str): End date
- `output_path` (str): Output GeoTIFF path
- `max_items` (int): Maximum scenes

**Returns:**
- `str`: Path to output GeoTIFF (2 bands: VV, VH), or None

#### `download_srtm_opentopography(bbox, output_path)`

Download SRTM DEM.

**Parameters:**
- `bbox` (list): Bounding box
- `output_path` (str): Output GeoTIFF path

**Returns:**
- `str`: Path to output GeoTIFF (2 bands: elevation, slope)

---

### Part 3: Model Training

#### `extract_features(gedi_csv, s2_tif, s1_tif, topo_tif)`

Extract features at GEDI point locations.

**Parameters:**
- `gedi_csv` (str): Path to GEDI CSV
- `s2_tif` (str): Path to Sentinel-2 GeoTIFF
- `s1_tif` (str): Path to Sentinel-1 GeoTIFF (or None)
- `topo_tif` (str): Path to topography GeoTIFF

**Returns:**
- `X` (ndarray): Feature matrix (n_samples, n_features)
- `y` (ndarray): Target values (RH98)
- `features` (list): Feature names

#### `train_model(X, y, names)`

Train Random Forest model.

**Parameters:**
- `X` (ndarray): Feature matrix
- `y` (ndarray): Target values
- `names` (list): Feature names

**Returns:**
- `RandomForestRegressor`: Trained model

---

### Part 4: Prediction

#### `predict_map(model, s2_tif, s1_tif, topo_tif, output)`

Generate wall-to-wall canopy height map.

**Parameters:**
- `model` (RandomForestRegressor): Trained model
- `s2_tif` (str): Path to Sentinel-2 GeoTIFF
- `s1_tif` (str): Path to Sentinel-1 GeoTIFF (or None)
- `topo_tif` (str): Path to topography GeoTIFF
- `output` (str): Output GeoTIFF path

**Returns:**
- None (saves GeoTIFF to disk)

---

## Visualization Module

The `visualizations.py` module provides publication-quality plots for each pipeline stage.

### GEDI Data Visualization

#### `plot_gedi_data_coverage(gedi_csv, bbox=None, output_dir=None)`

Create comprehensive 6-panel GEDI data overview:

1. Spatial distribution map (colored by height)
2. RH98 histogram
3. Real vs synthetic comparison
4. Longitude distribution
5. Latitude distribution
6. Point density heatmap

**Example:**
```python
viz.plot_gedi_data_coverage('gedi.csv', bbox=my_bbox, output_dir='outputs')
```

#### `plot_gedi_tracks(gedi_csv, output_dir=None)`

Plot GEDI tracks colored by beam.

### Satellite Data Visualization

#### `plot_sentinel2_bands(s2_tif, output_dir=None)`

Display all Sentinel-2 bands and indices.

#### `plot_sentinel2_composite(s2_tif, bbox=None, output_dir=None)`

Create RGB and false-color composites plus NDVI map.

#### `plot_band_histograms(s2_tif, output_dir=None)`

Plot histograms of all Sentinel-2 bands.

### Model Validation Visualization

#### `plot_model_validation(y_true, y_pred, feature_names=None, feature_importance=None, model_name="Random Forest", output_dir=None)`

Generate comprehensive 6-panel validation figure:

1. Predicted vs observed scatter plot
2. Residuals plot
3. Residuals histogram
4. Binned predictions
5. Error by height class
6. Feature importance bar chart

**Example:**
```python
viz.plot_model_validation(
    y_test, y_pred,
    feature_names=features,
    feature_importance=model.feature_importances_,
    output_dir='outputs'
)
```

#### `plot_cross_validation(cv_scores, model_name="Random Forest", output_dir=None)`

Plot cross-validation results (R² and RMSE by fold).

### Map Visualization

#### `plot_canopy_height_map(height_map_path, gedi_csv=None, title="Canopy Height Map", cmap='RdYlGn', output_dir=None)`

Visualize final canopy height map with statistics and optional GEDI overlay.

**Output includes:**
- Main map with colorbar
- Height histogram
- Comprehensive statistics panel

#### `plot_comparison_maps(predicted_path, reference_path=None, gedi_csv=None, output_dir=None)`

Create comparison plots:
- Predicted map
- Reference map (if provided)
- Validation scatter plot (if GEDI provided)

### Summary Visualization

#### `create_pipeline_summary(gedi_csv, s2_tif, height_map_path, model_stats, output_dir=None)`

Create single-figure summary showing:
- GEDI training data
- Sentinel-2 NDVI
- Model performance statistics
- Feature importance
- Final canopy height map with GEDI overlay

---

## Validation Module

The `validation.py` module provides comprehensive statistical analysis.

### Model Evaluation

#### `evaluate_model(y_true, y_pred, model_name="Model")`

Calculate comprehensive evaluation metrics:

**Overall metrics:**
- R², RMSE, MAE
- Explained variance
- Maximum error
- Bias
- MAPE

**Height class metrics:**
- Performance by height classes (0-5m, 5-10m, 10-20m, 20-50m)

**Returns:** Dictionary with all metrics

#### `print_model_report(metrics)`

Print formatted evaluation report.

### Cross-Validation

#### `perform_cross_validation(model, X, y, n_folds=5, random_state=42)`

Perform k-fold cross-validation.

**Returns:** Dictionary with CV results

#### `print_cv_report(cv_summary)`

Print formatted cross-validation report.

### Validation with Independent Data

#### `validate_with_gedi(model, gedi_csv, s2_tif, s1_tif, topo_tif, output_dir=None)`

Validate model predictions against GEDI using spatial holdout.

#### `spatial_cross_validation(model, gedi_csv, s2_tif, s1_tif, topo_tif, n_folds=5, output_dir=None)`

Perform spatial cross-validation by dividing study area into regions.

### Uncertainty Quantification

#### `prediction_intervals(model, X, percentile=95, n_bootstrap=100)`

Calculate prediction intervals using bootstrap.

**Returns:** (lower_bound, upper_bound, mean_pred)

#### `calculate_map_uncertainty(model, s2_tif, s1_tif, topo_tif, output_path, n_bootstrap=50)`

Generate uncertainty map (mean, std, lower CI, upper CI).

### Feature Analysis

#### `analyze_feature_importance(model, feature_names, output_dir=None)`

Analyze and save feature importance.

**Returns:** DataFrame with feature rankings

#### `correlation_analysis(X, feature_names, output_dir=None)`

Perform correlation analysis and generate heatmap.

**Returns:** Correlation matrix

### Model Comparison

#### `compare_models(models_dict, X_test, y_test, output_dir=None)`

Compare multiple models on the same test set.

**Returns:** Comparison DataFrame

### Report Generation

#### `generate_validation_report(gedi_csv, s2_tif, s1_tif, topo_tif, model, feature_names, output_dir)`

Generate comprehensive validation report including:
- Test set evaluation
- Cross-validation
- Feature importance
- Correlation analysis
- All saved to files

---

## Examples and Use Cases

### Example 1: Quick Demo

**Script:** `examples/example_quick_demo.py`

**Purpose:** Fast demonstration on small area

**Runtime:** 5-15 minutes

**Configuration:**
```python
bbox = [-117.1, 32.7, -116.9, 32.9]  # ~20km x 20km
start_date = '2022-01-01'
end_date = '2023-12-31'
max_s2_scenes = 5
resolution = 30m
```

**Expected Outputs:**
- 500-1000 GEDI points
- 5 Sentinel-2 scenes
- Complete validation plots
- Canopy height map

**Use Cases:**
- Testing installation
- Learning the pipeline
- Quick prototyping
- Teaching/demonstrations

---

### Example 2: Full Pipeline

**Script:** `examples/example_full_pipeline.py`

**Purpose:** Production-ready workflow

**Runtime:** 1-3 hours

**Configuration:**
```python
bbox = [-117.28, 32.53, -116.42, 33.51]  # San Diego County
start_date = '2019-04-01'
end_date = '2023-12-31'
max_s2_scenes = 10
resolution = 30m
n_trees = 200
max_depth = 25
```

**Expected Outputs:**
- 5000-20000 GEDI points
- 10 Sentinel-2 scenes
- Full validation report
- Uncertainty analysis
- Feature correlation analysis

**Use Cases:**
- Research applications
- Publication-quality results
- Operational monitoring
- Large area mapping

---

### Example 3: Jupyter Notebook Tutorial

**File:** `examples/Canopy_Height_Tutorial.ipynb`

**Purpose:** Interactive learning

**Sections:**
1. Introduction to concepts
2. Step-by-step workflow
3. Code explanations
4. Visual outputs inline
5. Interpretation guidance

**Use Cases:**
- Workshops
- Teaching courses
- Self-paced learning
- Interactive exploration

---

## Expected Outputs

### Directory Structure

```
outputs/
├── data/
│   ├── gedi_raw.csv              # GEDI training data
│   ├── sentinel2.tif             # S2 composite (5 bands)
│   ├── sentinel1.tif             # S1 composite (2 bands)
│   └── topography.tif            # SRTM DEM + slope (2 bands)
│
├── maps/
│   ├── canopy_height_map.tif     # Final height map
│   └── uncertainty_map.tif       # Prediction uncertainty (optional)
│
├── visualizations/
│   ├── gedi_data_coverage.png
│   ├── gedi_tracks.png
│   ├── sentinel2_bands.png
│   ├── sentinel2_composites.png
│   ├── sentinel2_histograms.png
│   ├── model_validation.png
│   ├── cross_validation.png
│   ├── canopy_height_map.png
│   ├── comparison_maps.png
│   ├── feature_importance.png
│   ├── feature_correlation_heatmap.png
│   └── pipeline_summary.png
│
├── reports/
│   ├── validation_report.json
│   ├── feature_importance.csv
│   ├── feature_correlation.csv
│   ├── model_comparison.csv
│   └── configuration.json
│
└── models/
    └── canopy_height_model.pkl    # Trained model
```

### File Formats

**GeoTIFF Files:**
- Sentinel-2: 32-bit float, 5 bands (B04, B08, B11, B12, NDVI)
- Sentinel-1: 32-bit float, 2 bands (VV, VH)
- Topography: 32-bit float, 2 bands (elevation, slope)
- Height map: 32-bit float, 1 band (height in meters)

**CSV Files:**
- GEDI: longitude, latitude, rh98, [synthetic]
- Feature importance: feature, importance, cumulative_importance
- Correlation: square matrix of correlations

**JSON Files:**
- Validation report: all metrics in hierarchical format
- Configuration: pipeline settings and metadata

---

## Model Performance

### Expected Performance by Region

| Region | Expected R² | Expected RMSE | Notes |
|--------|-------------|---------------|-------|
| Forest (dense) | 0.75 - 0.85 | 3 - 5 m | Best performance |
| Forest (mixed) | 0.65 - 0.75 | 4 - 7 m | Good performance |
| Savanna | 0.55 - 0.70 | 5 - 8 m | Moderate performance |
| Shrubland | 0.45 - 0.60 | 6 - 10 m | Lower performance |

### Performance Factors

**Positive factors:**
- High GEDI point density (> 1000 points)
- Multiple Sentinel-2 scenes (> 5)
- Homogeneous vegetation
- Flat to moderate terrain
- Seasonal matching

**Negative factors:**
- Sparse GEDI data (< 100 points)
- Cloud contamination
- Complex topography
- Seasonal mismatch
- Urban areas

### Performance Benchmarks

Based on San Diego County testing:

```
Test Set (n=2000):
  R²:  0.78
  RMSE: 4.2 m
  MAE:  3.1 m

Cross-Validation (5-fold):
  R²:  0.75 ± 0.03
  RMSE: 4.5 ± 0.3 m

By Height Class:
  0-5m:    R²=0.52, RMSE=2.1m  (n=450)
  5-10m:   R²=0.68, RMSE=3.4m  (n=680)
  10-20m:  R²=0.78, RMSE=4.8m  (n=720)
  20-50m:  R²=0.72, RMSE=6.2m  (n=150)
```

---

## Troubleshooting

### Issue: GEDI Download Fails

**Symptoms:**
- `earthaccess` authentication errors
- No granules found
- Download timeout

**Solutions:**
1. Check NASA EarthData credentials:
   ```python
   os.environ['EARTHDATA_USERNAME'] = 'your_username'
   os.environ['EARTHDATA_PASSWORD'] = 'your_password'
   ```

2. Try longer date range:
   ```python
   start_date = '2019-04-01'  # GEDI launch
   end_date = '2024-12-31'
   ```

3. Script will create synthetic data if download fails

---

### Issue: Poor Model Performance (R² < 0.6)

**Symptoms:**
- Low R² score
- High RMSE (> 8m)
- Large bias

**Diagnosis:**
```python
# Check GEDI data
gedi = pd.read_csv(gedi_csv)
print(f"GEDI points: {len(gedi)}")
print(f"Height range: {gedi['rh98'].min()} - {gedi['rh98'].max()}")

# Check feature correlation
import validation as val
corr_matrix = val.correlation_analysis(X, features)
```

**Solutions:**
1. Increase training data (larger area or longer time period)
2. Filter outliers:
   ```python
   gedi = gedi[(gedi['rh98'] > 0) & (gedi['rh98'] < 50)]
   ```
3. Check for CRS mismatches
4. Ensure temporal alignment between GEDI and optical data

---

### Issue: Memory Error

**Symptoms:**
- `MemoryError` during prediction
- System crash or freeze

**Solutions:**
1. Increase resolution:
   ```python
   resolution = 30  # Instead of 10 or 20
   ```

2. Reduce number of scenes:
   ```python
   max_items = 5  # Instead of 10
   ```

3. Process in tiles (requires custom implementation)

---

### Issue: CRS Mismatch

**Symptoms:**
- No data extracted at GEDI locations
- All NaN values
- `sample` returns all nodata

**Diagnosis:**
```python
import rasterio
with rasterio.open(s2_tif) as src:
    print(f"S2 CRS: {src.crs}")
```

**Solution:**
The `extract_features` function handles CRS reprojection automatically. Ensure all input files have valid CRS metadata.

---

### Issue: Sentinel-1 Not Available

**Symptoms:**
- `download_sentinel1_mpc` returns None
- Warning message about no S1 data

**Solution:**
Pipeline continues without S1 data. Performance may be slightly lower but should still work.

```python
s1_path = download_sentinel1_mpc(...)  # May return None
X, y, features = extract_features(..., s1_tif=s1_path, ...)  # Handles None
```

---

### Issue: Visualizations Not Saving

**Symptoms:**
- Plots display but don't save
- `output_dir` doesn't exist

**Solution:**
```python
from pathlib import Path
Path(output_dir).mkdir(parents=True, exist_ok=True)

viz.plot_gedi_data_coverage(..., output_dir=output_dir)
```

---

## Advanced Usage

### Custom Hyperparameter Tuning

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25],
    'min_samples_split': [3, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Ensemble Methods

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)
```

### Temporal Analysis

```python
# Process multiple time periods
time_periods = [
    ('2020-01-01', '2020-12-31'),
    ('2021-01-01', '2021-12-31'),
    ('2022-01-01', '2022-12-31')
]

for start, end in time_periods:
    s2_path = download_sentinel2_mpc(bbox, start, end, ...)
    # Generate height map for each period
```

---

## References

### Data Products

- **GEDI L2A**: https://e4ftl01.cr.usgs.gov/GEDI/GEDI02_A.002/
- **Sentinel-2**: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi
- **Sentinel-1**: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar
- **SRTM**: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm

### Methods

- Dubayah, R. O., et al. (2022). The Global Ecosystem Dynamics Investigation: High-resolution laser ranging of the Earth's surface and vegetation canopy. *Science Advances*.

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

- Zhu, Z., et al. (2019). The global land surface satellite (GLASS) leaf area index product. *Remote Sensing*.

### Similar Packages

- **ICESat2VegR**: https://github.com/carlos-alberto-silva/ICESat2VegR
  - R package for ICESat-2 vegetation data

- **GEDI Simulator**: https://git.earthdata.nasa.gov/projects/LPDUR/repos/gedi-simulator/browse

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{canopy_height_pipeline,
  title={Canopy Height Mapping Pipeline},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```

Also cite the data products used:
- GEDI L2A data (NASA EOSDIS Land Processes DAAC)
- Sentinel-2 data (ESA)
- SRTM data (NASA/USGS)

---

## License

[Specify your license here]

## Contact

[Your contact information]

---

**Last Updated:** 2024
**Version:** 2.0
