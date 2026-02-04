# Canopy Height Mapping Examples

This directory contains example scripts and tutorials demonstrating the canopy height mapping pipeline.

## 📁 Contents

### 1. Quick Demo (`example_quick_demo.py`)
**Runtime:** 5-15 minutes
**Purpose:** Fast demonstration on a small area

```bash
python example_quick_demo.py
```

**Features:**
- Small study area (~0.2° x 0.2°)
- Limited scenes (5 Sentinel-2, 3 Sentinel-1)
- All visualizations included
- Perfect for testing and learning

**Outputs:** `demo_outputs/`
- Canopy height map
- All validation plots
- Trained model

---

### 2. Full Pipeline (`example_full_pipeline.py`)
**Runtime:** 1-3 hours
**Purpose:** Complete workflow for a larger area

```bash
python example_full_pipeline.py
```

**Features:**
- Full San Diego County (~0.86° x 0.98°)
- Maximum data (10 Sentinel-2 scenes)
- Comprehensive validation
- Cross-validation
- Uncertainty quantification

**Outputs:** `full_outputs/`
- Complete canopy height map
- All validation reports
- Feature analysis
- Comparison plots

---

### 3. Jupyter Notebook Tutorial (`Canopy_Height_Tutorial.ipynb`)
**Purpose:** Interactive, step-by-step learning

```bash
jupyter notebook Canopy_Height_Tutorial.ipynb
```

**Features:**
- Detailed explanations
- Code cells with output
- Visualizations inline
- Perfect for workshops and teaching

**Coverage:**
1. Introduction to the pipeline
2. GEDI data acquisition
3. Satellite data download
4. Feature extraction
5. Model training
6. Validation
7. Map generation
8. Analysis and interpretation

---

## 🚀 Quick Start

### Option 1: Run the Quick Demo
```bash
cd examples
python example_quick_demo.py
```

### Option 2: Launch Jupyter Notebook
```bash
cd examples
jupyter notebook Canopy_Height_Tutorial.ipynb
```

### Option 3: Run Full Pipeline
```bash
cd examples
python example_full_pipeline.py
```

---

## 📊 Expected Outputs

### Visualizations
All examples generate the following plots:

**GEDI Data:**
- `gedi_data_coverage.png` - Point distribution and statistics
- `gedi_tracks.png` - Track visualization

**Sentinel-2:**
- `sentinel2_bands.png` - Individual band displays
- `sentinel2_composites.png` - RGB and false-color composites
- `sentinel2_histograms.png` - Band value distributions

**Model Validation:**
- `model_validation.png` - 6-panel validation figure
  - Predicted vs observed
  - Residuals plot
  - Residuals histogram
  - Binned predictions
  - Error by height
  - Feature importance

**Cross-Validation:**
- `cross_validation.png` - Fold-by-fold results

**Final Maps:**
- `canopy_height_map.png` - Height map with statistics
- `comparison_maps.png` - Validation comparisons

**Summary:**
- `pipeline_summary.png` - Complete workflow overview

### Data Files
- `gedi_raw.csv` - GEDI training data
- `sentinel2.tif` - Sentinel-2 composite (5 bands)
- `sentinel1.tif` - Sentinel-1 composite (2 bands)
- `topography.tif` - SRTM DEM and slope (2 bands)
- `canopy_height_map.tif` - Final height map

### Reports
- `validation_report.json` - Complete validation metrics
- `feature_importance.csv` - Feature importance table
- `feature_correlation.csv` - Feature correlation matrix
- `configuration.json` - Pipeline configuration

### Model
- `canopy_height_model.pkl` - Trained Random Forest model

---

## 🎯 Customizing Examples

### Change Study Area

Edit the `bbox` parameter:

```python
# Format: [min_lon, min_lat, max_lon, max_lat]
bbox = [-117.1, 32.7, -116.9, 32.9]  # Small area
bbox = [-117.28, 32.53, -116.42, 33.51]  # San Diego County
```

### Change Time Period

```python
start_date = '2019-04-01'  # GEDI launch
end_date = '2023-12-31'
```

### Change Resolution

```python
resolution = 30  # 30m (default)
resolution = 20  # 20m (better quality, larger files)
resolution = 10  # 10m (best quality, very large)
```

---

## 📈 Model Performance Expectations

Based on testing in San Diego County:

| Metric | Expected Range | Good Performance |
|--------|---------------|------------------|
| R² | 0.6 - 0.85 | > 0.75 |
| RMSE | 3 - 8 m | < 5 m |
| MAE | 2 - 5 m | < 3 m |

**Factors affecting performance:**
- GEDI point density (more is better)
- Vegetation type variability
- Topographic complexity
- Cloud-free imagery availability

---

## ⚠️ Common Issues

### Issue: GEDI download fails
**Solution:** Script will create synthetic data for demonstration

### Issue: Sentinel-1 not available
**Solution:** Pipeline continues without S1 data

### Issue: Out of memory
**Solution:** Reduce `max_items` or increase `resolution`

### Issue: Poor model performance (R² < 0.6)
**Possible causes:**
- Too few GEDI points (< 50)
- Mismatched spatial extents
- Poor quality satellite data
- Seasonal mismatch between GEDI and optical data

---

## 📚 Additional Resources

### Main Documentation
- `../PIPELINE.md` - Complete technical guide
- `../API_REFERENCE.md` - Complete API reference

### Visualization Module
See `visualizations.py` for all plotting functions.

### Validation Module
See `validation.py` for statistical analysis functions.

---

## 🤝 Contributing Examples

To add your own example:

1. Create a new Python script in this directory
2. Use the `example_*.py` naming convention
3. Include clear documentation
4. Update this README

---

## 📧 Support

For questions or issues:
1. Check the main documentation
2. Review example outputs
3. Open an issue on GitHub

---

**Happy mapping!** 🌲🗺️
