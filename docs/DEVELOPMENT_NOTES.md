# Development Notes: GEDI Canopy Height Map V2 Service

**Last Updated:** 2026-02-27
**Status:** Production Ready ✅

---

## Project Overview

Implemented a new FastAPI service `GEDI_canopy_height_map` (V2) that expands upon the existing canopy height service with user-configurable temporal parameters, additional visualizations, and JSON metadata output.

---

## Key Implementation Decisions

### 1. File Naming Convention

**CRITICAL:** All V2 files MUST use the `gedi_canopy_height_` prefix to maintain isolation from V1.

**V2 Files (new, isolated):**
- `gedi_canopy_height_map_service.py` - FastAPI endpoint
- `gedi_canopy_height_map.py` - Worker script
- `gedi_canopy_height_metadata_builder.py` - Metadata builder
- `gedi_canopy_height_pipeline_partitions.py` - Pipeline with new params
- `gedi_canopy_height_visualizations.py` - Visualizations

**V1 Files (unchanged):**
- `canopy_height_service.py`
- `canopy_height.py`
- `canopy_height_pipeline_partitions.py`

### 2. Descriptive Output Filenames

Per manager feedback, use descriptive names:
- `sentinel2_optical_composite.tif` (not `sentinel2.tif`)
- `sentinel1_sar_composite.tif` (not `sentinel1.tif`)

### 3. Tag Convention

Use `"Private"` tag for FastAPI router (not "GEDI Canopy Height" or similar).

---

## Common Issues & Solutions

### Issue 1: Missing Python Modules (seaborn)

**Error:** `ModuleNotFoundError: No module named 'seaborn'`

**Solution:** Make seaborn optional in visualizations:
```python
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
except ImportError:
    plt.style.use('bmh')
```

### Issue 2: UnboundLocalError for train_test_split

**Error:** `cannot access local variable 'train_test_split' where it is not associated with a value`

**Solution:** Import sklearn modules at TOP of file, not inside functions:
```python
# At top with other imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

### Issue 3: Changes Affecting V1 Service

**Problem:** Modifying shared files breaks V1 service.

**Solution:** ALWAYS create new prefixed files. Never modify existing V1 files. Copy and rename instead:
```bash
cp canopy_height_pipeline_partitions.py gedi_canopy_height_pipeline_partitions.py
# Then modify the copy
```

---

## Deployment Checklist

### Files to Deploy

```bash
# To services/
gedi_canopy_height_map_service.py

# To scripts/
gedi_canopy_height_map.py
gedi_canopy_height_metadata_builder.py
gedi_canopy_height_pipeline_partitions.py
gedi_canopy_height_visualizations.py
```

### Deployment Steps

1. Copy files to FastAPI server
2. Set permissions: `sudo chown of4duser:of4duser`
3. Clear Python cache: `sudo find /home/of4duser/apps/fastAPI/scripts -name "*.pyc" -delete`
4. Restart service: `sudo systemctl restart fastapi.service`
5. Verify in Swagger UI (look under "Private" tag)
6. Test with sample JSON

---

## Test JSON (Los Angeles Area)

```json
{
  "min_lon": -118.5,
  "min_lat": 34.0,
  "max_lon": -118.0,
  "max_lat": 34.3,
  "gedi_temporal_min": "2024-01-01",
  "gedi_temporal_max": "2025-12-31",
  "sentinel_temporal_min": "2024-01-01",
  "sentinel_temporal_max": "2025-12-31",
  "sentinel2_l2a_cloud_threshold": 20,
  "DEM_dataset": "COP30",
  "ML_algorithm": "RFR",
  "resolution": 30
}
```

---

## Monitoring Commands

```bash
# Check service logs
sudo journalctl -u fastapi.service -n 100 --no-pager

# Check job stdout
sudo cat /home/of4duser/apps/fastAPI/output/{jobID}/stdout.txt

# Check job stderr
sudo cat /home/of4duser/apps/fastAPI/output/{jobID}/stderr.txt

# List output files
sudo ls -la /home/of4duser/apps/fastAPI/output/{jobID}/

# Check if process running
ps aux | grep gedi_canopy_height_map
```

---

## Expected Output Files

| File | Description |
|------|-------------|
| `gedi.csv` | GEDI training points |
| `sentinel2_optical_composite.tif` | Sentinel-2 composite |
| `sentinel1_sar_composite.tif` | Sentinel-1 SAR |
| `topography.tif` | COP30 DEM |
| `canopy_height_model.pkl` | Trained model |
| `metrics.json` | Performance metrics |
| `canopy_height_map.tif` | Final prediction |
| `canopy_height_map.png` | Visualization |
| `metadata.json` | Structured metadata |
| `gedi_data_coverage.png` | GEDI visualization |
| `gedi_tracks.png` | GEDI tracks |
| `sentinel2_bands.png` | S2 bands |
| `sentinel2_composites.png` | S2 composites |
| `sentinel2_histograms.png` | S2 histograms |
| `model_validation.png` | Model validation |
| `pipeline_summary.png` | Pipeline overview |

---

## File Locations

**Development Machine:**
```
/data/home/klin/misc/test_gee/canopy_height_app/fastapi/
```

**Production Machine:**
```
/home/of4duser/apps/fastAPI/services/     # API endpoints
/home/of4duser/apps/fastAPI/scripts/       # Worker scripts
/home/of4duser/apps/fastAPI/output/{jobID}/ # Job outputs
```

---

## Code Patterns

### Adding New ML Algorithms

Edit `gedi_canopy_height_map.py`:
```python
def get_model(ml_algorithm="RFR", **kwargs):
    algorithms = {
        "RFR": RandomForestRegressor,
        # Add new here:
        # "XGB": XGBRegressor,
    }
    if ml_algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {ml_algorithm}")
    # ... configuration logic
```

### Adding New Parameters

1. Add to request schema in service file
2. Add argparse arguments in worker script
3. Pass to pipeline functions
4. Update metadata builder
5. Update documentation

---

## Performance Baseline

Based on test run (Los Angeles area):

| Metric | Value |
|--------|-------|
| Execution Time | 5-6 minutes |
| GEDI Points | ~48,000 |
| Sentinel-2 Scenes | 10 |
| Sentinel-1 Scenes | 5 |
| Model R² | 0.24-0.85 |
| Model RMSE | 3-6 m |

---

## Contact Points

**Swagger UI:** https://of4d-beta.sdsc.edu/API/docs
**Endpoint:** POST `/v1/GEDI_canopy_height_map`
**Tag:** "Private"

---

## Future Work Ideas

1. Add XGBoost algorithm support
2. Add more DEM datasets (SRTMGL1, NASADEM)
3. Add more visualization types
4. Optimize for larger areas (>50km)
5. Add async status callback mechanism

---

## 2026-02-27 Updates: Lat/Lon Features & Global GEDI Data

### Summary of Changes

**Major improvements to model accuracy and global coverage:**

| Change | Before | After | Improvement |
|--------|--------|-------|-------------|
| Input features | 16 | 18 (+lat/lon) | - |
| Model R² | 0.34 | 0.46 | +35% |
| Model RMSE | 10.88 m | 6.83 m | -37% |
| GEDI coverage | USA only | Global | - |
| Git size | 97GB | 63GB | -34GB |

### 1. Latitude/Longitude Features Added

**Decision:** After empirical testing, added lat/lon as input features for region-specific modeling.

**Test Results:**
- Created comparison test in `lat_lon_feature_comparison/`
- WITHOUT lat/lon: R² = 0.2702, RMSE = 7.04 m
- WITH lat/lon: R² = 0.3542, RMSE = 6.62 m
- **+31.1% R² improvement**

**Implementation:**
- Updated `complete_canopy_height_pipeline.py`:
  - `extract_features()`: Adds latitude/longitude columns to feature matrix
  - `predict_map()`: Creates lat/lon coordinate grids for prediction
  - `load_gedi_from_partitions()`: New helper to load from local partitions
- Updated `fastapi/gedi_canopy_height_pipeline_partitions.py`: Same changes

**Feature importance:** Lat/lon ranked #2-3 in feature importance, confirming their value.

### 2. Global GEDI Data Merge

**Background:** Previously only had USA GEDI data. Manager requested global coverage.

**Process:**
1. Downloaded 7 regional GEDI datasets (Africa, Asia, Europe, NA, Oceania, SA, USA)
2. Merged into single global dataset: `gedi_global_2024_2025/`
3. Updated `.env`: `GEDI_DATA_DIR=gedi_global_2024_2025`

**Merge Results:**
| Metric | Value |
|--------|-------|
| Total partitions | 22,673 |
| GEDI points | 3.37 billion |
| Total size | 62.83 GB |
| Source regions | 6 (USA subset merged into NA) |
| Border merges | 637 partitions |

**Script:** `merge_gedi_regions.py` - merges regional GEDI parquet partitions

### 3. API Endpoint Versioning

**Changed:** `/GEDI_canopy_height_map` → `/v1/GEDI_canopy_height_map`

**Rationale:**
- First versioned endpoint for production release
- Allows future `/v2/` without breaking changes
- Matches standard API versioning practices

**Files updated:**
- `fastapi/gedi_canopy_height_map_service.py`: Endpoint path updated
- Documentation updated to reflect V1 status

### 4. Model Performance Improvements

**Train/Test Split Comparison:**
- Manager requested 70/30 vs 80/20 comparison
- Created test in `train_test_split_comparison/`
- Result: 80/20 performs better (R² 0.2702 vs 0.2642)
- Kept 80/20 split to maximize training data with sparse GEDI

**Feature Set Expansions:**
- Sentinel-2: 4 bands → 10 bands (full multispectral)
- DEM: 2 bands (elevation, slope) → 3 bands (+aspect)

### 5. Git Repository Cleanup

**Issue:** `.git/objects` grew to 34GB due to reflog accumulation

**Solution:**
```bash
git gc --aggressive --prune=now
git reflog expire --expire=now --all
```

**Result:** Repository reduced from 97GB → 63GB (34GB reclaimed)

**Prevention:** Consider setting `git config gc.pruneExpire now`

### 6. FastAPI Deployment

**Files deployed to production:**
```
fastapi/gedi_canopy_height_map_service.py    # /v1/ endpoint
fastapi/gedi_canopy_height_pipeline_partitions.py  # 18 features
```

**Production server:** of4d-beta.sdsc.edu

**Test Results (Europe region):**
- Execution time: 2m 6s
- R²: 0.4559
- RMSE: 6.83 m
- Features: 18 (confirmed)

### 7. Test Scripts Created

| Folder | Purpose | Result |
|--------|---------|--------|
| `train_test_split_comparison/` | Compare 70/30 vs 80/20 | 80/20 better |
| `lat_lon_feature_comparison/` | Test lat/lon features | +31% R² |
| `europe_latlon_test/` | Quick Europe region test | R² 0.416 |

### 8. Reports Created

| File | Purpose |
|------|---------|
| `docs/reports/GEDI_DOWNLOAD_REPORT.md` | Global GEDI download & merge documentation |
| `FEATURE_EXPANSION_2025-02-25.md` | 10 bands + aspect changes |

---

## Updated Endpoint Configuration

**New endpoint:** `POST /v1/GEDI_canopy_height_map`

**Request schema:** Same as before, but now uses:
- 18 features (including lat/lon)
- Global GEDI data (`gedi_global_2024_2025/`)
- 10 Sentinel-2 bands
- 3 DEM bands (elevation, slope, aspect)

**Swagger UI:** https://of4d-beta.sdsc.edu/API/docs

---

## Updated Performance Baseline

Based on production test (Europe region, small area):

| Metric | Value |
|--------|-------|
| Execution Time | 2-6 minutes (size dependent) |
| Input Features | 18 (16 + lat/lon) |
| Model R² | 0.40-0.46 |
| Model RMSE | 6-7 m |
| Sentinel-2 Bands | 10 |
| DEM Bands | 3 |

---

## Global GEDI Data Structure

```
gedi_global_2024_2025/
├── lat_XX_lon_YY/
│   └── part.parquet          # GEDI points for this 1°×1° tile
├── PARTITION_SUMMARY.csv     # Partition metadata
└── .checkpoint_from_regions.json
```

**Columns:** `latitude`, `longitude`, `rh98` (relative height at 98th percentile)

**Time period:** 2024-2025 (most recent GEDI L2A available)

**Coverage:** Global (all continents)
