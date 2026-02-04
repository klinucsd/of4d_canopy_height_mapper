# Canopy Height Mapping Pipeline - Complete Guide

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Key Concepts](#key-concepts)
3. [System Architecture](#system-architecture)
4. [Part 1: GEDI Data Acquisition](#part-1-gedi-data-acquisition)
5. [Part 2: Satellite Data Download](#part-2-satellite-data-download)
6. [Part 3: Model Training](#part-3-model-training)
7. [Part 4: Wall-to-Wall Prediction](#part-4-wall-to-wall-prediction)
8. [End-to-End Workflow](#end-to-end-workflow)
9. [Common Issues & Solutions](#common-issues--solutions)
10. [Interpretation & Validation](#interpretation--validation)

---

## Overview & Motivation

### What Is This Code Trying to Do?

This pipeline creates a **continuous map of forest canopy height** across large geographic areas. Instead of having sparse height measurements (GEDI), it uses machine learning to "fill in the gaps" and predict height everywhere.

### Real-World Problem

Imagine you want to measure how tall trees are in a rainforest:

- **GEDI satellite**: Shoots laser beams down, measures exact tree heights, but only gets ~1 measurement per 25 km²
- **Result**: You have scattered accurate measurements, but 99% of the forest is unmapped

**Solution**: Use optical and radar satellite images (which cover everything) + ML to predict heights at every pixel.

### Pipeline Output

A GeoTIFF file showing:

- Every pixel = predicted canopy height (in meters)
- Example: "This pixel has 15.3m tall trees"
- Spatial resolution: (10m - 60m) × (10m - 60m)
- Can be opened in QGIS, ArcGIS, or Python with rasterio

---

## Key Concepts

### 1. What Is GEDI?

**GEDI = Global Ecosystem Dynamics Investigation**

- **What**: Space-based LiDAR (laser) satellite operated by NASA
- **How it works**:
  - Shoots laser pulses down to forest
  - Laser bounces off top of tree (canopy) and ground
  - Time difference = tree height
- **Accuracy**: ±0.5m (very precise)
- **Coverage**: Sparse (~1 shot per 25 km² in swath mode)
- **Data**: Height measurements + GPS location

**Why use it?**: Ground truth for training the ML model. GEDI tells us real heights; we'll use other satellites to see what those heights "look like" in optical/radar data.

### 2. What Is Sentinel-2?

**Sentinel-2 = Multispectral optical satellite (ESA)**

- **What**: Takes color + infrared photos of Earth
- **Bands**: 11 different "colors" (not just RGB)
  - B04 = Red light
  - B08 = Near-infrared (NIR)
  - B11, B12 = Shortwave infrared
- **Resolution**: 10-60m per pixel (we use 30m by default)
- **Revisit rate**: Every 5 days
- **Data availability**: Free, easy to download

**Why use it?**:

- Dense forests look dark + high NIR (near-infrared)
- Tall trees have different spectral signature than short trees
- High NIR relative to red = vegetated area (NDVI index)

**Key index - NDVI (Normalized Difference Vegetation Index):**

```
NDVI = (NIR - Red) / (NIR + Red)
     = (B08 - B04) / (B08 + B04)
```

**Interpretation:**
- NDVI > 0.7  = Dense vegetation
- NDVI 0.5-0.7 = Moderate vegetation
- NDVI < 0.3 = Sparse or no vegetation

### 3. What Is Sentinel-1?

**Sentinel-1 = Synthetic Aperture Radar (SAR) satellite (ESA)**

- **What**: Shoots radio waves (microwave) at Earth, measures reflection
- **Polarizations**:
  - VV = vertical transmit, vertical receive
  - VH = vertical transmit, horizontal receive
- **Resolution**: 10-20m
- **Advantage**: Works through clouds & at night (unlike optical)
- **Data**: Measured in dB (decibels, log scale)

**Why use it for canopy height?**

- Radio waves penetrate tree canopy differently for tall vs short trees
- Tall trees = stronger backscatter from interior
- Short trees = backscatter mostly from ground

### 4. What Is SRTM DEM?

**SRTM = Shuttle Radar Topography Mission**

- **What**: Global elevation map (digital elevation model)
- **Resolution**: 30m
- **Data**: Two bands:
  - Elevation (meters above sea level)
  - Slope (steepness of terrain)

**Why use it?**

- Trees grow differently on hills vs flat land
- Slope affects water availability → tree height
- Elevation correlates with climate → affects tree growth

### 5. Machine Learning Concept

**Why Random Forest for this task?**

Random Forest is a tree-based ensemble model that:

1. Learns relationships between satellite images and GEDI heights
2. Handles non-linear relationships (e.g., NDVI < 0.4 = short trees, but NDVI 0.7 ≠ necessarily tall)
3. Robust to individual feature importance variations

**Training idea:**

```
Input:  B04 (red), B08 (NIR), B11, B12, VV, VH, elevation, slope
Output: Canopy height (from GEDI)
```

The model learns: "When I see this combination of satellite values, GEDI says height is usually ~18m"

---

## System Architecture

### High-Level Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   GEDI      │      │  Sentinel-2 │      │  Sentinel-1 │      │    SRTM     │
│  (LiDAR)    │      │   (Optical) │      │   (Radar)   │      │   (DEM)     │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │                    │
       │  Height labels     │  Spectral bands    │  Backscatter       │  Elevation
       │  (sparse)          │  (dense)           │  (dense)           │  & slope
       ▼                    ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Feature Extraction & Alignment                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Random Forest Model Training                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Wall-to-Wall Height Prediction                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │  Height Map     │
                      │  (GeoTIFF)      │
                      └─────────────────┘
```

### Data Flow Between Steps

**STEP 1: Download GEDI**
```
├─ Query NASA's earthaccess servers
├─ Find granules (files) covering study area & dates
├─ Download HDF5 files
└─ Extract: lat, lon, rh98 (98th percentile height)
   Output: gedi_raw.csv (2977 points in example)
```

**STEP 2: Download Satellite Data**
```
├─ Sentinel-2: 78 scenes found → use 10 most recent
│  ├─ Download B04, B08, B11, B12
│  ├─ Stack & compute median composite
│  └─ Calculate NDVI
│  Output: sentinel2.tif (5 bands, 30m resolution)
│
├─ Sentinel-1: 5 RTC scenes
│  ├─ Download VV, VH polarizations
│  └─ Stack & compute median
│  Output: sentinel1.tif (2 bands, 10m resolution)
│
└─ SRTM: 8 tiles
   ├─ Download elevation data
   ├─ Merge tiles
   └─ Calculate slope
   Output: topography.tif (2 bands, 30m resolution)
```

**STEP 3: Extract Features**
```
├─ Load all satellite rasters
├─ For each GEDI point:
│  ├─ Reproject point to raster CRS
│  ├─ Sample S2, S1, topo at that location
│  └─ Store values
├─ Remove points with missing data
└─ Output: Feature matrix X (2977 points × 8 features)
           Target vector y (2977 heights from GEDI)
```

**STEP 4: Train Model**
```
├─ Split: 80% training, 20% test
├─ Train RandomForestRegressor
├─ Evaluate: R² = 0.634, RMSE = 3.82m
└─ Output: Trained model (ready to predict)
```

**STEP 5: Predict**
```
├─ Load satellite data for entire study area
├─ For each pixel, extract S2, S1, topo values
├─ Run through trained model
├─ Get predicted height for every pixel
└─ Output: canopy_height_map.tif (371 × 326 pixels)
```

---

## Part 1: GEDI Data Acquisition

### Function: `download_gedi_earthaccess()`

**Goal**: Download GEDI laser measurements that tell us true canopy heights

**What it does:**

```python
def download_gedi_earthaccess(bbox, start_date, end_date):
    # Step 1: Login to NASA EarthData
    earthaccess.login()

    # Step 2: Search for GEDI L2A granules (files)
    # L2A = Level 2A product (geolocated height measurements)
    results = earthaccess.search_data(
        short_name='GEDI02_A',  # GEDI Level 2A data
        version='002',
        bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),
        temporal=(start_date, end_date)
    )
    # Result: List of 8 granules (files)

    # Step 3: Download HDF5 files
    files = earthaccess.download(results, local_path='gedi_downloads')
    # Result: 8 HDF5 files (~100 MB each)

    # Step 4: Extract canopy heights from HDF5
    for file in files:
        with h5py.File(file) as hdf:
            # HDF5 structure:
            # BEAM0000/lat_lowestmode = [array of latitudes]
            # BEAM0000/lon_lowestmode = [array of longitudes]
            # BEAM0000/rh = [array of relative heights]
            #   - rh[:, 0] = 1% percentile (ground)
            #   - rh[:, 98] = 98% percentile (canopy top)
            rh98 = hdf['BEAM0000/rh'][:, 98]  # Get top of canopy
            quality_flag = hdf['BEAM0000/quality_flag'][:]

            # Filter bad data:
            # - quality_flag == 1 means good shot
            # - 0 < rh98 < 100 means reasonable height
            valid_points = (quality_flag == 1) & (rh98 > 0) & (rh98 < 100)

            # Save to CSV
            df = pd.DataFrame({
                'latitude': lat[valid_points],
                'longitude': lon[valid_points],
                'rh98': rh98[valid_points]  # Height in meters
            })
    return gedi_df  # 2977 points
```

**Key insight**: GEDI gives us exact answers (98% percentile height = top of canopy), but only scattered measurements. This is our "ground truth" for training.

### Function: `augment_gedi_with_synthetic()`

**Problem**: 2977 GEDI points are sparse. ML models prefer more training data.

**Solution**: Create synthetic points (with caution!)

```python
def augment_gedi_with_synthetic(gedi_csv, multiplier=3):
    gedi = pd.read_csv(gedi_csv)  # 2977 points

    # For each real point, create 3 synthetic nearby points
    synthetic = []
    for _, row in gedi.iterrows():
        for _ in range(multiplier):
            # Add small random noise around each real point
            synthetic.append({
                'longitude': row['longitude'] + np.random.normal(0, 0.001),
                # 0.001° ≈ 111 meters
                'latitude': row['latitude'] + np.random.normal(0, 0.001),
                'rh98': row['rh98'] * (1 + np.random.normal(0, 0.15))
                # ±15% height variation
            })
    return pd.concat([gedi, pd.DataFrame(synthetic)])
    # Result: ~12,000 points (2977 real + 9,000 synthetic)
```

**Warning**: Synthetic data inflates dataset but doesn't add real diversity. A better approach would be spatial clustering (augment only in sparse regions).

---

## Part 2: Satellite Data Download

### Function: `download_sentinel2_mpc()`

**Goal**: Get optical satellite images to see what the landscape looks like

```python
def download_sentinel2_mpc(bbox, start_date, end_date, resolution=30):
    # Step 1: Query Microsoft Planetary Computer catalog
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    # STAC = SpatialTemporal Asset Catalog (standardized way to find imagery)
    search = catalog.search(
        collections=["sentinel-2-l2a"],  # Level 2A = atmospheric corrected
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}}  # Less than 20% clouds
    )

    items = list(search.items())  # 78 scenes found
    items = sorted(items, key=lambda x: x.datetime, reverse=True)[:10]
    # Use 10 most recent (minimizes vegetation change)

    # Step 2: For each scene, download specific bands
    all_bands = {k: [] for k in ['B04', 'B08', 'B11', 'B12']}
    for item in items:
        # Item = one satellite pass = one .jp2 file per band
        # Sentinel-2 uses UTM projection, not lat/lon
        with rasterio.open(item.assets['B04'].href) as src:
            # B04 = Red band
            # src.crs = EPSG:32617 (UTM zone 17N)
            # src.res = (10, 10) meters = native resolution

            # Transform bbox from lat/lon to UTM
            bbox_utm = transform_bounds('EPSG:4326', src.crs, *bbox)

            # Get pixels covering study area
            window = from_bounds(*bbox_utm, transform=src.transform)

            # Read pixel data (10m resolution)
            data = src.read(1, window=window)  # Shape: (371, 326)
            all_bands['B04'].append(data)

    # Step 3: Create median composite
    # Why median? Removes cloud shadows & atmospheric noise
    medians = {k: np.nanmedian(np.stack(v), axis=0)
               for k, v in all_bands.items()}
    # Result: One image per band, made from 10 scenes

    # Step 4: Calculate NDVI
    ndvi = (medians['B08'] - medians['B04']) / \
           (medians['B08'] + medians['B04'] + 1e-10)

    # NDVI interpretation:
    # Dense forest: 0.6-0.8
    # Grassland: 0.3-0.5
    # Water/rock: <0.2

    # Step 5: Resample to target resolution (30m)
    # Sentinel-2 native resolution varies: B04/B08 = 10m, B11/B12 = 20m
    # We standardize to 30m for easier stacking
    # [resampling code omitted for brevity]

    # Step 6: Save as GeoTIFF
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(medians_resampled['B04'], 1)  # Band 1 = B04 (Red)
        dst.write(medians_resampled['B08'], 2)  # Band 2 = B08 (NIR)
        dst.write(medians_resampled['B11'], 3)  # Band 3 = B11 (SWIR1)
        dst.write(medians_resampled['B12'], 4)  # Band 4 = B12 (SWIR2)
        dst.write(ndvi, 5)                       # Band 5 = NDVI

    return output_path  # 1.8 MB GeoTIFF
```

**Why these bands?**

- **B04 (Red)**: Vegetation absorbs red light
- **B08 (NIR)**: Vegetation reflects NIR strongly
- **B11, B12 (SWIR)**: Detect moisture, distinguish vegetation types
- **NDVI**: Combines B04 & B08 to emphasize vegetation

**Why median composite?**

- Removes clouds & shadows
- Reduces noise
- Creates "average" landscape for period

### Function: `download_sentinel1_mpc()` (Radar Data)

**Goal**: Get radar backscatter to detect tree height through canopy

```python
def download_sentinel1_mpc(bbox, start_date, end_date):
    # Try RTC (Radiometrically Terrain Corrected) first
    # RTC = already corrected for terrain slope & radiometry
    # Better than raw GRD (Ground Range Detected)
    search = catalog.search(
        collections=["sentinel-1-rtc"],  # Terrain-corrected
        bbox=bbox,
        datetime=f"{start_date}/{end_date}"
    )
    items = list(search.items())  # 5 scenes

    # Download VV and VH polarizations
    vv_arrays = []  # Vertical transmit, vertical receive
    vh_arrays = []  # Vertical transmit, horizontal receive
    for item in items:
        with rasterio.open(item.assets['vv'].href) as src:
            # Data is in dB (decibels, log scale)
            # Typical range: -50 to +10 dB
            data = src.read(1, window=window)
            vv_arrays.append(data)

        with rasterio.open(item.assets['vh'].href) as src:
            data = src.read(1, window=window)
            vh_arrays.append(data)

    # Median composite
    vv = np.nanmedian(np.stack(vv_arrays), axis=0)
    vh = np.nanmedian(np.stack(vh_arrays), axis=0)

    # VV/VH ratio indicates canopy structure:
    # Tall forests: higher VV (signal penetrates, scatters from multiple layers)
    # Short vegetation: similar VV/VH

    return vv, vh  # 2 bands saved to GeoTIFF
```

**SAR concept** (Synthetic Aperture Radar):

- Shoots microwave pulses at forest
- Measures backscatter (how much bounces back)
- VV = both transmit & receive vertical → sensitive to volume scattering
- VH = transmit vertical, receive horizontal → sensitive to canopy structure
- VV/VH ratio helps distinguish tree height

### Function: `download_srtm_opentopography()` & `download_srtm_alternative()`

**Goal**: Get elevation & slope data (terrain context)

```python
def download_srtm_opentopography(bbox):
    # Try OpenTopography first (easy but rate-limited)
    response = requests.get(
        "https://portal.opentopography.org/API/globaldem",
        params={'demtype': 'SRTMGL1',  # SRTM Global 1-arcsec
                'west': bbox[0], 'east': bbox[2],
                'south': bbox[1], 'north': bbox[3],
                'outputFormat': 'GTiff'}
    )
    # If rate limited, falls back to NASA EarthData

    # Calculate slope from DEM
    dem = src.read(1)
    dx, dy = np.gradient(dem)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / pixel_size))

    # Result: 2-band GeoTIFF
    # Band 1: Elevation (meters)
    # Band 2: Slope (degrees)

    return dem, slope
```

**Why elevation matters?**

- Tall trees usually on flatter terrain (more stable)
- High elevation = colder → shorter trees
- Slope affects water drainage → tree growth

---

## Part 3: Model Training

### Function: `extract_features()`

**Goal**: Combine GEDI measurements with satellite data

**Challenge**: GEDI points are in lat/lon (EPSG:4326), but satellite data is in UTM (EPSG:32617)

```python
def extract_features(gedi_csv, s2_tif, s1_tif, topo_tif):
    # Load GEDI points
    gedi = pd.read_csv(gedi_csv)
    # Columns: latitude, longitude, rh98 (height in meters)

    # Load satellite rasters
    with rasterio.open(s2_tif) as s2_src:
        s2_crs = s2_src.crs  # EPSG:32617 (UTM)

    # Key problem: GEDI in EPSG:4326, S2 in EPSG:32617
    # Solution: Reproject GEDI points to match each raster
    lons = gedi['longitude'].values
    lats = gedi['latitude'].values

    # For S2 (UTM zone 17)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617")
    utm_x, utm_y = transformer.transform(lats, lons)
    # Now utm_x, utm_y are in meters in UTM coordinate system

    # For each GEDI point, sample satellite values
    s2_values = []

    for x, y in zip(utm_x, utm_y):
        with rasterio.open(s2_tif) as src:
            # src.sample() takes coordinates & returns pixel values
            sample = list(src.sample([(x, y)]))

            # Returns: [B04_value, B08_value, B11_value, B12_value, NDVI_value]
            s2_values.append(sample[0])

    s2_matrix = np.array(s2_values)  # Shape: (2977, 5)

    # Repeat for S1 and Topography
    s1_matrix = np.array([...])  # Shape: (2977, 2) - VV, VH
    topo_matrix = np.array([...])  # Shape: (2977, 2) - elevation, slope

    # Stack all features horizontally
    X = np.hstack([s2_matrix, s1_matrix, topo_matrix])
    # Shape: (2977, 9) - 9 features per point

    # Target variable (what we're predicting)
    y = gedi['rh98'].values  # Shape: (2977,) - actual heights

    # Remove rows with missing data (NaN)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid]  # Final: (2977, 9)
    y = y[valid]  # Final: (2977,)

    return X, y, feature_names
    # X = satellite data at GEDI locations
    # y = GEDI heights
    # feature_names = ['S2_B04', 'S2_B08', 'S2_B11', 'S2_B12', 'S2_NDVI',
    #                  'S1_VV', 'S1_VH', 'Topo_elev', 'Topo_slope']
```

**Key insight**: We now have 2977 pairs of (satellite values, true height). This is our training dataset.

### Function: `train_model()`

**Goal**: Teach ML model to map satellite values → canopy height

```python
def train_model(X, y, feature_names):
    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training set: 2381 points
    # Test set: 596 points

    # Initialize Random Forest
    model = RandomForestRegressor(
        n_estimators=100,      # 100 decision trees
        max_depth=20,          # Each tree ~20 levels deep
        min_samples_split=5,   # Require 5+ samples to split a node
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )

    # Training loop (fits all 100 trees)
    model.fit(X_train, y_train)

    # Each tree learns different patterns:
    # Tree 1: "If NDVI > 0.7 and elevation < 100m, predict 18m"
    # Tree 2: "If VV > -20dB and slope > 5°, predict 22m"
    # ... (100 trees total)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    # y_pred = [15.2, 8.3, 19.1, ...]  (predicted heights)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R²: {r2:.3f}")    # 0.634 = model explains 63.4% of variance
    print(f"RMSE: {rmse:.2f}m")  # 3.82m = average error

    # Feature importance (which satellite bands matter most?)
    importance = model.feature_importances_

    # [0.383, 0.136, 0.119, 0.105, 0.061, ...]
    # (high importance = tree relies on this feature)

    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.3f}")

    # Top features:
    # S2_B04 (Red): 0.383 (most important!)
    # S2_B08 (NIR): 0.136
    # S2_B11 (SWIR): 0.119
    # ...

    return model  # Ready to predict on new data!
```

**What R² = 0.634 means:**

- Model explains 63.4% of height variation
- Remaining 36.6% = model uncertainty + measurement noise
- For production: Target R² > 0.75

**Why Red band (B04) most important?**

- Tall dense forests are darker in red light (absorb more)
- Short grasslands are brighter
- Red band is best single predictor of vegetation density → height correlation

---

## Part 4: Wall-to-Wall Prediction

### Function: `predict_map()`

**Goal**: Apply trained model to every pixel in study area

```python
def predict_map(model, s2_tif, s1_tif, topo_tif, output='canopy_height.tif'):
    # Load satellite data for entire study area
    with rasterio.open(s2_tif) as s2_src:
        s2_data = s2_src.read()
        # Shape: (5 bands, 371 rows, 326 columns)
        # 371×326 = 120,946 pixels
        profile = s2_src.profile  # CRS, transform, etc.

    # Resample S1 to match S2 (in case different resolution)
    with rasterio.open(s1_tif) as s1_src:
        s1_data = np.zeros((2, 371, 326))  # Pre-allocate
        for i in range(2):  # VV and VH
            reproject(
                s1_src.read(i+1),      # Source data
                s1_data[i],            # Destination array
                src_transform=s1_src.transform,
                dst_transform=s2_src.transform,
                resampling=Resampling.bilinear
            )
        # Now s1_data matches S2 dimensions

    # Same for topography
    with rasterio.open(topo_tif) as topo_src:
        topo_data = np.zeros((2, 371, 326))
        # ... reproject ...

    # Stack all data: 9 bands × 371 × 326 pixels
    all_data = np.vstack([s2_data, s1_data, topo_data])

    # Reshape for prediction
    n_bands, height, width = all_data.shape
    data_2d = all_data.reshape(n_bands, -1).T

    # Now: (120946, 9) = each row is one pixel with 9 feature values

    # Apply model to all pixels
    pred = np.full(120946, np.nan)
    for i in range(120946):
        pred[i] = model.predict([data_2d[i]])  # Predict height for pixel i

    # Alternative (faster):
    valid = np.all(np.isfinite(data_2d), axis=1)  # Find valid pixels
    pred[valid] = model.predict(data_2d[valid])   # Batch predict

    # Reshape back to map
    pred_map = pred.reshape(height, width)  # (371, 326)

    # Post-processing: clip to reasonable values
    pred_map = np.clip(pred_map, 0, 100)  # 0-100m (no negative heights!)

    # Save as GeoTIFF
    profile.update(count=1, dtype='float32')
    with rasterio.open(output, 'w', **profile) as dst:
        dst.write(pred_map.astype('float32'), 1)

    return output_path  # 'canopy_height_map.tif'
```

**Output interpretation:**

```
Mean height: 10.2 m
Max height: 20.0 m
```

This means:
- Average tree height in study area: 10.2 meters
- Tallest trees: 20 meters
- Map shows every 30m × 30m pixel with predicted height

---

## End-to-End Workflow

### Complete Pipeline Execution

```python
if __name__ == "__main__":
    # STEP 0: Configure
    bbox = [-60.0, -3.5, -59.8, -3.3]  # Amazon rainforest
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    output_dir = 'output'

    print("STEP 1: GEDI Data")
    # → Downloads 8 granules
    # → Extracts 2977 height measurements
    # → Output: gedi_raw.csv
    gedi_csv = download_gedi_earthaccess(bbox, start_date, end_date)

    print("STEP 2: Satellite Data")
    # → S2: 78 scenes, uses 10 most recent
    # → Creates median composite
    # → Output: sentinel2.tif (1.8 MB, 5 bands)
    s2_path = download_sentinel2_mpc(bbox, start_date, end_date)

    # → S1: 5 RTC scenes
    # → Output: sentinel1.tif (9.4 MB, 2 bands)
    s1_path = download_sentinel1_mpc(bbox, start_date, end_date)

    # → SRTM: 8 tiles merged
    # → Output: topography.tif (199.9 MB, 2 bands)
    topo_path = download_srtm_opentopography(bbox)

    print("STEP 3: Feature Extraction")
    # → Samples all satellite bands at 2977 GEDI locations
    # → Handles CRS reprojection automatically
    # → Output: X (2977, 9), y (2977,)
    X, y, features = extract_features(gedi_csv, s2_path, s1_path, topo_path)

    print("STEP 4: Model Training")
    # → Splits data: 80% train, 20% test
    # → Trains 100 decision trees
    # → Evaluates: R² = 0.634
    # → Output: Trained RandomForestRegressor
    model = train_model(X, y, features)

    print("STEP 5: Prediction")
    # → Applies model to all 120,946 pixels
    # → Each pixel gets predicted height
    # → Output: canopy_height_map.tif (GeoTIFF)
    predict_map(model, s2_path, s1_path, topo_path)

    print("✓ COMPLETE!")
    print("Output: output/canopy_height_map.tif")
```

### Timing & Computational Requirements

```
STEP 1 (GEDI):     5-10 minutes  (mostly download time)

STEP 2 (Satellites): 20-30 minutes
  - S2: 10 min (10 scenes, large files)
  - S1: 5 min
  - SRTM: 10-15 min (8 tiles, 200MB)

STEP 3 (Features):  2-3 minutes  (CRS reprojection)

STEP 4 (Training):  1 minute      (100 trees on 2977 points)

STEP 5 (Prediction): 2-3 minutes  (predict 120k pixels)

TOTAL: ~45-60 minutes
```

**Computational requirements:**

- 8 GB RAM (comfortable)
- 4 GB disk space (temporary files)
- 4-core CPU (minimum; uses all cores)
- Internet: ~10 Mbps

---

## Common Issues & Solutions

### Issue 1: Low Model Accuracy (R² < 0.70)

**Cause**: Study area too small, time period too short

**Solution**:

```python
# Expand study area (0.2° → 2°)
bbox = [-62.0, -5.0, -58.0, -1.0]

# Expand time period (2 years → 5 years)
start_date = '2019-01-01'
end_date = '2024-12-31'

# More satellite scenes
s2_path = download_sentinel2_mpc(bbox, start_date, end_date, max_items=30)

# Better ML model
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=200)
```

**Expected improvement**: R² 0.63 → 0.75+

### Issue 2: Missing CRS Information (Sentinel-1 GRD)

**Cause**: GRD files sometimes lack proper CRS metadata

**Symptom**: `src.crs = None`

**Solution**: Code includes fallback for manual CRS assignment

```python
# Estimate UTM zone from bbox
zone = int((lon_center + 180) / 6) + 1
manual_crs = f"EPSG:{32600 + zone}"  # UTM zone
```

### Issue 3: Shape Mismatch (Different Scenes from Different UTM Zones)

**Cause**: Sentinel-2 scenes may cover multiple UTM zones

**Solution**: Automatic resampling in code

```python
# If scene shapes differ, resample to reference shape
from scipy.ndimage import zoom

zoom_factors = (ref_shape / scene_shape)
resampled_scene = zoom(scene, zoom_factors)
```

### Issue 4: Rate Limiting on OpenTopography

**Cause**: Too many API calls

**Solution**: Code includes fallback to NASA EarthData (earthaccess)

```python
try:
    # OpenTopography (easy but may be rate-limited)
    download_srtm_opentopography()
except:
    # Fallback to NASA EarthData
    download_srtm_alternative()
```

---

## Interpretation & Validation

### Reading the Output Map

The output GeoTIFF contains:

- **Pixel value** = predicted canopy height (meters)
- **NaN/no-data** = missing or invalid pixels
- **0-5m** = grassland, crops, sparse vegetation
- **5-15m** = secondary forest, early succession
- **15-25m** = mature forest, tall trees
- **25-40m** = primary forest (undisturbed)

### Validating Results

To assess accuracy:

1. **Visual inspection**: Does the map match known forest types?
2. **Compare to GEDI**: Reserve some GEDI points, check predictions
3. **Field data**: Measure trees on ground, compare to map
4. **Airborne LiDAR**: If available, compare to reference data

### Understanding Model Confidence

Model uncertainty is NOT provided in current code. Better approaches:

1. **Confidence intervals**: Use quantile regression
2. **Ensemble uncertainty**: Train multiple models, check variance
3. **Spatial filtering**: Smooth predictions to reduce noise

---

## Summary Table

| Component | Function | Input | Output | Time |
|:----------|:---------|:------|:--------|:-----|
| **GEDI** | download_gedi_earthaccess | bbox, dates | CSV: lat,lon,height | 5-10 min |
| **S2 Optical** | download_sentinel2_mpc | bbox, dates | GeoTIFF: 5 bands, 30m | 10 min |
| **S1 Radar** | download_sentinel1_mpc | bbox, dates | GeoTIFF: 2 bands, 10m | 5 min |
| **SRTM DEM** | download_srtm_* | bbox | GeoTIFF: 2 bands, 30m | 10-15 min |
| **Features** | extract_features | All rasters + GEDI | X (2977×9), y (2977) | 2-3 min |
| **Training** | train_model | X, y | RandomForestRegressor | 1 min |
| **Prediction** | predict_map | Model + rasters | GeoTIFF: heights | 2-3 min |

---

## Key Takeaways

1. **Multi-source data fusion**: GEDI (truth) + S2 (optical) + S1 (radar) + DEM (terrain)
2. **CRS handling**: Automatic reprojection between lat/lon and UTM
3. **Temporal compositing**: Median of multiple scenes removes noise
4. **Machine learning**: Random Forest learns non-linear mapping from satellite→height
5. **Wall-to-wall mapping**: Prediction extends from sparse training points to dense map
6. **Trade-offs**: Accuracy vs coverage (larger area = lower R² but more useful)
