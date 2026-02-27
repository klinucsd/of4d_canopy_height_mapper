"""
Complete Canopy Height Mapping Pipeline
- Downloads Sentinel-2, Sentinel-1, SRTM from alternative sources (no GEE limits)
- Enhanced GEDI data acquisition with multiple strategies
- Trains Random Forest model
- Generates wall-to-wall canopy height map

Requirements:
pip install rasterio numpy pandas scikit-learn requests pystac-client planetary-computer earthaccess h5py pyproj scipy python-dotenv
"""

import os
import numpy as np

# Load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pystac_client
import planetary_computer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# ============================================================================
# PART 1: ENHANCED GEDI DATA DOWNLOAD
# ============================================================================

def _process_gedi_hdf5(filepath, bbox):
    """
    Worker function to extract data from a single GEDI HDF5 file.
    Designed to be called in parallel by ProcessPoolExecutor.

    Returns:
        list: List of DataFrames containing extracted GEDI data
    """
    import h5py
    all_data = []

    try:
        with h5py.File(filepath, 'r') as hdf:
            for beam in hdf.keys():
                if beam.startswith('BEAM'):
                    try:
                        lat = hdf[f'{beam}/lat_lowestmode'][:]
                        lon = hdf[f'{beam}/lon_lowestmode'][:]
                        rh = hdf[f'{beam}/rh'][:]
                        rh98 = rh[:, 98]  # 98th percentile
                        quality = hdf[f'{beam}/quality_flag'][:]

                        # Filter
                        mask = (
                            (lon >= bbox[0]) & (lon <= bbox[2]) &
                            (lat >= bbox[1]) & (lat <= bbox[3]) &
                            (quality == 1) &
                            (rh98 > 0) & (rh98 < 100)
                        )

                        if mask.sum() > 0:
                            df = pd.DataFrame({
                                'latitude': lat[mask],
                                'longitude': lon[mask],
                                'rh98': rh98[mask]
                            })
                            all_data.append(df)
                    except:
                        continue
    except Exception as e:
        pass  # Silently skip files with errors

    return all_data


def download_gedi_earthaccess(bbox, start_date, end_date, output_csv='gedi_earthaccess.csv', n_workers=None):
    """
    Download GEDI using earthaccess library - EASIEST METHOD!
    This gets 10-100x MORE data than basic AppEEARS
    
    First time: pip install earthaccess
    First run: Will prompt for NASA EarthData login (free account)
    """
    try:
        import earthaccess
        import h5py
    except ImportError:
        print("⚠ Missing libraries. Install with:")
        print("  pip install earthaccess h5py")
        return None
    
    print("="*60)
    print("Downloading GEDI L2A using earthaccess")
    print("="*60 + "\n")
    
    # Login - read credentials from environment variables or .env file
    username = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')

    if not username or not password:
        print("  ⚠ EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables not set.")
        print("  Please set them in your environment or create a .env file:")
        print("    EARTHDATA_USERNAME=your_username")
        print("    EARTHDATA_PASSWORD=your_password")
        return None

    os.environ['EARTHDATA_USERNAME'] = username
    os.environ['EARTHDATA_PASSWORD'] = password
    earthaccess.login()
    
    # Search for GEDI L2A granules
    print("Searching for GEDI granules...")
    results = earthaccess.search_data(
        short_name='GEDI02_A',
        version='002',
        bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),
        temporal=(start_date, end_date)
    )
    
    print(f"  Found {len(results)} granules")
    
    if len(results) == 0:
        print("  ⚠ No GEDI data in this bbox/timeframe")
        return None
    
    # Download
    print(f"\nDownloading {len(results)} GEDI granules...")
    os.makedirs('gedi_downloads', exist_ok=True)
    
    files = earthaccess.download(results, local_path='gedi_downloads')
    print(f"  ✓ Downloaded {len(files)} HDF5 files")

    # Set number of workers (default to CPU count, but cap at 4)
    if n_workers is None:
        n_workers = min(4, multiprocessing.cpu_count())

    # Extract rh98 from HDF5 in parallel
    print(f"\nExtracting canopy heights (using {n_workers} workers)...")

    all_data = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_gedi_hdf5, f, bbox) for f in files]

        for i, future in enumerate(as_completed(futures)):
            try:
                data_list = future.result()
                if data_list:
                    all_data.extend(data_list)
                print(f"  Processed {i+1}/{len(files)} files")
            except Exception as e:
                print(f"  ✗ Error processing file: {str(e)[:50]}")

    if len(all_data) == 0:
        print("  ⚠ No valid points extracted")
        return None
    
    # Combine
    gedi_df = pd.concat(all_data, ignore_index=True)
    gedi_df.to_csv(output_csv, index=False)
    
    print(f"✓ Extracted {len(gedi_df)} GEDI points")
    print(f"  Saved to: {output_csv}")
    
    return output_csv


def augment_gedi_with_synthetic(gedi_csv, multiplier=3):
    """
    Augment sparse GEDI with synthetic points
    Helps when coverage is low!
    """
    print("\n" + "="*60)
    print("Augmenting with Synthetic Points")
    print("="*60 + "\n")
    
    gedi = pd.read_csv(gedi_csv)
    original = len(gedi)
    print(f"Original: {original} points")
    
    if original < 5:
        print("  ⚠ Too few to augment")
        return gedi_csv
    
    # Generate synthetic points near each real point
    synthetic = []
    for _, row in gedi.iterrows():
        for _ in range(multiplier):
            synthetic.append({
                'longitude': row['longitude'] + np.random.normal(0, 0.001),
                'latitude': row['latitude'] + np.random.normal(0, 0.001),
                'rh98': row['rh98'] * (1 + np.random.normal(0, 0.15)),
                'synthetic': True
            })
    
    gedi['synthetic'] = False
    augmented = pd.concat([gedi, pd.DataFrame(synthetic)], ignore_index=True)
    
    output = gedi_csv.replace('.csv', '_augmented.csv')
    augmented.to_csv(output, index=False)
    
    print(f"✓ Augmented: {len(augmented)} points total")
    print(f"  Original: {original}, Synthetic: {len(synthetic)}")
    print(f"  Saved to: {output}")
    
    return output


# ============================================================================
# PART 2: SATELLITE DATA DOWNLOAD
# ============================================================================

def _process_sentinel2_scene(item, bbox, bands, resolution, scene_idx):
    """
    Worker function to process a single Sentinel-2 scene.
    Designed to be called in parallel by ProcessPoolExecutor.

    Returns:
        dict: {'scene_idx': int, 'scene_data': dict or None, 'profile': dict or None, 'crs': CRS or None}
    """
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from scipy.ndimage import zoom
    from rasterio.transform import from_bounds as transform_from_bounds

    scene_data = {}
    scene_valid = True
    profile = None
    crs = None

    for b_name, b_asset in bands.items():
        if b_asset not in item.assets:
            return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': f'Missing {b_asset}'}

        try:
            href = item.assets[b_asset].href
            with rasterio.open(href) as src:
                if crs is None:
                    crs = src.crs

                # Transform bbox to source CRS
                if src.crs.to_string() != 'EPSG:4326':
                    bbox_transformed = transform_bounds('EPSG:4326', src.crs, *bbox)
                else:
                    bbox_transformed = bbox

                # Get window
                window = from_bounds(*bbox_transformed, transform=src.transform)
                window = window.round_offsets().round_lengths()

                if window.width <= 0 or window.height <= 0:
                    return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': 'Invalid window'}

                # Read data at native resolution
                data = src.read(1, window=window)

                # Calculate target profile if first band
                if profile is None:
                    native_res = src.res[0]
                    scale_factor = native_res / resolution

                    target_height = int(data.shape[0] * scale_factor)
                    target_width = int(data.shape[1] * scale_factor)

                    target_transform = transform_from_bounds(
                        *bbox_transformed,
                        target_width,
                        target_height
                    )

                    profile = {
                        'driver': 'GTiff',
                        'height': target_height,
                        'width': target_width,
                        'count': 5,
                        'dtype': 'float32',
                        'crs': src.crs,
                        'transform': target_transform,
                        'compress': 'lzw'
                    }

                # Resample to target resolution if needed
                target_shape = (profile['height'], profile['width'])

                if data.shape != target_shape:
                    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
                    data = zoom(data, zoom_factors, order=1)

                scene_data[b_name] = data

        except Exception as e:
            return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': str(e)}

    # Verify all bands were processed
    if scene_valid and len(scene_data) == len(bands):
        return {'scene_idx': scene_idx, 'scene_data': scene_data, 'profile': profile, 'crs': crs, 'error': None}
    else:
        return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': 'Incomplete bands'}


def download_sentinel2_mpc(bbox, start_date, end_date, output_path='sentinel2.tif', max_items=10, resolution=30, n_workers=None):
    """
    Download Sentinel-2 from Microsoft Planetary Computer - NO LIMITS!
    
    Parameters:
    -----------
    resolution : int
        Target resolution in meters (10, 20, 30, or 60)
        Lower = better quality but larger files
        30m is good balance for canopy height mapping
    """
    print("\n" + "="*60)
    print("Downloading Sentinel-2 from MS Planetary Computer")
    print("="*60 + "\n")
    
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling as RasterioResampling
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}}
    )
    
    items = list(search.items())
    print(f"  Found {len(items)} scenes")
    
    if len(items) == 0:
        raise Exception("No S2 images! Try longer date range.")
    
    items = sorted(items, key=lambda x: x.datetime, reverse=True)[:max_items]
    print(f"  Using {len(items)} most recent")
    print(f"  Bbox: {bbox}")
    print(f"  Target resolution: {resolution}m")
    
    # Band mapping - expanded to include Blue, Green, and Red Edge bands
    bands = {
        'B02': 'B02',  # Blue (10m)
        'B03': 'B03',  # Green (10m)
        'B04': 'B04',  # Red (10m)
        'B05': 'B05',  # Red Edge 1 (20m)
        'B06': 'B06',  # Red Edge 2 (20m)
        'B07': 'B07',  # Red Edge 3 (20m)
        'B08': 'B08',  # NIR (10m)
        'B8A': 'B8A',  # Red Edge 4 (20m)
        'B11': 'B11',  # SWIR1 (20m)
        'B12': 'B12'   # SWIR2 (20m)
    }
    all_bands = {k: [] for k in bands.keys()}
    ref_profile = None
    ref_crs = None

    # Set number of workers (default to CPU count, but cap at 4 to avoid overwhelming the API)
    if n_workers is None:
        n_workers = min(4, multiprocessing.cpu_count())

    print(f"  Using {n_workers} parallel workers for scene processing...\n")

    # Process scenes in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_sentinel2_scene, item, bbox, bands, resolution, i): item
            for i, item in enumerate(items)
        }

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result['error'] is None:
                    print(f"  ✓ {item.id[:40]}... (scene {result['scene_idx']})")
                else:
                    print(f"  ✗ {item.id[:40]}... {result['error'][:50]}")
            except Exception as e:
                print(f"  ✗ {item.id[:40]}... Exception: {str(e)[:50]}")

    # Sort results by scene_idx to maintain order
    results.sort(key=lambda x: x['scene_idx'])

    # Collect valid scenes and determine reference profile
    for result in results:
        if result['scene_data'] is not None:
            if ref_profile is None:
                ref_profile = result['profile']
                ref_crs = result['crs']
                print(f"\n  Reference CRS: {ref_crs}")
                print(f"  Target shape: {ref_profile['height']}x{ref_profile['width']} at {resolution}m")

            for b_name, data in result['scene_data'].items():
                all_bands[b_name].append(data)

    # Resample scenes to reference shape if needed (handles UTM zone mismatches)
    ref_shape = (ref_profile['height'], ref_profile['width'])
    print(f"\nChecking scene shapes (target: {ref_shape})...")

    from scipy.ndimage import zoom
    resampled_count = 0

    for band_name in all_bands.keys():
        if len(all_bands[band_name]) == 0:
            continue

        resampled_arrays = []
        for i, arr in enumerate(all_bands[band_name]):
            if arr.shape != ref_shape:
                zoom_factors = (ref_shape[0] / arr.shape[0], ref_shape[1] / arr.shape[1])
                arr_resampled = zoom(arr, zoom_factors, order=1)
                resampled_arrays.append(arr_resampled)
                resampled_count += 1
            else:
                resampled_arrays.append(arr)

        all_bands[band_name] = resampled_arrays

    if resampled_count > 0:
        print(f"  Resampled {resampled_count} arrays to match reference shape")

    # Check if we got any data
    valid_bands = {k: v for k, v in all_bands.items() if len(v) > 0}
    print(f"\n  Bands with data: {list(valid_bands.keys())}")
    print(f"  Scenes per band: {[len(v) for v in valid_bands.values()]}")
    
    if not valid_bands or not all(len(v) > 0 for v in all_bands.values()):
        raise Exception(f"Failed to download all bands. Got: {list(valid_bands.keys())}")
    
    if ref_profile is None:
        raise Exception("No valid scenes were read - ref_profile is None")
    
    # Median composite
    print("\nComputing median...")
    medians = {k: np.nanmedian(np.stack(v), axis=0) for k, v in all_bands.items()}
    
    # NDVI
    ndvi = (medians['B08'] - medians['B04']) / (medians['B08'] + medians['B04'] + 1e-10)
    medians['NDVI'] = ndvi
    
    # Save
    print(f"Saving to {output_path}...")

    # Update profile count to match actual number of bands (including NDVI)
    ref_profile = ref_profile.copy()
    ref_profile['count'] = len(medians)
    print(f"  Profile: {ref_profile['width']}x{ref_profile['height']}, {ref_profile['count']} bands, {resolution}m resolution")

    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        for i, (name, data) in enumerate(medians.items(), 1):
            dst.write(data.astype('float32'), i)
            dst.set_band_description(i, name)
    
    file_size_mb = os.path.getsize(output_path)/(1024**2)
    print(f"✓ Saved: {file_size_mb:.1f} MB")
    
    # Estimate what 10m vs 30m would be
    if resolution == 30:
        est_10m = file_size_mb * (30/10)**2
        print(f"  (10m resolution would be ~{est_10m:.0f} MB)")
    
    return output_path
    """Download Sentinel-2 from Microsoft Planetary Computer - NO LIMITS!"""
    print("\n" + "="*60)
    print("Downloading Sentinel-2 from MS Planetary Computer")
    print("="*60 + "\n")
    
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}}
    )
    
    items = list(search.items())
    print(f"  Found {len(items)} scenes")
    
    if len(items) == 0:
        raise Exception("No S2 images! Try longer date range.")
    
    items = sorted(items, key=lambda x: x.datetime, reverse=True)[:max_items]
    print(f"  Using {len(items)} most recent")
    print(f"  Bbox: {bbox}")
    
    # Band mapping - expanded to include Blue, Green, and Red Edge bands
    bands = {
        'B02': 'B02',  # Blue (10m)
        'B03': 'B03',  # Green (10m)
        'B04': 'B04',  # Red (10m)
        'B05': 'B05',  # Red Edge 1 (20m)
        'B06': 'B06',  # Red Edge 2 (20m)
        'B07': 'B07',  # Red Edge 3 (20m)
        'B08': 'B08',  # NIR (10m)
        'B8A': 'B8A',  # Red Edge 4 (20m)
        'B11': 'B11',  # SWIR1 (20m)
        'B12': 'B12'   # SWIR2 (20m)
    }
    all_bands = {k: [] for k in bands.keys()}
    ref_profile = None
    
    for item in items:
        print(f"  Processing: {item.id[:40]}...")
        scene_data = {}
        scene_valid = True
        
        for b_name, b_asset in bands.items():
            if b_asset not in item.assets:
                print(f"    ⚠ Missing {b_asset}")
                scene_valid = False
                break
                
            try:
                href = item.assets[b_asset].href
                with rasterio.open(href) as src:
                    # Debug: print source CRS and bounds
                    if b_name == 'B04':
                        print(f"    Source CRS: {src.crs}, Bounds: {src.bounds}")
                    
                    # Transform bbox to source CRS if needed
                    if src.crs.to_string() != 'EPSG:4326':
                        bbox_transformed = transform_bounds('EPSG:4326', src.crs, *bbox)
                    else:
                        bbox_transformed = bbox
                    
                    # Get window
                    window = from_bounds(*bbox_transformed, transform=src.transform)
                    
                    # Validate and round
                    window = window.round_offsets().round_lengths()
                    
                    if window.width <= 0 or window.height <= 0:
                        print(f"    ⚠ Invalid window: {window.width}x{window.height}")
                        scene_valid = False
                        break
                    
                    # Read data
                    data = src.read(1, window=window)
                    
                    if b_name == 'B04':
                        print(f"    Window: {window}, Data shape: {data.shape}")
                    
                    scene_data[b_name] = data
                    
                    if ref_profile is None:
                        ref_profile = {
                            'driver': 'GTiff',
                            'height': data.shape[0],
                            'width': data.shape[1],
                            'count': 5,
                            'dtype': 'float32',
                            'crs': src.crs,
                            'transform': src.window_transform(window),
                            'compress': 'lzw'
                        }
            except Exception as e:
                print(f"    ⚠ Error reading {b_asset}: {str(e)[:100]}")
                scene_valid = False
                break
        
        # Add scene if all bands valid
        if scene_valid and len(scene_data) == len(bands):
            for b_name, data in scene_data.items():
                all_bands[b_name].append(data)
            print(f"    ✓ Scene added")
        else:
            print(f"    ✗ Scene skipped")
    
    # Check if we got any data
    valid_bands = {k: v for k, v in all_bands.items() if len(v) > 0}
    print(f"\n  Bands with data: {list(valid_bands.keys())}")
    print(f"  Scenes per band: {[len(v) for v in valid_bands.values()]}")
    
    if not valid_bands or not all(len(v) > 0 for v in all_bands.values()):
        raise Exception(f"Failed to download all bands. Got: {list(valid_bands.keys())}")
    
    if ref_profile is None:
        raise Exception("No valid scenes were read - ref_profile is None")
    
    # Check for shape consistency (scenes from different UTM zones may differ)
    print("\nChecking scene shapes...")
    for band_name, arrays in all_bands.items():
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            print(f"  ⚠ {band_name} has inconsistent shapes: {set(shapes)}")
            print(f"    Resampling all to reference shape: {ref_profile['height']}x{ref_profile['width']}")
            
            # Resample all arrays to reference shape
            ref_shape = (ref_profile['height'], ref_profile['width'])
            resampled = []
            
            for arr in arrays:
                if arr.shape != ref_shape:
                    # Simple resize using zoom
                    from scipy.ndimage import zoom
                    zoom_factors = (ref_shape[0] / arr.shape[0], ref_shape[1] / arr.shape[1])
                    arr_resampled = zoom(arr, zoom_factors, order=1)
                    resampled.append(arr_resampled)
                else:
                    resampled.append(arr)
            
            all_bands[band_name] = resampled
        else:
            print(f"  ✓ {band_name}: all shapes consistent {shapes[0]}")
    
    # Median composite
    print("\nComputing median...")
    medians = {k: np.nanmedian(np.stack(v), axis=0) for k, v in all_bands.items()}
    
    # NDVI
    ndvi = (medians['B08'] - medians['B04']) / (medians['B08'] + medians['B04'] + 1e-10)
    medians['NDVI'] = ndvi
    
    # Save
    print(f"Saving to {output_path}...")

    # Update profile count to match actual number of bands (including NDVI)
    ref_profile = ref_profile.copy()
    ref_profile['count'] = len(medians)
    print(f"  Profile: {ref_profile['width']}x{ref_profile['height']}, {ref_profile['count']} bands")

    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        for i, (name, data) in enumerate(medians.items(), 1):
            dst.write(data.astype('float32'), i)
            dst.set_band_description(i, name)
    
    print(f"✓ Saved: {os.path.getsize(output_path)/(1024**2):.1f} MB")
    return output_path


def download_sentinel1_mpc(bbox, start_date, end_date, output_path='sentinel1.tif', max_items=5):
    """
    Download Sentinel-1 SAR
    Using RTC (Radiometrically Terrain Corrected) instead of GRD
    RTC has proper CRS metadata and is better for terrain analysis
    """
    print("\n" + "="*60)
    print("Downloading Sentinel-1 from MS Planetary Computer")
    print("="*60 + "\n")
    
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Try RTC collection first (has proper metadata)
    print("  Trying Sentinel-1 RTC (Radiometrically Terrain Corrected)...")
    try:
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
        )
        
        items = list(search.items())[:max_items]
        
        if len(items) > 0:
            print(f"  Found {len(items)} RTC scenes")
            return download_s1_rtc_scenes(items, bbox, output_path)
    except Exception as e:
        print(f"  ⚠ RTC search failed: {str(e)[:80]}")
    
    # Fallback: Try GRD with manual CRS assignment
    print("\n  Trying Sentinel-1 GRD (with manual CRS fix)...")
    try:
        search = catalog.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
        )
        
        items = list(search.items())[:max_items]
        print(f"  Found {len(items)} GRD scenes")
        
        if len(items) == 0:
            print("  ⚠ No S1 data, skipping")
            return None
        
        return download_s1_grd_scenes_fixed(items, bbox, output_path)
        
    except Exception as e:
        print(f"  ⚠ GRD download failed: {str(e)[:80]}")
        print("  ⚠ Continuing without Sentinel-1...")
        return None


def download_s1_rtc_scenes(items, bbox, output_path):
    """Download Sentinel-1 RTC scenes with shape mismatch handling"""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from scipy.ndimage import zoom
    
    vv_arrays, vh_arrays = [], []
    ref_profile = None
    
    for item in items:
        print(f"  {item.id[:40]}...")
        scene_vv = None
        scene_vh = None
        
        # RTC uses different asset names
        for pol_asset in [('vv', 'vh'), ('VV', 'VH')]:
            pol_vv, pol_vh = pol_asset
            
            if pol_vv in item.assets and pol_vh in item.assets:
                try:
                    with rasterio.open(item.assets[pol_vv].href) as src:
                        if src.crs.to_string() != 'EPSG:4326':
                            bbox_transformed = transform_bounds('EPSG:4326', src.crs, *bbox)
                        else:
                            bbox_transformed = bbox
                        
                        window = from_bounds(*bbox_transformed, transform=src.transform)
                        window = window.round_offsets().round_lengths()
                        
                        if window.width <= 0 or window.height <= 0:
                            continue
                        
                        scene_vv = src.read(1, window=window)
                        
                        if ref_profile is None:
                            ref_profile = {
                                'driver': 'GTiff',
                                'height': scene_vv.shape[0],
                                'width': scene_vv.shape[1],
                                'count': 2,
                                'dtype': 'float32',
                                'crs': src.crs,
                                'transform': src.window_transform(window),
                                'compress': 'lzw'
                            }
                    
                    with rasterio.open(item.assets[pol_vh].href) as src:
                        if src.crs.to_string() != 'EPSG:4326':
                            bbox_transformed = transform_bounds('EPSG:4326', src.crs, *bbox)
                        else:
                            bbox_transformed = bbox
                        
                        window = from_bounds(*bbox_transformed, transform=src.transform)
                        window = window.round_offsets().round_lengths()
                        scene_vh = src.read(1, window=window)
                    
                    # Resample to reference shape if needed
                    if scene_vv is not None and scene_vh is not None:
                        ref_shape = (ref_profile['height'], ref_profile['width'])
                        
                        if scene_vv.shape != ref_shape:
                            print(f"    Resampling VV: {scene_vv.shape} → {ref_shape}")
                            zoom_factors = (ref_shape[0] / scene_vv.shape[0], ref_shape[1] / scene_vv.shape[1])
                            scene_vv = zoom(scene_vv, zoom_factors, order=1)
                        
                        if scene_vh.shape != ref_shape:
                            print(f"    Resampling VH: {scene_vh.shape} → {ref_shape}")
                            zoom_factors = (ref_shape[0] / scene_vh.shape[0], ref_shape[1] / scene_vh.shape[1])
                            scene_vh = zoom(scene_vh, zoom_factors, order=1)
                        
                        vv_arrays.append(scene_vv)
                        vh_arrays.append(scene_vh)
                        print(f"    ✓ Scene added")
                    break
                    
                except Exception as e:
                    print(f"    ⚠ Error: {str(e)[:50]}")
                    continue
    
    if len(vv_arrays) == 0:
        print("  ⚠ No valid RTC data")
        return None
    
    vv = np.clip(np.nanmedian(np.stack(vv_arrays), axis=0), -50, 10)
    vh = np.clip(np.nanmedian(np.stack(vh_arrays), axis=0), -50, 10)
    
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        dst.write(vv.astype('float32'), 1)
        dst.write(vh.astype('float32'), 2)
        dst.set_band_description(1, 'VV')
        dst.set_band_description(2, 'VH')
    
    print(f"✓ Saved: {os.path.getsize(output_path)/(1024**2):.1f} MB")
    return output_path


def download_s1_grd_scenes_fixed(items, bbox, output_path):
    """Download GRD with manual CRS assignment (fallback)"""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from pyproj import CRS
    
    vv_arrays, vh_arrays = [], []
    ref_profile = None
    
    for item in items:
        print(f"  {item.id[:40]}...")
        
        # Get projection from item properties
        proj_string = item.properties.get('proj:epsg', None)
        if proj_string:
            manual_crs = f"EPSG:{proj_string}"
        else:
            # Default to UTM zone based on bbox center
            lon_center = (bbox[0] + bbox[2]) / 2
            zone = int((lon_center + 180) / 6) + 1
            hemisphere = 'north' if (bbox[1] + bbox[3]) / 2 >= 0 else 'south'
            manual_crs = f"EPSG:{32600 + zone if hemisphere == 'north' else 32700 + zone}"
        
        scene_vv = None
        scene_vh = None
        
        for pol in ['vv', 'vh']:
            if pol in item.assets:
                try:
                    with rasterio.open(item.assets[pol].href) as src:
                        # Manually assign CRS if missing
                        if src.crs is None:
                            src_crs = CRS.from_string(manual_crs)
                            print(f"    Assigning CRS: {manual_crs}")
                        else:
                            src_crs = src.crs
                        
                        # Transform bbox
                        bbox_transformed = transform_bounds('EPSG:4326', src_crs, *bbox)
                        window = from_bounds(*bbox_transformed, transform=src.transform)
                        window = window.round_offsets().round_lengths()
                        
                        if window.width <= 0 or window.height <= 0:
                            continue
                        
                        data = src.read(1, window=window)
                        
                        if pol == 'vv':
                            scene_vv = data
                        else:
                            scene_vh = data
                        
                        if ref_profile is None:
                            ref_profile = {
                                'driver': 'GTiff',
                                'height': data.shape[0],
                                'width': data.shape[1],
                                'count': 2,
                                'dtype': 'float32',
                                'crs': src_crs,
                                'transform': src.window_transform(window),
                                'compress': 'lzw'
                            }
                except Exception as e:
                    print(f"    ⚠ Error with {pol}: {str(e)[:50]}")
                    continue
        
        if scene_vv is not None and scene_vh is not None:
            vv_arrays.append(scene_vv)
            vh_arrays.append(scene_vh)
            print(f"    ✓ Scene added")
    
    if len(vv_arrays) == 0:
        print("  ⚠ No valid GRD data")
        return None
    
    vv = np.clip(np.nanmedian(np.stack(vv_arrays), axis=0), -50, 10)
    vh = np.clip(np.nanmedian(np.stack(vh_arrays), axis=0), -50, 10)
    
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        dst.write(vv.astype('float32'), 1)
        dst.write(vh.astype('float32'), 2)
        dst.set_band_description(1, 'VV')
        dst.set_band_description(2, 'VH')
    
    print(f"✓ Saved: {os.path.getsize(output_path)/(1024**2):.1f} MB")
    return output_path


def download_srtm_opentopography(bbox, output_path='topography.tif', demtype='COP30'):
    """
    Download DEM from OpenTopography

    Parameters:
    -----------
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    output_path : str
        Output path for the topography TIFF file
    demtype : str
        DEM dataset type. Options:
        - 'COP30' (default): Copernicus 30m Global DEM (more recent, recommended)
        - 'SRTMGL1': SRTM 1 Arc-Second Global (30m)
        - 'SRTMGL3': SRTM 3 Arc-Second Global (90m)
        - 'AW3D30': ALOS World 3D 30m

    Uses API key from OPENTOPOGRAPHY_API_KEY environment variable if available.
    """
    print("\n" + "="*60)
    print(f"Downloading {demtype} DEM from OpenTopography")
    print("="*60 + "\n")

    base_url = "https://portal.opentopography.org/API/globaldem"

    params = {
        'demtype': demtype,
        'south': bbox[1], 'north': bbox[3],
        'west': bbox[0], 'east': bbox[2],
        'outputFormat': 'GTiff',
    }

    # Add API key if available in environment
    api_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
    if api_key:
        params['API_Key'] = api_key
        print("  Using API key from environment")
    else:
        print("  No API key found - may hit rate limits")

    print("Requesting DEM...")
    try:
        response = requests.get(base_url, params=params, stream=True, timeout=300)
        
        if response.status_code == 401:
            print("  ⚠ OpenTopography requires API key or has rate limits")
            print("  Trying alternative method...")
            return download_srtm_alternative(bbox, output_path)
        
        if response.status_code != 200:
            print(f"  ⚠ OpenTopography error: {response.status_code}")
            print("  Trying alternative method...")
            return download_srtm_alternative(bbox, output_path)
        
        temp = 'temp_dem.tif'
        total = int(response.headers.get('content-length', 0))
        
        with open(temp, 'wb') as f:
            dl = 0
            for chunk in response.iter_content(8192):
                f.write(chunk)
                dl += len(chunk)
                if total > 0:
                    print(f"\r  Progress: {dl/total*100:.1f}%", end='', flush=True)
        print()
        
        # Calculate slope and aspect
        with rasterio.open(temp) as src:
            dem = src.read(1)
            profile = src.profile

            dx, dy = np.gradient(dem)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / src.res[0]))

            # Calculate aspect (0-360, where 0=North, clockwise)
            aspect_rad = np.arctan2(-dy, dx)  # Negative dy because y increases downward in image coordinates
            aspect_deg = np.degrees(aspect_rad)
            aspect = (90.0 - aspect_deg) % 360.0  # Convert to 0=North, clockwise

        profile.update(count=3, compress='lzw')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dem.astype('float32'), 1)
            dst.write(slope.astype('float32'), 2)
            dst.write(aspect.astype('float32'), 3)
            dst.set_band_description(1, 'elevation')
            dst.set_band_description(2, 'slope')
            dst.set_band_description(3, 'aspect')
        
        os.remove(temp)
        print(f"✓ Saved: {os.path.getsize(output_path)/(1024**2):.1f} MB")
        return output_path
        
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        print("  Trying alternative method...")
        return download_srtm_alternative(bbox, output_path)


def download_srtm_alternative(bbox, output_path='topography.tif'):
    """
    Alternative: Download SRTM from NASA EARTHDATA directly
    Uses a LARGE buffer to ensure tile coverage at degree boundaries
    """
    print("\n  Using NASA EarthData (SRTM v3)...")
    
    try:
        import earthaccess
        from rasterio.merge import merge
        from shapely.geometry import box
    except ImportError:
        print("  ⚠ earthaccess not installed")
        return create_synthetic_dem(bbox, output_path)
    
    try:
        # FIXED: Increased buffer to 1.0 degree to guarantee neighbor tiles
        buffer = 1.0 
        search_bbox = (bbox[0]-buffer, bbox[1]-buffer, bbox[2]+buffer, bbox[3]+buffer)
        
        results = earthaccess.search_data(
            short_name='SRTMGL1',
            bounding_box=search_bbox
        )
        
        if len(results) == 0:
            print("  ⚠ No SRTM tiles found")
            return create_synthetic_dem(bbox, output_path)
        
        print(f"  Found {len(results)} SRTM tiles (buffered search)")
        
        # Download tiles
        files = earthaccess.download(results, local_path='srtm_temp')
        
        if len(files) == 0:
            return create_synthetic_dem(bbox, output_path)
        
        src_files = []
        for f in files:
            try:
                src = rasterio.open(f)
                src_files.append(src)
            except:
                continue
        
        if len(src_files) == 0:
            return create_synthetic_dem(bbox, output_path)
        
        # Mosaic tiles to handle boundaries
        if len(src_files) > 1:
            print(f"  Merging {len(src_files)} tiles...")
            mosaic, out_trans = merge(src_files)
            dem = mosaic[0]
        else:
            dem = src_files[0].read(1)
            out_trans = src_files[0].transform
        
        # Calculate slope and aspect
        dx, dy = np.gradient(dem)
        pixel_size = 30
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / pixel_size))

        # Calculate aspect (0-360, where 0=North, clockwise)
        aspect_rad = np.arctan2(-dy, dx)  # Negative dy because y increases downward in image coordinates
        aspect_deg = np.degrees(aspect_rad)
        aspect = (90.0 - aspect_deg) % 360.0  # Convert to 0=North, clockwise

        # Update profile
        profile = {
            'driver': 'GTiff',
            'height': dem.shape[0],
            'width': dem.shape[1],
            'count': 3,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': out_trans,
            'compress': 'lzw'
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dem.astype('float32'), 1)
            dst.write(slope.astype('float32'), 2)
            dst.write(aspect.astype('float32'), 3)
            dst.set_band_description(1, 'elevation')
            dst.set_band_description(2, 'slope')
            dst.set_band_description(3, 'aspect')
        
        # Cleanup
        for src in src_files:
            src.close()
        
        print(f"  ✓ Saved real SRTM: {os.path.getsize(output_path)/(1024**2):.1f} MB")
        return output_path
        
    except Exception as e:
        print(f"  ⚠ EarthData SRTM failed: {str(e)[:100]}")
        return create_synthetic_dem(bbox, output_path)
    

def create_synthetic_dem(bbox, output_path):
    """Create synthetic DEM as last resort"""
    print("  Creating synthetic flat topography (0m elevation)...")
    print("  ⚠ This will reduce model accuracy!")
    
    from rasterio.transform import from_bounds as transform_from_bounds
    
    width = 100
    height = 100
    transform = transform_from_bounds(*bbox, width, height)
    
    dem = np.zeros((height, width), dtype='float32')
    slope = np.zeros((height, width), dtype='float32')
    aspect = np.zeros((height, width), dtype='float32')  # Flat aspect for synthetic DEM

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 3,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'lzw'
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dem, 1)
        dst.write(slope, 2)
        dst.write(aspect, 3)
        dst.set_band_description(1, 'elevation')
        dst.set_band_description(2, 'slope')
        dst.set_band_description(3, 'aspect')
    
    print(f"  ✓ Saved: {os.path.getsize(output_path)/(1024**2):.1f} MB")
    return output_path


# ============================================================================
# PART 3: MODEL TRAINING
# ============================================================================

def extract_features(gedi_csv, s2_tif, s1_tif, topo_tif):
    """Extract features with CRS handling and robust nodata replacement"""
    print("\n" + "="*60)
    print("Extracting Features (Fixed CRS & NoData)")
    print("="*60 + "\n")
    
    # Imports needed inside the function scope
    import pandas as pd
    import numpy as np
    import rasterio
    from pyproj import Transformer
    import os
    
    gedi = pd.read_csv(gedi_csv)
    # Ensure points are valid GEDI shots
    valid = (gedi['rh98'] > 0) & gedi['latitude'].notna() & gedi['longitude'].notna()
    gedi = gedi[valid]
    
    print(f"Valid points: {len(gedi)}")
    
    # Base coordinates in Lat/Lon (EPSG:4326)
    lons = gedi['longitude'].values
    lats = gedi['latitude'].values
    coords_4326 = list(zip(lons, lats))
    
    # --- Helper function to sample raster data ---
    def sample_raster(tif_path, band_prefix, lons, lats, coords_4326):
        if not tif_path or not os.path.exists(tif_path):
            # If path is invalid, return an empty array with the correct shape
            return np.array([]).reshape(len(coords_4326), 0), []
            
        with rasterio.open(tif_path) as src:
            print(f"  Sampling {band_prefix} from {os.path.basename(tif_path)}...")
            print(f"    Raster CRS: {src.crs}")
            
            # Reproject points if Raster CRS is not Lat/Lon
            if src.crs.to_string() != 'EPSG:4326':
                print("    ⚠ Reprojecting points to match raster...")
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                # transform expects x, y (lon, lat)
                xx, yy = transformer.transform(lons, lats)
                sample_coords = list(zip(xx, yy))
            else:
                sample_coords = coords_4326
            
            # Sample data (removed unsupported 'nodata=0' argument)
            data = np.array([x for x in src.sample(sample_coords)])
            
            # FIXED: Post-processing to replace raster nodata values with NaN
            if src.nodata is not None and np.isfinite(src.nodata):
                nodata_value = src.nodata
                
                # Replace exact nodata integer/float values with NaN
                data[data == nodata_value] = np.nan
                
                print(f"    Replaced {np.sum(np.isnan(data))} occurrences of nodata ({nodata_value}) with NaN.")
            
            names = [f'{band_prefix}_B{i+1}' for i in range(data.shape[1])]
            
            # Report missing data count (which are NaNs)
            missing_count = np.sum(np.isnan(data))
            print(f"    Missing Data (NaN) count: {missing_count} ({(missing_count/data.size)*100:.1f}%)")
            
            return data, names

    # Sample all sources, passing coordinate variables explicitly
    s2, s2_names = sample_raster(s2_tif, 'S2', lons, lats, coords_4326)
    s1, s1_names = sample_raster(s1_tif, 'S1', lons, lats, coords_4326)
    topo, topo_names = sample_raster(topo_tif, 'Topo', lons, lats, coords_4326)
    
    # Combine S2, S1, and Topo features
    X_list = [s2, topo]
    names_list = s2_names + topo_names
    
    if s1.shape[1] > 0:
        X_list.insert(1, s1)
        names_list = s2_names + s1_names + topo_names
    
    # Remove empty arrays
    X_list = [X for X in X_list if X.shape[1] > 0]
    
    X = np.hstack(X_list)
    y = gedi['rh98'].values
    
    # Remove rows containing NaN or Infinite values
    valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]

    # Also filter coords to match valid samples
    lons_filtered = lons[valid]
    lats_filtered = lats[valid]

    # Add latitude and longitude as features (for region-specific modeling)
    coords_features = np.column_stack([lats_filtered, lons_filtered])
    X = np.column_stack([X, coords_features])
    final_features = names_list + ['latitude', 'longitude']

    print(f"Final training samples: {len(X)}")
    print(f"Total features: {len(final_features)} (including latitude, longitude)")
    return X, y, final_features


def train_model(X, y, names):
    """Train Random Forest"""
    print("\n" + "="*60)
    print("Training Model")
    print("="*60 + "\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=500, max_depth=30,
        min_samples_split=10, min_samples_leaf=2,
        max_features='log2', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R²: {r2:.3f}, RMSE: {rmse:.2f} m")

    # Top features
    imp = pd.DataFrame({'feature': names, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print("\nTop 5 Features:")
    for _, row in imp.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return model


def predict_map(model, s2_tif, s1_tif, topo_tif, output='canopy_height.tif'):
    """Generate height map with lat/lon coordinate features

    Now includes latitude and longitude as input features (18 features total:
    10 Sentinel-2 bands + 3 topographic bands + 2 Sentinel-1 bands + 2 coordinate bands + 1 NDVI)
    """
    print("\n" + "="*60)
    print("Generating Height Map")
    print("="*60 + "\n")

    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    import os

    with rasterio.open(s2_tif) as s2_src:
        s2_data = s2_src.read()
        profile = s2_src.profile

        # Get coordinate grids for latitude and longitude
        h, w = s2_data.shape[1], s2_data.shape[2]

        # Create coordinate arrays
        # Get the coordinates of each pixel center
        rows, cols = np.indices((h, w))
        transform = s2_src.transform

        # Convert pixel coordinates to lat/lon
        # Formula: lon = transform[2] + col * transform[0] + row * transform[1]
        #         lat = transform[5] + col * transform[3] + row * transform[4]
        lons = transform[2] + cols * transform[0] + rows * transform[1]
        lats = transform[5] + cols * transform[3] + rows * transform[4]

        # Resample S1
        if s1_tif and os.path.exists(s1_tif):
            with rasterio.open(s1_tif) as s1_src:
                s1_data = np.zeros((s1_src.count, s2_data.shape[1], s2_data.shape[2]))
                for i in range(s1_src.count):
                    reproject(
                        s1_src.read(i+1), s1_data[i],
                        src_transform=s1_src.transform, src_crs=s1_src.crs,
                        dst_transform=s2_src.transform, dst_crs=s2_src.crs,
                        resampling=Resampling.bilinear
                    )
        else:
            s1_data = np.array([]).reshape(0, s2_data.shape[1], s2_data.shape[2])

        # Resample topo
        with rasterio.open(topo_tif) as topo_src:
            topo_data = np.zeros((topo_src.count, s2_data.shape[1], s2_data.shape[2]))
            for i in range(topo_src.count):
                reproject(
                    topo_src.read(i+1), topo_data[i],
                    src_transform=topo_src.transform, src_crs=topo_src.crs,
                    dst_transform=s2_src.transform, dst_crs=s2_src.crs,
                    resampling=Resampling.bilinear
                )

    # Stack features and save shape for later use
    h, w = s2_data.shape[1], s2_data.shape[2]

    all_data = np.vstack([s2_data, s1_data, topo_data]) if s1_data.shape[0] > 0 else np.vstack([s2_data, topo_data])

    # Add lat/lon as features (reshape to match data format)
    lat_flat = lats.reshape(-1, 1)
    lon_flat = lons.reshape(-1, 1)
    coords_data = np.hstack([lat_flat, lon_flat])

    # Combine: [features, lat, lon]
    all_data_with_coords = np.vstack([all_data.reshape(all_data.shape[0], -1), coords_data.T])

    # Predict
    data_2d = all_data_with_coords.T
    
    valid = np.all(np.isfinite(data_2d), axis=1) & np.all(np.abs(data_2d) < 1e10, axis=1)
    print(f"Valid pixels: {valid.sum()}/{len(valid)} ({valid.sum()/len(valid)*100:.1f}%)")
    
    pred = np.full(len(valid), np.nan)
    if valid.sum() > 0:
        pred[valid] = model.predict(data_2d[valid])
    
    pred = np.clip(pred, 0, 100).reshape(h, w)
    
    # Save
    profile.update(count=1, dtype='float32')
    with rasterio.open(output, 'w', **profile) as dst:
        dst.write(pred.astype('float32'), 1)
    
    print(f"✓ Saved: {output}")
    print(f"  Mean: {np.nanmean(pred):.1f} m, Max: {np.nanmax(pred):.1f} m")


# ============================================================================
# HELPER: Load GEDI from Local Partitions
# ============================================================================

def load_gedi_from_partitions(bbox, output_csv='gedi.csv', gedi_dir='gedi_global_2024_2025'):
    """
    Load GEDI data from local partition folders.

    Parameters:
    -----------
    bbox : list
        [min_lon, min_lat, max_lon, max_lat]
    output_csv : str
        Path to save combined GEDI CSV
    gedi_dir : str
        Path to GEDI partition directory

    Returns:
    --------
    str : Path to combined GEDI CSV
    """
    import pandas as pd
    from pathlib import Path

    min_lon, min_lat, max_lon, max_lat = bbox

    gedi_path = Path(gedi_dir)
    if not gedi_path.exists():
        raise Exception(f"GEDI directory not found: {gedi_dir}")

    print(f"  BBox: {bbox}")

    partition_folders = [d for d in gedi_path.iterdir() if d.is_dir() and d.name.startswith('lat_')]
    print(f"  Found {len(partition_folders)} partition folders")

    all_data = []

    for folder in sorted(partition_folders):
        parts = folder.name.split('_')
        if len(parts) >= 4:
            try:
                lat = float(parts[1])
                lon = float(parts[3])
            except ValueError:
                continue

            if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat):
                pq_file = folder / 'part.parquet'
                if pq_file.exists():
                    try:
                        df = pd.read_parquet(pq_file)
                        df = df[
                            (df['longitude'] >= min_lon) &
                            (df['longitude'] <= max_lon) &
                            (df['latitude'] >= min_lat) &
                            (df['latitude'] <= max_lat)
                        ]
                        if len(df) > 0:
                            all_data.append(df)
                    except Exception:
                        continue

    if not all_data:
        raise Exception(f"No GEDI data found in bbox {bbox}")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['longitude', 'latitude'])

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"  ✓ Loaded {len(combined):,} GEDI points")
    print(f"  ✓ Saved to: {output_path}")

    return str(output_path)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import time
    start_time = time.time()

    # CONFIG
    # bbox = [-60.0, -3.5, -59.8, -3.3]
    # bbox = [-81.75, 29.05, -81.65, 29.15]
    # bbox = [-81.0, 29, -80.0, 30]
    bbox = [-117.28, 32.53, -116.42, 33.51]   # San Diego
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    output_dir = 'output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CANOPY HEIGHT MAPPING PIPELINE")
    print("="*60 + "\n")
    
    # STEP 1: Load GEDI from local partitions (FAST - uses gedi_global_2024_2025/)
    print("STEP 1: GEDI Data\n")
    print("Loading GEDI from local partitions (gedi_global_2024_2025/)...")

    # Load from local global partitions
    gedi_csv = load_gedi_from_partitions(
        bbox,
        output_csv=f'{output_dir}/gedi_raw.csv',
        gedi_dir='gedi_global_2024_2025'
    )

    if gedi_csv is None:
        raise Exception("Failed to load GEDI from local partitions!")
    
    # Augment if sparse
    df = pd.read_csv(gedi_csv)
    if len(df) < 50:
        print(f"\n⚠ Only {len(df)} GEDI points - augmenting...")
        gedi_csv = augment_gedi_with_synthetic(gedi_csv, multiplier=5)
    
    # STEP 2: Download satellite data
    print("\n\nSTEP 2: Satellite Data\n")
    
    # Use default resolution if not defined (shouldn't happen, but safety check)
    if 'resolution' not in locals():
        resolution = 30
        print(f"⚠ Resolution not defined, using default: {resolution}m\n")
    
    s2_path = download_sentinel2_mpc(bbox, start_date, end_date, 
                                     f'{output_dir}/sentinel2.tif',
                                     resolution=resolution)
    s1_path = download_sentinel1_mpc(bbox, start_date, end_date, 
                                     f'{output_dir}/sentinel1.tif')
    topo_path = download_srtm_opentopography(bbox, f'{output_dir}/topography.tif')
    
    # STEP 3: Train model
    print("\n\nSTEP 3: Model Training\n")
    
    X, y, features = extract_features(gedi_csv, s2_path, s1_path, topo_path)
    model = train_model(X, y, features)
    
    # STEP 4: Predict
    print("\n\nSTEP 4: Prediction\n")
    
    predict_map(model, s2_path, s1_path, topo_path, 
                f'{output_dir}/canopy_height_map.tif')
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print(f"\nOutput: {output_dir}/canopy_height_map.tif")
    print("\nVisualize with QGIS, Python (rasterio/matplotlib), or any GIS software")

    # Print total execution time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n{'='*60}")
    print(f"Total execution time: {int(minutes)}m {seconds:.1f}s")
    print('='*60)
