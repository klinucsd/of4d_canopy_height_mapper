"""
Canopy Height Mapping Pipeline - Using Partitioned GEDI Data
===============================================================

This version uses pre-partitioned GEDI data instead of downloading HDF5 files.
Much faster for regions that already have downloaded GEDI data.

Key Differences from complete_canopy_height_pipeline.py:
- Uses load_gedi_for_bbox() to load from partitioned Parquet files
- No NASA EarthData credentials required for GEDI (already downloaded)
- Instant access to GEDI data for any bbox in USA
- Skips GEDI download step entirely

Requirements:
pip install rasterio numpy pandas scikit-learn requests pystac-client planetary-computer scipy python-dotenv pyarrow
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pystac_client
import planetary_computer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

# Import partition loader
from canopy_height_gedi_loader import load_gedi_for_bbox, get_gedi_stats

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# PART 1: GEDI DATA LOADING (FROM PARTITIONS)
# ============================================================================

def load_gedi_from_partitions(bbox, output_csv='gedi_from_partitions.csv', gedi_dir='gedi_usa_2024_2025'):
    """
    Load GEDI data from pre-partitioned Parquet files.

    This is MUCH faster than downloading HDF5 files and processing them.
    The data is already filtered by quality flag and has valid rh98 values.

    Parameters:
    -----------
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    output_csv : str
        Path to save loaded GEDI data as CSV (optional)
    gedi_dir : str
        Directory containing partitioned GEDI data

    Returns:
    --------
    str : Path to the CSV file with loaded GEDI data
    """
    print("\n" + "="*60)
    print("Loading GEDI Data from Partitions")
    print("="*60 + "\n")

    print(f"  BBox: {bbox}")
    print(f"  GEDI Directory: {gedi_dir}")

    # Get statistics first
    stats = get_gedi_stats(bbox, gedi_dir)

    if stats['total_rows'] == 0:
        print(f"  ⚠ No GEDI data found for this bbox!")
        print(f"  Partitions checked: {stats['partitions_loaded']}")
        return None

    print(f"  ✓ Found {stats['total_rows']:,} GEDI points")
    print(f"  Partitions loaded: {stats['partitions_loaded']}")
    print(f"  Area: {stats['bbox_area_km2']:.1f} km²")
    print(f"  Density: {stats['points_per_km2']:.1f} pts/km²")

    # Load the data
    gedi_df = load_gedi_for_bbox(bbox, gedi_dir)

    if len(gedi_df) == 0:
        print("  ⚠ No data returned from loader")
        return None

    # Save to CSV for compatibility with existing pipeline
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        gedi_df.to_csv(output_csv, index=False)
        print(f"  ✓ Saved to: {output_csv}")

    return output_csv


# ============================================================================
# PART 2: SATELLITE DATA DOWNLOAD (Same as original)
# ============================================================================

def _process_sentinel2_scene(item, bbox, bands, resolution, scene_idx):
    """Worker function to process a single Sentinel-2 scene."""
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

                if src.crs.to_string() != 'EPSG:4326':
                    bbox_transformed = transform_bounds('EPSG:4326', src.crs, *bbox)
                else:
                    bbox_transformed = bbox

                window = from_bounds(*bbox_transformed, transform=src.transform)
                window = window.round_offsets().round_lengths()

                if window.width <= 0 or window.height <= 0:
                    return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': 'Invalid window'}

                data = src.read(1, window=window)

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

                target_shape = (profile['height'], profile['width'])

                if data.shape != target_shape:
                    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
                    data = zoom(data, zoom_factors, order=1)

                scene_data[b_name] = data

        except Exception as e:
            return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': str(e)}

    if scene_valid and len(scene_data) == len(bands):
        return {'scene_idx': scene_idx, 'scene_data': scene_data, 'profile': profile, 'crs': crs, 'error': None}
    else:
        return {'scene_idx': scene_idx, 'scene_data': None, 'profile': None, 'crs': None, 'error': 'Incomplete bands'}


def download_sentinel2_mpc(bbox, start_date, end_date, output_path='sentinel2.tif', max_items=10, resolution=30, n_workers=None, cloud_threshold=20):
    """Download Sentinel-2 from Microsoft Planetary Computer.

    Parameters:
    -----------
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    output_path : str
        Output file path
    max_items : int
        Maximum number of scenes to use
    resolution : int
        Target resolution in meters
    n_workers : int, optional
        Number of parallel workers
    cloud_threshold : int
        Maximum cloud cover percentage (0-100)
    """
    print("\n" + "="*60)
    print("Downloading Sentinel-2 from MS Planetary Computer")
    print("="*60 + "\n")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_threshold}}
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

    if n_workers is None:
        n_workers = min(4, multiprocessing.cpu_count())

    print(f"  Using {n_workers} parallel workers...\n")

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
                    print(f"  ✓ {item.id[:40]}...")
                else:
                    print(f"  ✗ {item.id[:40]}... {result['error'][:50]}")
            except Exception as e:
                print(f"  ✗ {item.id[:40]}... Exception: {str(e)[:50]}")

    results.sort(key=lambda x: x['scene_idx'])

    for result in results:
        if result['scene_data'] is not None:
            if ref_profile is None:
                ref_profile = result['profile']
                ref_crs = result['crs']
                print(f"\n  Reference CRS: {ref_crs}")
                print(f"  Target shape: {ref_profile['height']}x{ref_profile['width']} at {resolution}m")

            for b_name, data in result['scene_data'].items():
                all_bands[b_name].append(data)

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

    valid_bands = {k: v for k, v in all_bands.items() if len(v) > 0}
    print(f"\n  Bands with data: {list(valid_bands.keys())}")

    if not valid_bands or not all(len(v) > 0 for v in all_bands.values()):
        raise Exception(f"Failed to download all bands. Got: {list(valid_bands.keys())}")

    if ref_profile is None:
        raise Exception("No valid scenes were read")

    print("\nComputing median...")
    medians = {k: np.nanmedian(np.stack(v), axis=0) for k, v in all_bands.items()}

    ndvi = (medians['B08'] - medians['B04']) / (medians['B08'] + medians['B04'] + 1e-10)
    medians['NDVI'] = ndvi

    print(f"Saving to {output_path}...")

    # Update profile count to match actual number of bands (including NDVI)
    ref_profile = ref_profile.copy()
    ref_profile['count'] = len(medians)

    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        for i, (name, data) in enumerate(medians.items(), 1):
            dst.write(data.astype('float32'), i)
            dst.set_band_description(i, name)

    file_size_mb = os.path.getsize(output_path)/(1024**2)
    print(f"✓ Saved: {file_size_mb:.1f} MB")

    return output_path


def download_sentinel1_mpc(bbox, start_date, end_date, output_path='sentinel1.tif', max_items=5):
    """Download Sentinel-1 SAR."""
    print("\n" + "="*60)
    print("Downloading Sentinel-1 from MS Planetary Computer")
    print("="*60 + "\n")

    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print("  Trying Sentinel-1 RTC...")
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

    print("\n  Trying Sentinel-1 GRD...")
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
    """Download Sentinel-1 RTC scenes."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from scipy.ndimage import zoom

    vv_arrays, vh_arrays = [], []
    ref_profile = None

    for item in items:
        print(f"  {item.id[:40]}...")
        scene_vv = None
        scene_vh = None

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

                    ref_shape = (ref_profile['height'], ref_profile['width'])

                    if scene_vv.shape != ref_shape:
                        zoom_factors = (ref_shape[0] / scene_vv.shape[0], ref_shape[1] / scene_vv.shape[1])
                        scene_vv = zoom(scene_vv, zoom_factors, order=1)

                    if scene_vh.shape != ref_shape:
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
    """Download GRD with manual CRS assignment."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from pyproj import CRS

    vv_arrays, vh_arrays = [], []
    ref_profile = None

    for item in items:
        print(f"  {item.id[:40]}...")

        proj_string = item.properties.get('proj:epsg', None)
        if proj_string:
            manual_crs = f"EPSG:{proj_string}"
        else:
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
                        if src.crs is None:
                            src_crs = CRS.from_string(manual_crs)
                            print(f"    Assigning CRS: {manual_crs}")
                        else:
                            src_crs = src.crs

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


def download_srtm_opentopography(bbox, output_path='topography.tif', dem_type='COP30'):
    """Download DEM from OpenTopography.

    Parameters:
    -----------
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    output_path : str
        Output file path
    dem_type : str
        DEM dataset type (COP30, SRTMGL1, etc.)
    """
    print("\n" + "="*60)
    print(f"Downloading {dem_type} DEM from OpenTopography")
    print("="*60 + "\n")

    import requests

    base_url = "https://portal.opentopography.org/API/globaldem"

    params = {
        'demtype': dem_type,
        'south': bbox[1], 'north': bbox[3],
        'west': bbox[0], 'east': bbox[2],
        'outputFormat': 'GTiff',
    }

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
    """Alternative: Download SRTM from NASA EARTHDATA directly."""
    print("\n  Using NASA EarthData (SRTM v3)...")

    try:
        import earthaccess
        from rasterio.merge import merge
    except ImportError:
        print("  ⚠ earthaccess not installed")
        return create_synthetic_dem(bbox, output_path)

    try:
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

        if len(src_files) > 1:
            print(f"  Merging {len(src_files)} tiles...")
            mosaic, out_trans = merge(src_files)
            dem = mosaic[0]
        else:
            dem = src_files[0].read(1)
            out_trans = src_files[0].transform

        dx, dy = np.gradient(dem)
        pixel_size = 30
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / pixel_size))

        # Calculate aspect (0-360, where 0=North, clockwise)
        aspect_rad = np.arctan2(-dy, dx)  # Negative dy because y increases downward in image coordinates
        aspect_deg = np.degrees(aspect_rad)
        aspect = (90.0 - aspect_deg) % 360.0  # Convert to 0=North, clockwise

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

        for src in src_files:
            src.close()

        print(f"  ✓ Saved real SRTM: {os.path.getsize(output_path)/(1024**2):.1f} MB")
        return output_path

    except Exception as e:
        print(f"  ⚠ EarthData SRTM failed: {str(e)[:100]}")
        return create_synthetic_dem(bbox, output_path)


def create_synthetic_dem(bbox, output_path):
    """Create synthetic DEM as last resort."""
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
# PART 3: MODEL TRAINING (Same as original)
# ============================================================================

def extract_features(gedi_csv, s2_tif, s1_tif, topo_tif):
    """Extract features with CRS handling."""
    print("\n" + "="*60)
    print("Extracting Features")
    print("="*60 + "\n")

    gedi = pd.read_csv(gedi_csv)
    valid = (gedi['rh98'] > 0) & gedi['latitude'].notna() & gedi['longitude'].notna()
    gedi = gedi[valid]

    print(f"Valid points: {len(gedi)}")

    lons = gedi['longitude'].values
    lats = gedi['latitude'].values
    coords_4326 = list(zip(lons, lats))

    def sample_raster(tif_path, band_prefix, lons, lats, coords_4326):
        if not tif_path or not os.path.exists(tif_path):
            return np.array([]).reshape(len(coords_4326), 0), []

        with rasterio.open(tif_path) as src:
            print(f"  Sampling {band_prefix} from {os.path.basename(tif_path)}...")
            print(f"    Raster CRS: {src.crs}")

            if src.crs.to_string() != 'EPSG:4326':
                from pyproj import Transformer
                print("    ⚠ Reprojecting points to match raster...")
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xx, yy = transformer.transform(lons, lats)
                sample_coords = list(zip(xx, yy))
            else:
                sample_coords = coords_4326

            data = np.array([x for x in src.sample(sample_coords)])

            if src.nodata is not None and np.isfinite(src.nodata):
                nodata_value = src.nodata
                data[data == nodata_value] = np.nan
                print(f"    Replaced {np.sum(np.isnan(data))} occurrences of nodata ({nodata_value}) with NaN.")

            names = [f'{band_prefix}_B{i+1}' for i in range(data.shape[1])]

            missing_count = np.sum(np.isnan(data))
            print(f"    Missing Data (NaN) count: {missing_count} ({(missing_count/data.size)*100:.1f}%)")

            return data, names

    s2, s2_names = sample_raster(s2_tif, 'S2', lons, lats, coords_4326)
    s1, s1_names = sample_raster(s1_tif, 'S1', lons, lats, coords_4326)
    topo, topo_names = sample_raster(topo_tif, 'Topo', lons, lats, coords_4326)

    X_list = [s2, topo]
    names_list = s2_names + topo_names

    if s1.shape[1] > 0:
        X_list.insert(1, s1)
        names_list = s2_names + s1_names + topo_names

    X_list = [X for X in X_list if X.shape[1] > 0]

    X = np.hstack(X_list)
    y = gedi['rh98'].values

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
    """Train Random Forest."""
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

    imp = pd.DataFrame({'feature': names, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print("\nTop 5 Features:")
    for _, row in imp.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return model


def predict_map(model, s2_tif, s1_tif, topo_tif, output='canopy_height.tif'):
    """Generate height map with lat/lon coordinate features."""
    print("\n" + "="*60)
    print("Generating Height Map")
    print("="*60 + "\n")

    with rasterio.open(s2_tif) as s2_src:
        s2_data = s2_src.read()
        profile = s2_src.profile

        # Get coordinate grids for latitude and longitude
        h, w = s2_data.shape[1], s2_data.shape[2]

        # Create coordinate arrays
        rows, cols = np.indices((h, w))
        transform = s2_src.transform

        # Convert pixel coordinates to lat/lon
        lons = transform[2] + cols * transform[0] + rows * transform[1]
        lats = transform[5] + cols * transform[3] + rows * transform[4]

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

        with rasterio.open(topo_tif) as topo_src:
            topo_data = np.zeros((topo_src.count, s2_data.shape[1], s2_data.shape[2]))
            for i in range(topo_src.count):
                reproject(
                    topo_src.read(i+1), topo_data[i],
                    src_transform=topo_src.transform, src_crs=topo_src.crs,
                    dst_transform=s2_src.transform, dst_crs=s2_src.crs,
                    resampling=Resampling.bilinear
                )

    # Stack features
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

    profile.update(count=1, dtype='float32')
    with rasterio.open(output, 'w', **profile) as dst:
        dst.write(pred.astype('float32'), 1)

    print(f"✓ Saved: {output}")
    print(f"  Mean: {np.nanmean(pred):.1f} m, Max: {np.nanmax(pred):.1f} m")
