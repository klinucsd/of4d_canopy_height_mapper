"""
Europe Lat/Lon Feature Test
===========================

Purpose: Quick test to verify lat/lon feature performance on a Europe region.

Study Area: Small region in Europe (around 50km x 50km near border of France/Germany/Switzerland)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import from main pipeline
from complete_canopy_height_pipeline import (
    download_sentinel2_mpc,
    download_sentinel1_mpc,
    download_srtm_opentopography,
    extract_features
)


def load_gedi_from_partitions(bbox, output_csv='gedi.csv', gedi_dir='gedi_global_2024_2025'):
    """Load GEDI from global partitions."""
    min_lon, min_lat, max_lon, max_lat = bbox

    gedi_path = Path(gedi_dir)
    if not gedi_path.exists():
        raise Exception(f"GEDI directory not found: {gedi_dir}")

    print(f"Loading GEDI from {gedi_dir}...")
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


def extract_features_with_coords(gedi_csv, s2_tif, s1_tif, topo_tif):
    """Extract features including lat/lon."""
    print("\nExtracting features at GEDI locations...")

    gedi = pd.read_csv(gedi_csv)
    gedi_coords = gedi[['latitude', 'longitude']].values

    # Extract features using standard function
    from complete_canopy_height_pipeline import extract_features as extract_features_base
    X, y, features = extract_features_base(gedi_csv, s2_tif, s1_tif, topo_tif)

    # Add coordinates as features
    X_with_coords = np.column_stack([X, gedi_coords])
    features_with_coords = features + ['latitude', 'longitude']

    print(f"  Extracted {len(X)} samples with {len(features)} features")
    print(f"  WITHOUT coords: {len(features)} features")
    print(f"  WITH coords: {len(features_with_coords)} features")

    return X, X_with_coords, y, features, features_with_coords


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, model_name="Model"):
    """Train model and return metrics."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'feature_importance': pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    }


def main():
    """Run Europe lat/lon test."""
    start_time = time.time()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    print("="*70)
    print("EUROPE LAT/LON FEATURE TEST")
    print("="*70)
    print("\nConfiguration:")
    print("  Region: Europe (France/Germany/Switzerland border area)")
    print("  Size: ~50km x 50km")
    print("  Train/Test: 80/20 split")
    print("="*70 + "\n")

    # Small region in Europe (around Basel/Mulhouse/Freiburg area)
    bbox = [7.3, 47.4, 8.0, 48.0]  # Near France/Germany/Switzerland border

    output_dir = 'europe_latlon_test/output'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load GEDI Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Load GEDI Data from Global Partitions")
    print("="*70 + "\n")

    script_dir = Path(__file__).parent.parent
    gedi_dir = str(script_dir / 'gedi_global_2024_2025')

    gedi_csv = load_gedi_from_partitions(
        bbox,
        output_csv=f'{output_dir}/gedi_raw.csv',
        gedi_dir=gedi_dir
    )

    # ========================================================================
    # STEP 2: Download Satellite Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Satellite Data Download")
    print("="*70 + "\n")

    satellite_start_date = '2022-01-01'
    satellite_end_date = '2024-12-31'

    print("Downloading Sentinel-2 (10 bands)...")
    s2_path = download_sentinel2_mpc(
        bbox, satellite_start_date, satellite_end_date,
        f'{output_dir}/sentinel2.tif',
        max_items=5,  # Fewer scenes for small region
        resolution=30,
        n_workers=4
    )

    print("\nDownloading Sentinel-1...")
    s1_path = download_sentinel1_mpc(
        bbox, satellite_start_date, satellite_end_date,
        f'{output_dir}/sentinel1.tif',
        max_items=3
    )

    print("\nDownloading COP30 Topography...")
    topo_path = download_srtm_opentopography(bbox, f'{output_dir}/topography.tif', demtype='COP30')

    # ========================================================================
    # STEP 3: Extract and Train Models
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Extract Features & Train Models")
    print("="*70 + "\n")

    print("Extracting features at GEDI locations...")
    X_baseline, X_with_coords, y, features_baseline, features_with_coords, coords = extract_features_with_coords(
        gedi_csv, s2_path, s1_path, topo_path
    )

    # Split data
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42
    )

    # Split coords data the same way
    X_train_coords, X_test_coords, _, _ = train_test_split(
        X_with_coords, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {len(X_train_base):,}, Test size: {len(X_test_base):,}\n")

    # Train baseline model (WITHOUT lat/lon)
    print("Training baseline model (WITHOUT lat/lon)...")
    results_baseline = train_and_evaluate(
        X_train_base, X_test_base, y_train, y_test,
        features_baseline, "Baseline (without coords)"
    )
    print(f"  R²:  {results_baseline['r2']:.4f}")
    print(f"  RMSE: {results_baseline['rmse']:.2f} m")
    print(f"  MAE:  {results_baseline['mae']:.2f} m")

    # Train model WITH lat/lon
    print("\nTraining model (WITH lat/lon)...")
    results_with_coords = train_and_evaluate(
        X_train_coords, X_test_coords, y_train, y_test,
        features_with_coords, "With coords"
    )
    print(f"  R²:  {results_with_coords['r2']:.4f}")
    print(f"  RMSE: {results_with_coords['rmse']:.2f} m")
    print(f"  MAE:  {results_with_coords['mae']:.2f} m")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")

    print("\n" + "-" * 70)
    print("SUMMARY:")
    print("-" * 70)
    print(f"  WITHOUT lat/lon: R²={results_baseline['r2']:.4f}, RMSE={results_baseline['rmse']:.2f}m")
    print(f"  WITH lat/lon:    R²={results_with_coords['r2']:.4f}, RMSE={results_with_coords['rmse']:.2f}m")

    r2_change = ((results_with_coords['r2'] - results_baseline['r2']) / results_baseline['r2']) * 100
    rmse_change = ((results_baseline['rmse'] - results_with_coords['rmse']) / results_baseline['rmse']) * 100

    print(f"\n  R² change: {r2_change:+.1f}%")
    print(f"  RMSE change: {rmse_change:+.1f}%")
    print("-" * 70)

    print(f"\nExecution time: {int(minutes)}m {seconds:.1f}s")
    print("\n" + "="*70)

    # Show feature importance
    print("\nTop 10 Features (WITH lat/lon):")
    for i, row in results_with_coords['feature_importance'].head(10).iterrows():
        highlight = " <--" if row['feature'] in ['latitude', 'longitude'] else ""
        print(f"  {i+1:2d}. {row['feature']:15s} {row['importance']:.4f} {highlight}")


if __name__ == "__main__":
    import time
    main()
