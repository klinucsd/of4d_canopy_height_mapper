"""
Quick Demo: Canopy Height Mapping Pipeline (Small Area)
========================================================

This script demonstrates the complete pipeline on a small area near San Diego.
It's designed to run quickly (5-15 minutes) for testing and demonstration.

Study Area: Part of San Diego County, California
Size: ~0.86° x 0.98° (~85 km x 95 km)
Time Period: 2022-2023

Outputs:
- All plots saved to examples/demo_outputs/
- Final canopy height map
- Model validation report
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from complete_canopy_height_pipeline import *
import visualizations as viz
import validation as val
import time
from pathlib import Path


def main():
    """Run the quick demo pipeline"""
    start_time = time.time()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    print("="*70)
    print("CANOPY HEIGHT MAPPING - QUICK DEMO")
    print("="*70)
    print("\nConfiguration:")
    print("  Study Area: San Diego County (subset)")
    print("  Time Period: 2022-2023")
    print("  Resolution: 30m")
    print("  Max Scenes: 5 (for speed)")
    print("="*70 + "\n")

    # Small bbox for quick demo (subset of San Diego)
    bbox = [-117.1, 32.7, -116.9, 32.9]  # ~0.2° x 0.2° area
    start_date = '2022-01-01'
    end_date = '2023-12-31'

    # Output directory
    output_dir = 'examples/demo_outputs'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Download GEDI Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: GEDI L2A Data Download")
    print("="*70 + "\n")

    gedi_csv = download_gedi_earthaccess(
        bbox, start_date, end_date,
        f'{output_dir}/gedi_raw.csv',
        n_workers=2
    )

    if gedi_csv is None:
        print("\n⚠ GEDI download failed. Using synthetic demonstration data...")
        # Create synthetic GEDI data for demo
        np.random.seed(42)
        n_points = 500

        synthetic_gedi = pd.DataFrame({
            'latitude': np.random.uniform(bbox[1], bbox[3], n_points),
            'longitude': np.random.uniform(bbox[0], bbox[2], n_points),
            'rh98': np.random.beta(2, 5, n_points) * 40 + 5  # Realistic height distribution
        })

        gedi_csv = f'{output_dir}/gedi_synthetic.csv'
        synthetic_gedi.to_csv(gedi_csv, index=False)
        print(f"  Created synthetic GEDI data: {n_points} points")

    # Visualize GEDI data
    print("\nVisualizing GEDI data coverage...")
    viz.plot_gedi_data_coverage(gedi_csv, bbox, output_dir)
    viz.plot_gedi_tracks(gedi_csv, output_dir)

    # ========================================================================
    # STEP 2: Download Satellite Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Satellite Data Download")
    print("="*70 + "\n")

    # Sentinel-2 (fewer scenes for speed)
    print("Downloading Sentinel-2 (max 5 scenes for demo)...")
    s2_path = download_sentinel2_mpc(
        bbox, start_date, end_date,
        f'{output_dir}/sentinel2.tif',
        max_items=5,
        resolution=30,
        n_workers=2
    )

    # Visualize Sentinel-2
    print("\nVisualizing Sentinel-2 data...")
    viz.plot_sentinel2_bands(s2_path, output_dir)
    viz.plot_sentinel2_composite(s2_path, bbox, output_dir)
    viz.plot_band_histograms(s2_path, output_dir)

    # Sentinel-1 (optional, may fail)
    print("\nDownloading Sentinel-1 (optional)...")
    s1_path = download_sentinel1_mpc(
        bbox, start_date, end_date,
        f'{output_dir}/sentinel1.tif',
        max_items=3
    )

    # SRTM Topography
    print("\nDownloading SRTM topography...")
    topo_path = download_srtm_opentopography(bbox, f'{output_dir}/topography.tif')

    # ========================================================================
    # STEP 3: Model Training
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Model Training and Validation")
    print("="*70 + "\n")

    # Extract features
    print("Extracting features at GEDI locations...")
    X, y, features = extract_features(gedi_csv, s2_path, s1_path, topo_path)
    print(f"  Extracted {len(X)} samples with {len(features)} features")

    # Train model
    print("\nTraining Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nModel Performance:")
    print("-" * 50)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  R²:  {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} m")
    print(f"  MAE:  {mae:.2f} m")

    # Create validation plots
    print("\nGenerating validation plots...")
    viz.plot_model_validation(
        y_test, y_pred,
        feature_names=features,
        feature_importance=model.feature_importances_,
        output_dir=output_dir
    )

    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_summary = val.perform_cross_validation(model, X, y, n_folds=5)
    val.print_cv_report(cv_summary)

    # Feature importance
    print("\nFeature Importance:")
    print("-" * 50)
    imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # ========================================================================
    # STEP 4: Generate Canopy Height Map
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Generating Canopy Height Map")
    print("="*70 + "\n")

    predict_map(
        model, s2_path, s1_path, topo_path,
        f'{output_dir}/canopy_height_map.tif'
    )

    # Visualize final map
    print("\nVisualizing final canopy height map...")
    viz.plot_canopy_height_map(
        f'{output_dir}/canopy_height_map.tif',
        gedi_csv=gedi_csv,
        title="Canopy Height Map - San Diego Demo Area",
        output_dir=output_dir
    )

    # ========================================================================
    # STEP 5: Generate Summary Report
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Generating Summary Report")
    print("="*70 + "\n")

    model_stats = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'feature_importance': imp.to_dict('records')
    }

    viz.create_pipeline_summary(
        gedi_csv, s2_path, f'{output_dir}/canopy_height_map.tif',
        model_stats, output_dir
    )

    # Save model
    import joblib
    joblib.dump(model, f'{output_dir}/canopy_height_model.pkl')
    print("\n  Saved trained model to demo_outputs/canopy_height_model.pkl")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)

    print("\n" + "="*70)
    print("QUICK DEMO COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  Data:")
    print("    - gedi_raw.csv (or gedi_synthetic.csv)")
    print("    - sentinel2.tif")
    print("    - topography.tif")
    print("  Maps:")
    print("    - canopy_height_map.tif")
    print("  Visualizations:")
    print("    - gedi_data_coverage.png")
    print("    - gedi_tracks.png")
    print("    - sentinel2_bands.png")
    print("    - sentinel2_composites.png")
    print("    - model_validation.png")
    print("    - canopy_height_map.png")
    print("    - pipeline_summary.png")
    print("  Model:")
    print("    - canopy_height_model.pkl")
    print("\nModel Performance:")
    print(f"    R²:  {r2:.4f}")
    print(f"    RMSE: {rmse:.2f} m")
    print(f"    MAE:  {mae:.2f} m")
    print(f"\nExecution time: {int(minutes)}m {seconds:.1f}s")
    print("="*70)


if __name__ == "__main__":
    main()
