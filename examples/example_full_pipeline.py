"""
Complete Pipeline: Canopy Height Mapping (Full Study Area)
===========================================================

This script demonstrates the complete canopy height mapping pipeline
for a larger study area with full processing and validation.

Study Area: San Diego County, California
Size: ~0.86° x 0.98° (~85 km x 95 km)
Time Period: 2019-2023 (extended for more GEDI data)

Features:
- Complete GEDI L2A data acquisition
- Full Sentinel-2 and Sentinel-1 processing
- Topographic variables from SRTM
- Comprehensive model validation
- Cross-validation
- Uncertainty quantification
- Comparison maps

Expected runtime: 1-3 hours (depending on data availability)
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
    """Run the complete pipeline"""
    start_time = time.time()

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    print("="*70)
    print("CANOPY HEIGHT MAPPING - FULL PIPELINE")
    print("="*70)
    print("\nStudy Area: San Diego County, California")
    print("  Bbox: [-117.28, 32.53, -116.42, 33.51]")
    print("  Size: ~0.86° x 0.98° (~85 km x 95 km)")
    print("\nTime Period: 2019-2023")
    print("  Extended range for maximum GEDI coverage")
    print("\nProcessing Options:")
    print("  Resolution: 30m")
    print("  Max S2 scenes: 10")
    print("  Max S1 scenes: 5")
    print("  Workers: 4")
    print("="*70 + "\n")

    # Configuration
    bbox = [-117.28, 32.53, -116.42, 33.51]
    start_date = '2019-04-01'
    end_date = '2023-12-31'

    output_dir = 'examples/full_outputs'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PART 1: GEDI DATA ACQUISITION AND VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("PART 1: GEDI L2A DATA ACQUISITION")
    print("="*70 + "\n")

    gedi_csv = download_gedi_earthaccess(
        bbox, start_date, end_date,
        f'{output_dir}/gedi_raw.csv',
        n_workers=4
    )

    if gedi_csv is None:
        # Try existing files
        for f in ['gedi_downloads/GEDI_L2A_rh98_2019-04-01_2024-12-31_clean.csv',
                  'gedi_downloads/GEDI_L2A_rh98_2019-04-01_2024-12-31.csv']:
            if Path(f).exists():
                df = pd.read_csv(f)
                df_bbox = df[(df['longitude'] >= bbox[0]) & (df['longitude'] <= bbox[2]) &
                            (df['latitude'] >= bbox[1]) & (df['latitude'] <= bbox[3])]
                gedi_csv = f'{output_dir}/gedi_raw.csv'
                df_bbox.to_csv(gedi_csv, index=False)
                print(f"✓ Using existing file: {len(df_bbox)} points")
                break

    if gedi_csv is None:
        raise Exception("No GEDI data available!")

    # Check if augmentation needed
    df = pd.read_csv(gedi_csv)
    print(f"\nGEDI points: {len(df)}")

    if len(df) < 100:
        print(f"\n⚠ Sparse GEDI coverage ({len(df)} points)")
        print("  Augmenting with synthetic points...")
        gedi_csv = augment_gedi_with_synthetic(gedi_csv, multiplier=5)

    # Visualize GEDI data
    print("\nGenerating GEDI visualizations...")
    viz.plot_gedi_data_coverage(gedi_csv, bbox, output_dir)
    viz.plot_gedi_tracks(gedi_csv, output_dir)

    # ========================================================================
    # PART 2: SATELLITE DATA ACQUISITION
    # ========================================================================
    print("\n" + "="*70)
    print("PART 2: SATELLITE DATA ACQUISITION")
    print("="*70 + "\n")

    # Sentinel-2
    print("Sentinel-2 L2A")
    print("-" * 40)
    s2_path = download_sentinel2_mpc(
        bbox, start_date, end_date,
        f'{output_dir}/sentinel2.tif',
        max_items=10,
        resolution=30,
        n_workers=4
    )

    # Visualize Sentinel-2
    print("\nGenerating Sentinel-2 visualizations...")
    viz.plot_sentinel2_bands(s2_path, output_dir)
    viz.plot_sentinel2_composite(s2_path, bbox, output_dir)
    viz.plot_band_histograms(s2_path, output_dir)

    # Sentinel-1
    print("\nSentinel-1 SAR")
    print("-" * 40)
    s1_path = download_sentinel1_mpc(
        bbox, start_date, end_date,
        f'{output_dir}/sentinel1.tif',
        max_items=5
    )

    # SRTM
    print("\nSRTM Topography")
    print("-" * 40)
    topo_path = download_srtm_opentopography(bbox, f'{output_dir}/topography.tif')

    # ========================================================================
    # PART 3: FEATURE EXTRACTION AND MODEL TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print("PART 3: FEATURE EXTRACTION AND MODEL TRAINING")
    print("="*70 + "\n")

    # Extract features
    print("Extracting features at GEDI locations...")
    X, y, features = extract_features(gedi_csv, s2_path, s1_path, topo_path)
    print(f"\nExtracted {len(X)} samples with {len(features)} features:")
    for feat in features:
        print(f"  - {feat}")

    # Train/test split
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    # ========================================================================
    # PART 4: MODEL VALIDATION
    # ========================================================================
    print("\n" + "="*70)
    print("PART 4: MODEL VALIDATION")
    print("="*70 + "\n")

    # Test set evaluation
    print("Test Set Evaluation")
    print("-" * 40)
    y_pred = model.predict(X_test)

    test_metrics = val.evaluate_model(y_test, y_pred, "Test Set")
    val.print_model_report(test_metrics)

    # Validation plots
    viz.plot_model_validation(
        y_test, y_pred,
        feature_names=features,
        feature_importance=model.feature_importances_,
        output_dir=output_dir
    )

    # Cross-validation
    print("\n5-Fold Cross-Validation")
    print("-" * 40)
    cv_summary = val.perform_cross_validation(model, X, y, n_folds=5)
    val.print_cv_report(cv_summary)

    viz.plot_cross_validation(cv_summary['fold_scores'], output_dir=output_dir)

    # Feature importance
    print("\nFeature Importance")
    print("-" * 40)
    importance_df = val.analyze_feature_importance(model, features, output_dir)

    # Feature correlation
    print("\nFeature Correlation Analysis")
    print("-" * 40)
    corr_matrix = val.correlation_analysis(X, features, output_dir)

    # ========================================================================
    # PART 5: CANOPY HEIGHT MAP GENERATION
    # ========================================================================
    print("\n" + "="*70)
    print("PART 5: CANOPY HEIGHT MAP GENERATION")
    print("="*70 + "\n")

    predict_map(
        model, s2_path, s1_path, topo_path,
        f'{output_dir}/canopy_height_map.tif'
    )

    # Visualize final map
    print("\nGenerating canopy height map visualizations...")
    viz.plot_canopy_height_map(
        f'{output_dir}/canopy_height_map.tif',
        gedi_csv=gedi_csv,
        title="Canopy Height Map - San Diego County",
        output_dir=output_dir
    )

    # Comparison map with GEDI validation
    print("\nGenerating validation comparison...")
    viz.plot_comparison_maps(
        f'{output_dir}/canopy_height_map.tif',
        gedi_csv=gedi_csv,
        output_dir=output_dir
    )

    # ========================================================================
    # PART 6: COMPREHENSIVE VALIDATION REPORT
    # ========================================================================
    print("\n" + "="*70)
    print("PART 6: COMPREHENSIVE VALIDATION REPORT")
    print("="*70 + "\n")

    # Generate comprehensive report
    val.generate_validation_report(
        gedi_csv, s2_path, s1_path, topo_path,
        model, features, output_dir
    )

    # ========================================================================
    # PART 7: SUMMARY PRODUCTS
    # ========================================================================
    print("\n" + "="*70)
    print("PART 7: SUMMARY PRODUCTS")
    print("="*70 + "\n")

    # Create summary statistics dictionary
    model_stats = {
        'r2': test_metrics['r2'],
        'rmse': test_metrics['rmse'],
        'mae': test_metrics['mae'],
        'cv_r2_mean': cv_summary['r2_mean'],
        'cv_r2_std': cv_summary['r2_std'],
        'cv_rmse_mean': cv_summary['rmse_mean'],
        'cv_rmse_std': cv_summary['rmse_std'],
        'n_samples': len(X),
        'n_features': len(features),
        'feature_importance': importance_df.to_dict('records')
    }

    # Pipeline summary figure
    viz.create_pipeline_summary(
        gedi_csv, s2_path, f'{output_dir}/canopy_height_map.tif',
        model_stats, output_dir
    )

    # Save model
    import joblib
    joblib.dump(model, f'{output_dir}/canopy_height_model.pkl')
    print("Saved trained model to full_outputs/canopy_height_model.pkl")

    # Save configuration
    config = {
        'bbox': bbox,
        'start_date': start_date,
        'end_date': end_date,
        'resolution': 30,
        'n_gedi_points': len(pd.read_csv(gedi_csv)),
        'features': features,
        'model_params': model.get_params()
    }

    import json
    with open(f'{output_dir}/configuration.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Saved configuration to full_outputs/configuration.json")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*70)
    print("FULL PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated Products:")
    print("\n  DATA FILES:")
    print("    - gedi_raw.csv")
    print("    - sentinel2.tif")
    print("    - sentinel1.tif (if available)")
    print("    - topography.tif")
    print("\n  MAPS:")
    print("    - canopy_height_map.tif")
    print("\n  VISUALIZATIONS:")
    print("    - gedi_data_coverage.png")
    print("    - gedi_tracks.png")
    print("    - sentinel2_bands.png")
    print("    - sentinel2_composites.png")
    print("    - model_validation.png")
    print("    - cross_validation.png")
    print("    - canopy_height_map.png")
    print("    - comparison_maps.png")
    print("    - pipeline_summary.png")
    print("    - feature_correlation_heatmap.png")
    print("\n  REPORTS:")
    print("    - validation_report.json")
    print("    - feature_importance.csv")
    print("    - feature_correlation.csv")
    print("    - configuration.json")
    print("\n  MODEL:")
    print("    - canopy_height_model.pkl")
    print("\nMODEL PERFORMANCE:")
    print(f"  Test R²:         {test_metrics['r2']:.4f}")
    print(f"  Test RMSE:       {test_metrics['rmse']:.2f} m")
    print(f"  Test MAE:        {test_metrics['mae']:.2f} m")
    print(f"  CV R²:           {cv_summary['r2_mean']:.4f} ± {cv_summary['r2_std']:.4f}")
    print(f"  CV RMSE:         {cv_summary['rmse_mean']:.2f} ± {cv_summary['rmse_std']:.2f} m")
    print("\nExecution time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print("="*70)

    # Print recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)

    if test_metrics['r2'] < 0.7:
        print("  ⚠ Model R² is below 0.7. Consider:")
        print("     - Increasing training data (larger area or longer time period)")
        print("     - Adding more features (e.g., phenology metrics)")
        print("     - Tuning hyperparameters")

    if test_metrics['rmse'] > 10:
        print("  ⚠ RMSE is above 10m. Consider:")
        print("     - Checking for data quality issues")
        print("     - Filtering outliers from GEDI data")
        print("     - Using ensemble methods")

    if abs(test_metrics['bias']) > 2:
        print("  ⚠ Model bias is significant. Consider:")
        print("     - Checking for systematic errors")
        print("     - Stratified sampling by height class")

    print("\n✓ Pipeline completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
