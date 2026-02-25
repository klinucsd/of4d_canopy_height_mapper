#!/usr/bin/env python3
"""
GEDI Canopy Height Map Worker Script (V2)
==========================================

Called by FastAPI service to run canopy height mapping pipeline with
user-configurable temporal parameters and additional visualizations.

Key differences from canopy_height.py:
- User-configurable temporal parameters for GEDI and Sentinel
- User-configurable cloud threshold for Sentinel-2
- User-configurable DEM dataset and ML algorithm
- Generates all visualizations (GEDI tracks, Sentinel histograms, pipeline summary)
- JSON-based metadata output instead of HTML report

Usage:
    gedi_canopy_height_map.py \\
        --bbox min_lon,min_lat,max_lon,max_lat \\
        --gedi-temporal-min 2024-01-01 \\
        --gedi-temporal-max 2025-12-31 \\
        --sentinel-temporal-min 2024-01-01 \\
        --sentinel-temporal-max 2025-12-31 \\
        --cloud-threshold 20 \\
        --dem-dataset COP30 \\
        --ml-algorithm RFR \\
        --resolution 30 \\
        --output-dir /path/to/output \\
        --gedi-dir gedi_usa_2024_2025

Exit codes:
    0: Success
    1: Processing error
    10: No GEDI data available
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import pipeline functions
from canopy_height_gedi_loader import load_gedi_for_bbox, get_gedi_stats
import gedi_canopy_height_pipeline_partitions as chp
from gedi_canopy_height_metadata_builder import build_metadata_from_job

# Import visualizations
import gedi_canopy_height_visualizations as viz

# Load environment variables for credentials
from dotenv import load_dotenv
load_dotenv()


def parse_bbox(bbox_str):
    """Parse bbox string 'min_lon,min_lat,max_lon,max_lat'"""
    try:
        return [float(x) for x in bbox_str.split(',')]
    except Exception as e:
        raise ValueError(f"Invalid bbox format: {bbox_str}. Expected: min_lon,min_lat,max_lon,max_lat")


def get_model(ml_algorithm="RFR", **kwargs):
    """
    Factory function to create ML model based on algorithm name.

    Parameters:
    -----------
    ml_algorithm : str
        Algorithm name (RFR, XGB, SVR, etc.)
    **kwargs : dict
        Algorithm-specific parameters

    Returns:
    --------
    model : sklearn estimator
        Configured ML model
    """
    from sklearn.ensemble import RandomForestRegressor

    algorithms = {
        "RFR": RandomForestRegressor,
        # Future algorithms can be added here:
        # "XGB": XGBRegressor,
        # "SVR": SVR,
    }

    if ml_algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {ml_algorithm}. Supported: {list(algorithms.keys())}")

    model_class = algorithms[ml_algorithm]

    if ml_algorithm == "RFR":
        return model_class(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 20),
            min_samples_split=kwargs.get('min_samples_split', 5),
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(f"Model configuration not implemented for: {ml_algorithm}")


def generate_all_visualizations(output_dir, gedi_csv, s2_tif, s1_tif, topo_tif,
                                height_map_tif, model, X_test, y_test, y_pred, features):
    """
    Generate all required visualization PNGs.

    Visualizations:
    - gedi_data_coverage.png
    - gedi_tracks.png
    - sentinel2_bands.png
    - sentinel2_composites.png
    - sentinel2_histograms.png
    - model_validation.png
    - pipeline_summary.png
    - canopy_height_map.png
    """
    print("  Generating visualizations...")

    # 1. GEDI Data Coverage
    try:
        viz.plot_gedi_data_coverage(gedi_csv, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ GEDI coverage: {e}")

    # 2. GEDI Tracks
    try:
        viz.plot_gedi_tracks(gedi_csv, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ GEDI tracks: {e}")

    # 3. Sentinel-2 Bands
    try:
        viz.plot_sentinel2_bands(s2_tif, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ S2 bands: {e}")

    # 4. Sentinel-2 Composites
    try:
        viz.plot_sentinel2_composite(s2_tif, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ S2 composites: {e}")

    # 5. Sentinel-2 Histograms
    try:
        viz.plot_band_histograms(s2_tif, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ S2 histograms: {e}")

    # 6. Model Validation
    try:
        feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        viz.plot_model_validation(y_test, y_pred, features, feature_importance,
                                  model_name="Random Forest", output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ Model validation: {e}")

    # 7. Pipeline Summary
    try:
        model_stats = {
            'r2': float(np.corrcoef(y_test, y_pred)[0, 1]**2),
            'rmse': float(np.sqrt(np.mean((y_test - y_pred)**2))),
            'mae': float(np.mean(np.abs(y_test - y_pred)))
        }
        viz.create_pipeline_summary(gedi_csv, s2_tif, height_map_tif, model_stats, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ Pipeline summary: {e}")

    # 8. Canopy Height Map (with legend)
    try:
        viz.plot_canopy_height_map(height_map_tif, output_dir=output_dir)
    except Exception as e:
        print(f"    ⚠ Canopy height map: {e}")

    print("  ✓ Visualizations complete")


def main():
    """Main worker function"""
    start_time = time.time()

    parser = argparse.ArgumentParser(description="GEDI Canopy Height Map Worker (V2)")

    # Bounding box
    parser.add_argument("--bbox", required=True,
                        help="Bounding box: min_lon,min_lat,max_lon,max_lat")

    # GEDI temporal parameters
    parser.add_argument("--gedi-temporal-min", default="2024-01-01",
                        help="GEDI start date (YYYY-MM-DD)")
    parser.add_argument("--gedi-temporal-max", default="2025-12-31",
                        help="GEDI end date (YYYY-MM-DD)")

    # Sentinel temporal parameters
    parser.add_argument("--sentinel-temporal-min", default="2024-01-01",
                        help="Sentinel start date (YYYY-MM-DD)")
    parser.add_argument("--sentinel-temporal-max", default="2025-12-31",
                        help="Sentinel end date (YYYY-MM-DD)")

    # Sentinel-2 cloud threshold
    parser.add_argument("--cloud-threshold", type=int, default=20,
                        help="Sentinel-2 max cloud cover percentage (0-100)")

    # DEM dataset
    parser.add_argument("--dem-dataset", default="COP30",
                        help="DEM dataset (COP30, SRTMGL1, etc.)")

    # ML algorithm
    parser.add_argument("--ml-algorithm", default="RFR",
                        help="ML algorithm (RFR, XGB, SVR)")

    # Output and resolution
    parser.add_argument("--resolution", type=int, default=30,
                        help="Resolution in meters")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--gedi-dir", default="gedi_usa_2024_2025",
                        help="GEDI partition directory")

    # Optional parameters
    parser.add_argument("--use-sentinel1", action="store_true", default=True,
                        help="Include Sentinel-1 data")
    parser.add_argument("--no-sentinel1", action="store_false", dest="use_sentinel1",
                        help="Exclude Sentinel-1 data")
    parser.add_argument("--job-id", default=None,
                        help="Job ID for metadata")
    parser.add_argument("--output-url", default=None,
                        help="Output URL for metadata")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse bbox
    try:
        bbox = parse_bbox(args.bbox)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate job_id if not provided
    if args.job_id is None:
        from datetime import datetime
        args.job_id = f"CanopyHeightService_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("="*60)
    print("GEDI CANOPY HEIGHT MAP WORKER (V2)")
    print("="*60)
    print(f"  Job ID: {args.job_id}")
    print(f"  BBox: {bbox}")
    print(f"  GEDI Range: {args.gedi_temporal_min} to {args.gedi_temporal_max}")
    print(f"  Sentinel Range: {args.sentinel_temporal_min} to {args.sentinel_temporal_max}")
    print(f"  Cloud Threshold: {args.cloud_threshold}%")
    print(f"  DEM Dataset: {args.dem_dataset}")
    print(f"  ML Algorithm: {args.ml_algorithm}")
    print(f"  Resolution: {args.resolution}m")
    print(f"  Output Directory: {args.output_dir}")
    print("="*60)

    # Collect request parameters for metadata
    request_params = {
        "bbox": bbox,
        "gedi_temporal_min": args.gedi_temporal_min,
        "gedi_temporal_max": args.gedi_temporal_max,
        "sentinel_temporal_min": args.sentinel_temporal_min,
        "sentinel_temporal_max": args.sentinel_temporal_max,
        "cloud_threshold": args.cloud_threshold,
        "dem_dataset": args.dem_dataset,
        "ml_algorithm": args.ml_algorithm,
        "resolution": args.resolution
    }

    # ========================================================================
    # STEP 1: Load GEDI data from partitions
    # ========================================================================
    print("\n[1/5] Loading GEDI data from partitions...")

    gedi_csv = os.path.join(args.output_dir, "gedi.csv")

    try:
        gedi_csv = chp.load_gedi_from_partitions(
            bbox,
            gedi_csv,
            gedi_dir=args.gedi_dir
        )
    except Exception as e:
        print(f"Error loading GEDI data: {e}", file=sys.stderr)
        return 1

    if gedi_csv is None:
        print("Error: No GEDI data available for this bounding box", file=sys.stderr)
        return 10

    df = pd.read_csv(gedi_csv)
    print(f"✓ Loaded {len(df)} GEDI points")

    if len(df) < 100:
        print(f"Warning: Only {len(df)} GEDI points available. Model quality may be poor.")

    # ========================================================================
    # STEP 2: Download satellite data
    # ========================================================================
    print(f"\n[2/5] Downloading satellite data ({args.sentinel_temporal_min} to {args.sentinel_temporal_max})...")

    # Sentinel-2 (with user-specified cloud threshold)
    s2_path = os.path.join(args.output_dir, "sentinel2_optical_composite.tif")
    s2_scenes_used = 0
    try:
        s2_path = chp.download_sentinel2_mpc(
            bbox, args.sentinel_temporal_min, args.sentinel_temporal_max,
            s2_path,
            max_items=10,
            resolution=args.resolution,
            n_workers=4,
            cloud_threshold=args.cloud_threshold
        )
        s2_scenes_used = 10  # Approximate count
    except Exception as e:
        print(f"Error downloading Sentinel-2: {e}", file=sys.stderr)
        return 1

    # Sentinel-1 (optional)
    s1_path = None
    if args.use_sentinel1:
        print("\nDownloading Sentinel-1...")
        s1_path = os.path.join(args.output_dir, "sentinel1_sar_composite.tif")
        try:
            s1_path = chp.download_sentinel1_mpc(
                bbox, args.sentinel_temporal_min, args.sentinel_temporal_max,
                s1_path,
                max_items=5
            )
            if s1_path is None:
                print("  Note: Sentinel-1 not available, continuing without it")
                s1_path = None
        except Exception as e:
            print(f"  Warning: Sentinel-1 download failed: {e}")
            s1_path = None

    # Topography (with user-specified DEM type)
    print(f"\nDownloading {args.dem_dataset} topography...")
    topo_path = os.path.join(args.output_dir, "topography.tif")
    try:
        topo_path = chp.download_srtm_opentopography(bbox, topo_path, dem_type=args.dem_dataset)
    except Exception as e:
        print(f"Error downloading topography: {e}", file=sys.stderr)
        return 1

    # ========================================================================
    # STEP 3: Train model
    # ========================================================================
    print("\n[3/5] Training model...")

    try:
        X, y, features = chp.extract_features(gedi_csv, s2_path, s1_path, topo_path)
        print(f"✓ Extracted {len(X)} samples with {len(features)} features")
    except Exception as e:
        print(f"Error extracting features: {e}", file=sys.stderr)
        return 1

    if len(X) < 50:
        print(f"Error: Too few samples ({len(X)}) for training", file=sys.stderr)
        return 1

    # Train model using factory function
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        model = get_model(ml_algorithm=args.ml_algorithm)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✓ Model Performance ({args.ml_algorithm}):")
    print(f"    R²:  {r2:.4f}")
    print(f"    RMSE: {rmse:.2f} m")
    print(f"    MAE:  {mae:.2f} m")

    # Save model
    import joblib
    model_path = os.path.join(args.output_dir, "canopy_height_model.pkl")
    joblib.dump(model, model_path)
    print(f"✓ Saved model to {model_path}")

    # Save metrics
    metrics = {
        "r2": float(r2),
        "rmse_m": float(rmse),
        "mae_m": float(mae),
        "n_samples": len(X),
        "n_features": len(features)
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")

    # ========================================================================
    # STEP 4: Generate canopy height map
    # ========================================================================
    print("\n[4/5] Generating canopy height map...")

    output_tif = os.path.join(args.output_dir, "canopy_height_map.tif")
    try:
        chp.predict_map(model, s2_path, s1_path, topo_path, output_tif)
    except Exception as e:
        print(f"Error generating height map: {e}", file=sys.stderr)
        return 1

    # ========================================================================
    # STEP 5: Generate visualizations and metadata
    # ========================================================================
    print("\n[5/5] Generating visualizations and metadata...")

    # Generate all visualizations
    try:
        generate_all_visualizations(
            args.output_dir, gedi_csv, s2_path, s1_path, topo_path,
            output_tif, model, X_test, y_test, y_pred, features
        )
    except Exception as e:
        print(f"    ⚠ Some visualizations failed: {e}")

    # Build metadata JSON
    try:
        elapsed = time.time() - start_time
        execution_minutes = elapsed / 60

        # Generate output_url if not provided
        if args.output_url is None:
            args.output_url = f"file://{args.output_dir}"

        build_metadata_from_job(
            output_dir=args.output_dir,
            job_id=args.job_id,
            output_url=args.output_url,
            request_params=request_params,
            gedi_csv=gedi_csv,
            gedi_dir=args.gedi_dir,
            gedi_temporal_min=args.gedi_temporal_min,
            gedi_temporal_max=args.gedi_temporal_max,
            sentinel_temporal_min=args.sentinel_temporal_min,
            sentinel_temporal_max=args.sentinel_temporal_max,
            cloud_threshold=args.cloud_threshold,
            s1_available=(s1_path is not None),
            dem_dataset=args.dem_dataset,
            model=model,
            X_train=X_train,
            y_test=y_test,
            y_pred=y_pred,
            features=features,
            ml_algorithm=args.ml_algorithm,
            execution_time_minutes=execution_minutes
        )
        print(f"✓ Saved metadata to {args.output_dir}/metadata.json")
    except Exception as e:
        print(f"    ⚠ Metadata generation failed: {e}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)

    print("\n" + "="*60)
    print("✓ CANOPY HEIGHT MAPPING COMPLETE")
    print("="*60)
    print(f"  Execution time: {int(minutes)}m {seconds:.0f}s")
    print(f"  Output files:")
    print(f"    Data:")
    print(f"      - gedi.csv")
    print(f"      - sentinel2_optical_composite.tif")
    if s1_path:
        print(f"      - sentinel1_sar_composite.tif")
    print(f"      - topography.tif")
    print(f"    Model:")
    print(f"      - canopy_height_model.pkl")
    print(f"      - metrics.json")
    print(f"    Output:")
    print(f"      - canopy_height_map.tif")
    print(f"    Visualizations:")
    print(f"      - gedi_data_coverage.png")
    print(f"      - gedi_tracks.png")
    print(f"      - sentinel2_bands.png")
    print(f"      - sentinel2_composites.png")
    print(f"      - sentinel2_histograms.png")
    print(f"      - model_validation.png")
    print(f"      - pipeline_summary.png")
    print(f"      - canopy_height_map.png")
    print(f"    Metadata:")
    print(f"      - metadata.json")
    print("="*60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
