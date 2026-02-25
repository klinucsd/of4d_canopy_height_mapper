"""
Metadata Builder for GEDI Canopy Height Map Service

This module builds structured JSON metadata for the canopy height mapping
pipeline, including all data sources, model performance, and output files.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


class MetadataBuilder:
    """Builder for creating structured metadata JSON"""

    def __init__(self, job_id: str, output_url: str):
        """
        Initialize metadata builder.

        Parameters:
        -----------
        job_id : str
            Unique job identifier
        output_url : str
            URL to access output files
        """
        self.job_id = job_id
        self.output_url = output_url
        self.metadata = {
            "job_id": job_id,
            "status": "success",
            "output_url": output_url,
            "canopy_height_map": {},
            "data_sources": {
                "gedi": {},
                "sentinel": {},
                "topography": {}
            },
            "model_training": {},
            "prediction_results": {},
            "execution_time_minutes": 0
        }

    def set_status(self, status: str, error: Optional[str] = None):
        """Set job status and optional error message"""
        self.metadata["status"] = status
        if error:
            self.metadata["error"] = error

    def set_execution_time(self, minutes: float):
        """Set execution time in minutes"""
        self.metadata["execution_time_minutes"] = round(minutes, 2)

    def add_canopy_height_info(self, output_dir: str):
        """Add canopy height map information"""
        height_tif = os.path.join(output_dir, "canopy_height_map.tif")

        if os.path.exists(height_tif):
            try:
                import rasterio
                with rasterio.open(height_tif) as src:
                    data = src.read(1)
                    mean_h = float(np.nanmean(data))
                    max_h = float(np.nanmax(data))
                    median_h = float(np.nanmedian(data))
                    std_h = float(np.nanstd(data))
                    min_h = float(np.nanmin(data))

                file_size_mb = os.path.getsize(height_tif) / (1024 ** 2)

                self.metadata["canopy_height_map"] = {
                    "image": "canopy_height_map.png",
                    "download": "canopy_height_map.tif",
                    "mean_height": f"{mean_h:.1f} m",
                    "max_height": f"{max_h:.1f} m",
                    "median_height": f"{median_h:.1f} m",
                    "std_deviation": f"{std_h:.1f} m",
                    "min_height": f"{min_h:.1f} m",
                    "file_size_mb": f"{file_size_mb:.1f}"
                }
            except Exception as e:
                self.metadata["canopy_height_map"] = {
                    "image": "canopy_height_map.png",
                    "download": "canopy_height_map.tif",
                    "error": f"Could not read statistics: {str(e)}"
                }
        else:
            self.metadata["canopy_height_map"] = {
                "image": "canopy_height_map.png",
                "download": "canopy_height_map.tif"
            }

    def add_gedi_metadata(self, gedi_csv: str, gedi_dir: str, gedi_temporal_min: str, gedi_temporal_max: str):
        """Add GEDI data source metadata"""
        gedi_info = {
            "images": ["gedi_data_coverage.png", "gedi_tracks.png"],
            "download": "gedi.csv",
            "metadata": {
                "source": "Pre-partitioned Parquet files",
                "gedi_dir": gedi_dir,
                "temporal_range": f"{gedi_temporal_min} to {gedi_temporal_max}"
            }
        }

        if os.path.exists(gedi_csv):
            try:
                df = pd.read_csv(gedi_csv)
                gedi_info["metadata"]["valid_points_extracted"] = len(df)
                gedi_info["metadata"]["output_csv_size_mb"] = f"{os.path.getsize(gedi_csv) / (1024 ** 2):.1f}"
                gedi_info["metadata"]["mean_height_m"] = f"{df['rh98'].mean():.2f}"
                gedi_info["metadata"]["max_height_m"] = f"{df['rh98'].max():.2f}"
            except Exception as e:
                gedi_info["metadata"]["error"] = str(e)

        self.metadata["data_sources"]["gedi"] = gedi_info

    def add_sentinel_metadata(self, output_dir: str, sentinel_temporal_min: str, sentinel_temporal_max: str,
                              cloud_threshold: int, s1_available: bool, scenes_used: int = 0):
        """Add Sentinel data source metadata"""
        # Detect which bands are available by checking the actual file
        s2_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI"]  # Default to new 10-band version
        s2_tif = os.path.join(output_dir, "sentinel2_optical_composite.tif")
        if os.path.exists(s2_tif):
            import rasterio
            with rasterio.open(s2_tif) as src:
                # If we have 5 bands, it's the old 4-band version + NDVI
                if src.count <= 5:
                    s2_bands = ["B04", "B08", "B11", "B12", "NDVI"]

        sentinel_info = {
            "images": ["sentinel2_bands.png", "sentinel2_composites.png", "sentinel2_histograms.png"],
            "downloads": ["sentinel2_optical_composite.tif"],
            "sentinel2": {
                "source": "Microsoft Planetary Computer",
                "scenes_used": scenes_used,
                "bands_used": s2_bands,
                "cloud_threshold": cloud_threshold,
                "temporal_range": f"{sentinel_temporal_min} to {sentinel_temporal_max}"
            }
        }

        # Add Sentinel-2 file size
        s2_tif = os.path.join(output_dir, "sentinel2_optical_composite.tif")
        if os.path.exists(s2_tif):
            sentinel_info["sentinel2"]["output_size_mb"] = f"{os.path.getsize(s2_tif) / (1024 ** 2):.1f}"

        # Add Sentinel-1 info if available
        if s1_available:
            s1_tif = os.path.join(output_dir, "sentinel1_sar_composite.tif")
            sentinel_info["downloads"].append("sentinel1_sar_composite.tif")
            sentinel_info["sentinel1"] = {
                "source": "Microsoft Planetary Computer",
                "collection": "Sentinel-1 RTC/GRD",
                "polarizations": ["VV", "VH"]
            }
            if os.path.exists(s1_tif):
                sentinel_info["sentinel1"]["output_size_mb"] = f"{os.path.getsize(s1_tif) / (1024 ** 2):.1f}"

        self.metadata["data_sources"]["sentinel"] = sentinel_info

    def add_topography_metadata(self, output_dir: str, dem_dataset: str = "COP30"):
        """Add topography data source metadata"""
        # Detect which bands are available by checking the actual file
        topo_bands = ["Elevation", "Slope", "Aspect"]  # Default to new 3-band version
        topo_tif = os.path.join(output_dir, "topography.tif")
        if os.path.exists(topo_tif):
            import rasterio
            with rasterio.open(topo_tif) as src:
                # If we have 2 bands, it's the old version
                if src.count <= 2:
                    topo_bands = ["Elevation", "Slope"]

        topo_info = {
            "download": "topography.tif",
            "metadata": {
                "source": "OpenTopography",
                "dataset": dem_dataset,
                "bands": topo_bands,
                "resolution": "30m"
            }
        }

        if os.path.exists(topo_tif):
            topo_info["metadata"]["output_size_mb"] = f"{os.path.getsize(topo_tif) / (1024 ** 2):.1f}"

        self.metadata["data_sources"]["topography"] = topo_info

    def add_model_training_metadata(self, model, X_train, y_test, y_pred, features,
                                    ml_algorithm: str = "RFR", test_split: float = 0.2):
        """Add model training metadata"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        model_info = {
            "images": ["model_validation.png", "pipeline_summary.png"],
            "algorithm": ml_algorithm,
            "performance": {
                "r2": round(float(r2), 4),
                "rmse_m": round(float(rmse), 2),
                "mae_m": round(float(mae), 2),
                "training_samples": len(X_train),
                "test_split": test_split
            }
        }

        # Add model-specific parameters
        if ml_algorithm == "RFR":
            model_info["parameters"] = {
                "n_estimators": int(model.n_estimators),
                "max_depth": int(model.max_depth),
                "min_samples_split": int(model.min_samples_split)
            }

        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = {
                features[i]: float(importances[i])
                for i in range(len(features))
            }
            # Sort by importance and take top 10
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            model_info["top_features"] = [
                {"name": name, "importance": round(imp, 4)}
                for name, imp in sorted_features
            ]

        self.metadata["model_training"] = model_info

    def add_prediction_results(self, output_dir: str):
        """Add prediction results metadata"""
        height_tif = os.path.join(output_dir, "canopy_height_map.tif")
        results = {
            "output_files": ["canopy_height_map.tif", "canopy_height_map.png"],
            "coverage": "Complete (wall-to-wall)",
            "statistics": {}
        }

        if os.path.exists(height_tif):
            try:
                import rasterio
                with rasterio.open(height_tif) as src:
                    data = src.read(1)
                    results["statistics"] = {
                        "mean_height_m": round(float(np.nanmean(data)), 2),
                        "median_height_m": round(float(np.nanmedian(data)), 2),
                        "std_deviation_m": round(float(np.nanstd(data)), 2),
                        "min_height_m": round(float(np.nanmin(data)), 2),
                        "max_height_m": round(float(np.nanmax(data)), 2),
                        "valid_pixels": int(np.sum(np.isfinite(data))),
                        "total_pixels": int(data.size)
                    }
            except Exception as e:
                results["statistics"]["error"] = str(e)

        self.metadata["prediction_results"] = results

    def add_input_parameters(self, request_params: Dict[str, Any]):
        """Add input parameters to metadata"""
        self.metadata["input_parameters"] = request_params

    def build(self) -> Dict[str, Any]:
        """Return the complete metadata dictionary"""
        return self.metadata

    def save(self, output_dir: str):
        """Save metadata to JSON file"""
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        return metadata_path


def build_metadata_from_job(output_dir: str, job_id: str, output_url: str,
                            request_params: Dict[str, Any],
                            gedi_csv: str, gedi_dir: str,
                            gedi_temporal_min: str, gedi_temporal_max: str,
                            sentinel_temporal_min: str, sentinel_temporal_max: str,
                            cloud_threshold: int, s1_available: bool,
                            dem_dataset: str, model, X_train, y_test, y_pred,
                            features, ml_algorithm: str, execution_time_minutes: float) -> str:
    """
    Convenience function to build complete metadata from job parameters.

    Returns:
    --------
    str
        Path to saved metadata.json file
    """
    builder = MetadataBuilder(job_id, output_url)

    # Add all metadata sections
    builder.add_input_parameters(request_params)
    builder.add_canopy_height_info(output_dir)
    builder.add_gedi_metadata(gedi_csv, gedi_dir, gedi_temporal_min, gedi_temporal_max)
    builder.add_sentinel_metadata(output_dir, sentinel_temporal_min, sentinel_temporal_max,
                                   cloud_threshold, s1_available)
    builder.add_topography_metadata(output_dir, dem_dataset)
    builder.add_model_training_metadata(model, X_train, y_test, y_pred, features, ml_algorithm)
    builder.add_prediction_results(output_dir)
    builder.set_execution_time(execution_time_minutes)

    return builder.save(output_dir)
