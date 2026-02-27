"""
GEDI Canopy Height Map FastAPI Service (V2)
============================================

FastAPI endpoint for generating canopy height maps with user-configurable
temporal parameters, additional visualizations, and JSON metadata response.

Key differences from canopy_height_service.py:
- User-configurable temporal parameters (GEDI and Sentinel date ranges)
- User-configurable cloud threshold for Sentinel-2
- User-configurable DEM dataset and ML algorithm
- Additional visualizations (GEDI tracks, Sentinel histograms, pipeline summary)
- JSON-based metadata output (no HTML report)
- Structured metadata matching report sections

Job submission pattern:
1. Accepts bbox and user parameters
2. Submits job to background executor
3. Returns job ID and status URL immediately
4. Worker script runs asynchronously
5. User polls status_url for results (includes metadata)

Author: Canopy Height Team
Date: 2025-02-23
"""

import os
import json
import logging
import subprocess
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

from database import create_job_entry, update_job_status
import utils

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Private"]
)

# Configuration from environment
PYTHON_ENV_PATH = os.path.join(os.getenv("FASTAPI_HOME"), "fastapi_env")
PYTHON_EXECUTABLE = os.path.join(PYTHON_ENV_PATH, "bin", "python")

SECRET_TOKEN = os.getenv("SECRET_TOKEN")
SCRIPTS_DIR = os.path.join(os.getenv("FASTAPI_HOME"), "scripts")
MAX_MEM_MB = int(os.getenv("MAX_MEM_MB", "8192"))

# GEDI data directory
DEFAULT_GEDI_DIR = os.getenv("GEDI_DATA_DIR", "gedi_usa_2024_2025")

# BBox size limits (in km)
MIN_BBOX_SIZE_KM = 30  # 30km × 30km minimum
MAX_BBOX_SIZE_KM = 50  # 50km × 50km maximum

# Environment for subprocess
ENVIRONMENT = {
    "PATH": f"{os.path.join(PYTHON_ENV_PATH, 'bin')}:/usr/bin:/bin",
    "LD_LIBRARY_PATH": f"{os.path.join(PYTHON_ENV_PATH, 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"
}

security = HTTPBearer()


class BBoxModel(BaseModel):
    """Bounding box model"""
    min_lon: float = Field(..., description="Minimum longitude", ge=-180, le=180)
    min_lat: float = Field(..., description="Minimum latitude", ge=-90, le=90)
    max_lon: float = Field(..., description="Maximum longitude", ge=-180, le=180)
    max_lat: float = Field(..., description="Maximum latitude", ge=-90, le=90)

    @validator('max_lon')
    def validate_longitude(cls, v, values):
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon must be greater than min_lon')
        return v

    @validator('max_lat')
    def validate_latitude(cls, v, values):
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat must be greater than min_lat')
        return v

    def get_size_km(self) -> tuple:
        """Calculate bbox size in km (width, height)"""
        import math
        # Approximate conversion: 1 degree lat ≈ 111 km
        # Longitude varies with latitude
        center_lat = (self.min_lat + self.max_lat) / 2
        lat_km = (self.max_lat - self.min_lat) * 111
        lon_km = (self.max_lon - self.min_lon) * 111 * math.cos(math.radians(center_lat))
        return lon_km, lat_km


class GediCanopyHeightRequest(BaseModel):
    """
    GEDI Canopy Height Mapping Request (V2)

    User-facing parameters with configurable temporal ranges and options.
    """
    # Bounding box
    min_lon: float = Field(..., description="Minimum longitude", ge=-180, le=180)
    min_lat: float = Field(..., description="Minimum latitude", ge=-90, le=90)
    max_lon: float = Field(..., description="Maximum longitude", ge=-180, le=180)
    max_lat: float = Field(..., description="Maximum latitude", ge=-90, le=90)

    # GEDI temporal parameters
    gedi_temporal_min: str = Field(
        default="2024-01-01",
        description="GEDI start date (YYYY-MM-DD)"
    )
    gedi_temporal_max: str = Field(
        default="2025-12-31",
        description="GEDI end date (YYYY-MM-DD)"
    )

    # Sentinel temporal parameters
    sentinel_temporal_min: str = Field(
        default="2024-01-01",
        description="Sentinel start date (YYYY-MM-DD)"
    )
    sentinel_temporal_max: str = Field(
        default="2025-12-31",
        description="Sentinel end date (YYYY-MM-DD)"
    )

    # Sentinel-2 cloud threshold
    sentinel2_l2a_cloud_threshold: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Max cloud cover percentage for Sentinel-2 (0-100)"
    )

    # DEM dataset
    DEM_dataset: str = Field(
        default="COP30",
        description="DEM dataset (COP30, SRTMGL1, etc.)"
    )

    # ML algorithm
    ML_algorithm: str = Field(
        default="RFR",
        description="ML algorithm: RFR (Random Forest), extensible for XGBoost, etc."
    )

    # Resolution
    resolution: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Output resolution in meters"
    )

    @validator('ML_algorithm')
    def validate_algorithm(cls, v):
        """Validate ML algorithm is supported"""
        supported = ["RFR"]  # Add more as they are implemented
        if v not in supported:
            raise ValueError(f'Unsupported algorithm: {v}. Supported: {", ".join(supported)}')
        return v

    @validator('max_lon')
    def validate_longitude(cls, v, values):
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon must be greater than min_lon')
        return v

    @validator('max_lat')
    def validate_latitude(cls, v, values):
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat must be greater than min_lat')
        return v

    def get_size_km(self) -> tuple:
        """Calculate bbox size in km (width, height)"""
        import math
        center_lat = (self.min_lat + self.max_lat) / 2
        lat_km = (self.max_lat - self.min_lat) * 111
        lon_km = (self.max_lon - self.min_lon) * 111 * math.cos(math.radians(center_lat))
        return lon_km, lat_km


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token"""
    if credentials.credentials != SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials


@router.post(
    "/v1/GEDI_canopy_height_map",
    operation_id="gedi_canopy_height_mapping_v1",
    include_in_schema=True,
    summary="Generate GEDI Canopy Height Map (V1)",
    description="""
Generate wall-to-wall canopy height predictions using GEDI L2A lidar data
and Sentinel satellite imagery with user-configurable parameters.

**NEW in V1:**
- Latitude/longitude coordinates added as input features (improves regional accuracy)
- 10 Sentinel-2 bands (full multispectral)
- 3 DEM bands (elevation, slope, aspect)
- Uses global GEDI partitions (gedi_global_2024_2025/)

**Parameters:**
- **Bounding box**: Area of interest (min_lon, min_lat, max_lon, max_lat)
- **GEDI dates**: Temporal range for GEDI data (default: 2024-2025)
- **Sentinel dates**: Temporal range for Sentinel imagery (default: 2024-2025)
- **Cloud threshold**: Max cloud cover for Sentinel-2 (0-100%, default: 20%)
- **DEM dataset**: Digital elevation model (default: COP30)
- **ML algorithm**: Machine learning algorithm (default: RFR)
- **Resolution**: Output resolution in meters (default: 30)

**Output:**
- Returns immediately with job ID and status URL
- Poll status_url for results and metadata
- Processing typically takes 5-15 minutes depending on area

**Response includes:**
- Canopy height map (GeoTIFF + PNG visualization)
- All data source files (GEDI CSV, Sentinel TIFFs, topography)
- 8 visualization PNGs (GEDI coverage/tracks, Sentinel bands/composites/histograms, model validation, pipeline summary)
- Structured JSON metadata with all statistics
    """
)
async def execute_gedi_canopy_height(
    request_data: GediCanopyHeightRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Submit a GEDI canopy height mapping job with user-configurable parameters.

    The job runs asynchronously and results are available at the status_url.
    The status endpoint returns structured metadata with all results.
    """
    import utils
    import os

    cilogon_id = ""

    # Calculate bbox size for logging
    width_km, height_km = request_data.get_size_km()
    bbox_list = [
        request_data.min_lon,
        request_data.min_lat,
        request_data.max_lon,
        request_data.max_lat
    ]

    logger.info(
        f"GEDI canopy height request: "
        f"bbox={bbox_list}, "
        f"resolution={request_data.resolution}m, "
        f"size={width_km:.1f}km×{height_km:.1f}km, "
        f"gedi_range={request_data.gedi_temporal_min} to {request_data.gedi_temporal_max}, "
        f"sentinel_range={request_data.sentinel_temporal_min} to {request_data.sentinel_temporal_max}, "
        f"cloud_threshold={request_data.sentinel2_l2a_cloud_threshold}%, "
        f"dem={request_data.DEM_dataset}, "
        f"algorithm={request_data.ML_algorithm}"
    )

    # Generate service ID and working directory
    service_name = "CanopyHeightService"
    service_id = utils.generate_unique_serviceID(service_name)
    working_directory = utils.create_working_directory(service_id)
    os.makedirs(working_directory, exist_ok=True)

    # Request parameters
    request_params = {
        "bbox": bbox_list,
        "resolution": request_data.resolution,
        "gedi_temporal_min": request_data.gedi_temporal_min,
        "gedi_temporal_max": request_data.gedi_temporal_max,
        "sentinel_temporal_min": request_data.sentinel_temporal_min,
        "sentinel_temporal_max": request_data.sentinel_temporal_max,
        "cloud_threshold": request_data.sentinel2_l2a_cloud_threshold,
        "dem_dataset": request_data.DEM_dataset,
        "ml_algorithm": request_data.ML_algorithm,
        "bbox_size_km": {"width": round(width_km, 2), "height": round(height_km, 2)}
    }
    client_ip = request.client.host

    # Create database job entry
    db_job = create_job_entry(service_name, service_id, request_params, client_ip, cilogon_id=cilogon_id)
    if db_job is None:
        raise HTTPException(status_code=500, detail="Failed to create job entry in the database")

    # Calculate output_url and update immediately
    host_url = utils.get_full_hostname()
    output_url = f"{host_url}/output/{service_id}"
    update_job_status(db_job.id, status="Pending", output_url=output_url)

    # Build the command for the worker script
    script_path = os.path.join(SCRIPTS_DIR, "gedi_canopy_height_map.py")

    # Verify worker script exists
    if not os.path.exists(script_path):
        logger.error(f"Worker script not found: {script_path}")
        raise HTTPException(status_code=500, detail=f"Worker script not found at {script_path}")

    bbox_str = f"{bbox_list[0]},{bbox_list[1]},{bbox_list[2]},{bbox_list[3]}"
    command = [
        PYTHON_EXECUTABLE,
        script_path,
        "--bbox=" + bbox_str,  # Use = format because bbox starts with negative number
        "--gedi-temporal-min", request_data.gedi_temporal_min,
        "--gedi-temporal-max", request_data.gedi_temporal_max,
        "--sentinel-temporal-min", request_data.sentinel_temporal_min,
        "--sentinel-temporal-max", request_data.sentinel_temporal_max,
        "--cloud-threshold", str(request_data.sentinel2_l2a_cloud_threshold),
        "--dem-dataset", request_data.DEM_dataset,
        "--ml-algorithm", request_data.ML_algorithm,
        "--output-dir", working_directory,
        "--resolution", str(request_data.resolution),
        "--gedi-dir", DEFAULT_GEDI_DIR,
        "--job-id", service_id
    ]

    # Log the command for debugging
    logger.info(f"Command: {command}")

    # Submit the job to the executor
    def run_job_with_metadata(db_job_id, command, working_directory, service_id, request_params):
        """Run the canopy height job and update status"""
        import signal
        from utils import update_job_status, get_full_hostname

        def _tail_file(path, max_bytes=8192):
            try:
                with open(path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    if size <= max_bytes:
                        f.seek(0)
                        data = f.read()
                    else:
                        f.seek(-max_bytes, os.SEEK_END)
                        data = f.read()
                return data.decode("utf-8", errors="replace")
            except Exception as e:
                logger.error(f"Failed to read tail of {path}: {e}")
                return ""

        def _interpret_return_code(rc: int) -> str:
            if rc is None:
                return "Process ended without a return code"
            if rc < 0:
                sig = -rc
                try:
                    sig_name = signal.Signals(sig).name
                except Exception:
                    sig_name = f"SIG{sig}"
                if sig == signal.SIGKILL:
                    return "Terminated by SIGKILL (likely OOM)"
                if sig == signal.SIGTERM:
                    return "Terminated by SIGTERM"
                if sig == signal.SIGSEGV:
                    return "Segmentation fault"
                return f"Terminated by signal {sig_name}"
            return f"Exit status {rc}"

        host_url = get_full_hostname()
        output_url = f"{host_url}/output/{service_id}"

        # Load metadata if it was generated
        metadata_path = os.path.join(working_directory, "metadata.json")
        metadata = {
            "job_id": service_id,
            "output_url": output_url,
            "input_parameters": {
                "bbox": request_params["bbox"],
                "resolution": request_params["resolution"],
                "bbox_size_km": request_params["bbox_size_km"],
                "gedi_temporal_min": request_params.get("gedi_temporal_min"),
                "gedi_temporal_max": request_params.get("gedi_temporal_max"),
                "sentinel_temporal_min": request_params.get("sentinel_temporal_min"),
                "sentinel_temporal_max": request_params.get("sentinel_temporal_max"),
                "cloud_threshold": request_params.get("cloud_threshold"),
                "dem_dataset": request_params.get("dem_dataset"),
                "ml_algorithm": request_params.get("ml_algorithm")
            }
        }

        stdout_path = os.path.join(working_directory, "stdout.txt")
        stderr_path = os.path.join(working_directory, "stderr.txt")

        process = None
        timeout_hit = False
        max_execution_time = 3600  # 1 hour max

        try:
            with open(stdout_path, "w") as _out, open(stderr_path, "w") as _err:
                process = subprocess.Popen(
                    command,
                    cwd=working_directory,
                    env={**os.environ, **ENVIRONMENT},
                    stdout=_out,
                    stderr=_err
                )
                logger.info(f"Job {db_job_id}: process started with PID {process.pid}")

                try:
                    process.wait(timeout=max_execution_time)
                except subprocess.TimeoutExpired:
                    timeout_hit = True
                    logger.warning(f"Job {db_job_id}: timeout after {max_execution_time}s")
                    try:
                        process.terminate()
                        process.wait(timeout=10)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                        finally:
                            try:
                                process.wait(timeout=5)
                            except Exception:
                                pass

            rc = process.returncode if process else None

            if rc == 0 and not timeout_hit:
                # Check if output files exist
                output_files = []
                expected_files = [
                    "canopy_height_map.tif",
                    "canopy_height_model.pkl",
                    "metadata.json",
                    "gedi_data_coverage.png",
                    "gedi_tracks.png",
                    "sentinel2_bands.png",
                    "sentinel2_composites.png",
                    "sentinel2_histograms.png",
                    "model_validation.png",
                    "pipeline_summary.png",
                    "canopy_height_map.png"
                ]
                for fname in expected_files:
                    fpath = os.path.join(working_directory, fname)
                    if os.path.exists(fpath):
                        output_files.append(f"{output_url}/{fname}")

                if output_files:
                    update_job_status(db_job_id, "Completed", output_url=output_url, response_code=200)

                    # Load generated metadata
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path) as f:
                                generated_metadata = json.load(f)
                                metadata.update(generated_metadata)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata: {e}")
                    else:
                        metadata["status"] = "success"
                        metadata["output_files"] = output_files

                    logger.info(f"Job {db_job_id} completed successfully")
                else:
                    update_job_status(db_job_id, "Failed", output_url=output_url,
                                     error="No output files generated", response_code=500)
                    metadata["status"] = "error"
                    metadata["error"] = "No output files generated"
                    logger.error(f"Job {db_job_id} failed: no output files")

            elif rc == 10:
                # No GEDI data for this bbox
                update_job_status(db_job_id, "Failed", output_url=output_url,
                                 error="No GEDI data available for this bounding box. The area may be outside GEDI coverage.", response_code=400)
                metadata["status"] = "no_data"
                metadata["error"] = "No GEDI data available for this bounding box"
                logger.info(f"Job {db_job_id} failed: no GEDI data")

            else:
                if timeout_hit:
                    error_msg = f"Timeout exceeded after {max_execution_time} seconds"
                    response_code = 408
                else:
                    error_msg = _interpret_return_code(rc)
                    response_code = rc if isinstance(rc, int) and rc >= 0 else 500
                    detail = _tail_file(stderr_path)

                final_error = f"{error_msg}. Detail: {detail}" if detail else error_msg

                if detail:
                    logger.error(f"Job {db_job_id} failed. Detail (tail):\n{detail}")

                update_job_status(db_job_id, "Failed", output_url=output_url,
                                 error=final_error, response_code=response_code)
                metadata["status"] = "error"
                metadata["error"] = final_error
                logger.info(f"Job {db_job_id} failed with code {rc}")

        except FileNotFoundError as e:
            error_msg = f"Execution failed: {e}"
            update_job_status(db_job_id, "Failed", output_url=output_url, error=error_msg, response_code=500)
            metadata["status"] = "error"
            metadata["error"] = error_msg
            logger.error(f"Job {db_job_id} failed to start: {e}")

        except Exception as e:
            error_msg = str(e)
            update_job_status(db_job_id, "Failed", output_url=output_url, error=error_msg, response_code=500)
            metadata["status"] = "error"
            metadata["error"] = error_msg
            logger.error(f"Job {db_job_id} failed with exception: {error_msg}")

        finally:
            # Ensure metadata is saved even if job failed
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save metadata.json: {e}")

    # Submit the job to the executor
    try:
        utils.executor.submit(run_job_with_metadata, db_job.id, command, working_directory, service_id, request_params)
        logger.info(f"Job {db_job.id} submitted to executor. Job ID: {service_id}")
    except MemoryError:
        logger.error("Job exceeded memory limit")
        raise HTTPException(status_code=503, detail="The job exceeded the memory limit")
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {e}")

    # Return immediately with job ID and status URL
    host_url = utils.get_full_hostname()
    return {
        "jobID": service_id,
        "status_url": f"{host_url}/output/{service_id}",
        "message": "GEDI canopy height mapping job submitted. Poll status_url for results."
    }
