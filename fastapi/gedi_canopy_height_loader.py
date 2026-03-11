#!/usr/bin/env python3
"""
GEDI Partition Loader for Service API

Efficiently load GEDI data for a given bbox from partitioned Parquet files.
Designed for use in the canopy height service API.

Usage:
    from gedi_partition_loader import load_gedi_for_bbox
    gedi_df = load_gedi_for_bbox([-117.28, 32.53, -116.42, 33.51])
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


# Default path to partitioned GEDI data
DEFAULT_GEDI_DIR = 'gedi_usa_2024_2025'


def get_partitions_for_bbox(
    bbox: List[float],
    gedi_dir: str = DEFAULT_GEDI_DIR
) -> List[str]:
    """
    Get list of partition directories that may contain data for the bbox.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        gedi_dir: Root directory of partitioned GEDI data

    Returns:
        List of partition directory paths
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    gedi_path = Path(gedi_dir)

    # Determine which grid cells intersect the bbox
    # Include ±1 cell buffer for edge cases
    partitions = []

    for lat in range(int(np.floor(min_lat)) - 1, int(np.floor(max_lat)) + 2):
        for lon in range(int(np.floor(min_lon)) - 1, int(np.floor(max_lon)) + 2):
            key = f'lat_{lat}_lon_{lon}'
            partition_dir = gedi_path / key
            if partition_dir.exists():
                partitions.append(partition_dir)

    return partitions


def load_gedi_for_bbox(
    bbox: List[float],
    gedi_dir: str = DEFAULT_GEDI_DIR,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load GEDI data for a specific bbox from partitioned Parquet files.

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        gedi_dir: Root directory of partitioned GEDI data
        max_rows: Optional maximum number of rows to return (for testing)

    Returns:
        pd.DataFrame: GEDI data with columns [latitude, longitude, rh98]
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Get relevant partitions
    partitions = get_partitions_for_bbox(bbox, gedi_dir)

    if not partitions:
        print(f"⚠ No GEDI partitions found for bbox {bbox}")
        return pd.DataFrame()

    # Load and combine data
    dfs = []
    total_rows = 0

    for partition_dir in partitions:
        parquet_file = partition_dir / 'part.parquet'
        if parquet_file.exists():
            # Read parquet file (fast columnar read)
            df = pd.read_parquet(parquet_file)

            # Filter to exact bbox
            df_filtered = df[
                (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
                (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)
            ]

            if len(df_filtered) > 0:
                dfs.append(df_filtered)
                total_rows += len(df_filtered)

                # Early exit if max_rows reached
                if max_rows and total_rows >= max_rows:
                    break

    if dfs:
        result = pd.concat(dfs, ignore_index=True)

        # Trim to max_rows if needed
        if max_rows and len(result) > max_rows:
            result = result.head(max_rows)

        return result

    return pd.DataFrame()


def get_gedi_stats(bbox: List[float], gedi_dir: str = DEFAULT_GEDI_DIR) -> dict:
    """
    Get statistics about GEDI data availability for a bbox.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        gedi_dir: Root directory of partitioned GEDI data

    Returns:
        dict with keys: total_rows, partitions_loaded, bbox_area_km2
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Calculate approximate area
    lat_km = 111  # 1 degree latitude ≈ 111 km
    lon_km = 111 * np.cos(np.radians((min_lat + max_lat) / 2))
    area_km2 = (max_lat - min_lat) * lat_km * (max_lon - min_lon) * lon_km

    # Count partitions
    partitions = get_partitions_for_bbox(bbox, gedi_dir)

    # Load data to count rows
    gedi_df = load_gedi_for_bbox(bbox, gedi_dir)

    return {
        'total_rows': len(gedi_df),
        'partitions_loaded': len(partitions),
        'partitions_list': [str(p.name) for p in partitions],
        'bbox': bbox,
        'bbox_area_km2': round(area_km2, 2),
        'points_per_km2': round(len(gedi_df) / area_km2, 2) if area_km2 > 0 else 0
    }


def is_gedi_data_available(bbox: List[float], gedi_dir: str = DEFAULT_GEDI_DIR) -> bool:
    """
    Quick check if GEDI data is available for a bbox.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        gedi_dir: Root directory of partitioned GEDI data

    Returns:
        True if at least one partition exists for this bbox
    """
    partitions = get_partitions_for_bbox(bbox, gedi_dir)
    return len(partitions) > 0


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("GEDI Partition Loader - Test")
    print("="*60)

    # Test bboxes
    test_cases = [
        ([-117.28, 32.53, -116.42, 33.51], "San Diego County"),
        ([-118.27, 33.975, -117.73, 34.425], "Angeles National Forest"),
        ([-122.45, 37.72, -122.35, 37.78], "San Francisco (small)"),
        ([-80.25, 25.75, -80.05, 25.85], "Miami (small)"),
    ]

    for bbox, label in test_cases:
        print(f"\n{label}: {bbox}")

        # Check availability
        available = is_gedi_data_available(bbox)
        print(f"  Available: {available}")

        if available:
            # Get stats
            stats = get_gedi_stats(bbox)
            print(f"  Rows: {stats['total_rows']:,}")
            print(f"  Partitions: {stats['partitions_loaded']}")
            print(f"  Area: {stats['bbox_area_km2']:.1f} km²")
            print(f"  Density: {stats['points_per_km2']:.1f} pts/km²")
