#!/usr/bin/env python3
"""
Download and Partition GEDI L2A Data by Continental Regions (2022)
Partitions by 1° × 1° grid cells for efficient spatial access.

RESUME CAPABILITY: Can be interrupted and resumed. Progress is tracked
in a checkpoint file. Already processed granules are skipped.

Output structure:
    gedi_na_2022/     # North America
    ├── lat_32_lon_-118/
    │   └── part.parquet
    ├── .checkpoint.json
    ...

    gedi_eu_2022/     # Europe
    └── ...

Usage:
    # List all available regions
    python download_gedi_by_region.py --list

    # Download specific region
    python download_gedi_by_region.py --region north_america

    # Download multiple regions
    python download_gedi_by_region.py --region north_america south_america

    # Download all regions sequentially
    python download_gedi_by_region.py --all

    # Resume interrupted region
    python download_gedi_by_region.py --region africa
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import earthaccess
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing

# GEDI coverage limits (ISS orbit inclination)
GEDI_MAX_LAT = 51.6
GEDI_MIN_LAT = -51.6

# Date range
START_DATE = '2022-04-01'
END_DATE = '2022-09-30'

# Processing parameters
BATCH_SIZE = 16  # Process in batches of N granules
RATE_LIMIT_DELAY = 60  # Seconds to wait if rate limited
CHECKPOINT_FILE = '.gedi_checkpoint.json'

# Continental regions (within GEDI coverage: 51.6°N to 51.6°S)
REGIONS = {
    "north_america": {
        "name": "North America",
        "bbox": [-180, 15, -34, GEDI_MAX_LAT],
        "output": "gedi_na_2022",
        "description": "Canada, USA, Mexico, Central America"
    },
    "south_america": {
        "name": "South America",
        "bbox": [-180, GEDI_MIN_LAT, -34, 15],
        "output": "gedi_sa_2022",
        "description": "South American continent"
    },
    "europe": {
        "name": "Europe",
        "bbox": [-34, 34, 45, GEDI_MAX_LAT],
        "output": "gedi_eu_2022",
        "description": "European continent"
    },
    "africa": {
        "name": "Africa",
        "bbox": [-34, GEDI_MIN_LAT, 45, 34],
        "output": "gedi_af_2022",
        "description": "African continent + Madagascar"
    },
    "asia": {
        "name": "Asia",
        "bbox": [45, -10, 180, GEDI_MAX_LAT],
        "output": "gedi_as_2022",
        "description": "Asia (including Middle East, Japan, Southeast Asia)"
    },
    "oceania": {
        "name": "Oceania",
        "bbox": [45, GEDI_MIN_LAT, 180, -10],
        "output": "gedi_oc_2022",
        "description": "Australia, New Zealand, Pacific Islands"
    }
}


def list_regions():
    """Print all available regions with details."""
    print("\n" + "="*70)
    print("Available Continental Regions (2022)")
    print("="*70)
    print(f"\nNote: GEDI covers {GEDI_MIN_LAT}° to {GEDI_MAX_LAT}° (ISS orbit limits)\n")

    for key, region in REGIONS.items():
        print(f"  {key:20} | {region['name']:15} | BBox: {region['bbox']}")
        print(f"{'':20} | {'':15} | Output: {region['output']}")
        print(f"{'':20} | {'':15} | {region['description']}")
        print()

    print("="*70)
    print("\nUsage:")
    print("  python download_gedi_by_region.py --region <region_name>")
    print("  python download_gedi_by_region.py --all")
    print()


def get_partition_key(lat, lon):
    """Get partition key for a point (1° × 1° grid cell)."""
    lat_floor = int(np.floor(lat))
    lon_floor = int(np.floor(lon))
    return f'lat_{lat_floor}_lon_{lon_floor}'

def process_single_granule(h5_file_path, bbox=None):
    """
    Extract GEDI data from a single HDF5 file and partition by grid cell.
    Optimized to only read valid rows from disk and include requested sensitivity fields.
    """
    try:
        partitions = {}

        # Validate file can be opened
        try:
            f = h5py.File(h5_file_path, 'r')
        except Exception as e:
            print(f"    Warning: Cannot open HDF5 file: {e}")
            return None

        with f:
            # Get all beam names (e.g., BEAM0000, BEAM0001, etc.)
            beams = [k for k in f.keys() if k.startswith('BEAM')]

            for beam in beams:
                try:
                    # --- STEP 1: FAST READ FOR FILTERING ---
                    # Only read the flags and RH98 to decide what to keep
                    quality = f[f'{beam}/quality_flag'][:]
                    degrade = f[f'{beam}/degrade_flag'][:]
                    rh_all = f[f'{beam}/rh'][:]
                    rh98 = rh_all[:, 98]

                    # APPLY STRICT FILTER: quality=1 AND degrade=0 AND height limits
                    valid = (quality == 1) & (degrade == 0) & (rh98 > 0) & (rh98 < 130)

                    # Optional bounding box filter (requires lat/lon)
                    lat = f[f'{beam}/lat_lowestmode'][:]
                    lon = f[f'{beam}/lon_lowestmode'][:]
                    if bbox is not None:
                        valid &= (
                            (lon >= bbox[0]) & (lon <= bbox[2]) &
                            (lat >= bbox[1]) & (lat <= bbox[3])
                        )

                    # EARLY EXIT: If no shots in this beam are valid, skip to next beam
                    if not np.any(valid):
                        continue

                    # --- STEP 2: LAZY READ OF REMAINING FIELDS ---
                    # We only pull the data for rows where 'valid' is True
                    # This saves significant memory and I/O time
                    df = pd.DataFrame({
                        'beam': f[f'{beam}/beam'][valid],
                        'latitude': lat[valid],
                        'longitude': lon[valid],
                        'rh98': rh98[valid],
                        'solar_elevation': f[f'{beam}/solar_elevation'][valid],
                        'sensitivity': f[f'{beam}/sensitivity'][valid]
                    })

                    # --- STEP 3: ORIGINAL PARTITIONING LOGIC ---
                    # Vectorized partition assignment (100x+ faster than loop)
                    lat_floor = np.floor(df['latitude']).astype(int)
                    lon_floor = np.floor(df['longitude']).astype(int)

                    # Add temporary keys to the DataFrame for grouping
                    df['_lat_key'] = lat_floor
                    df['_lon_key'] = lon_floor

                    # Group by partition key and collect data
                    for (lat_k, lon_k), group in df.groupby(['_lat_key', '_lon_key']):
                        key = f'lat_{lat_k}_lon_{lon_k}'
                        if key not in partitions:
                            partitions[key] = []

                        # Define exactly which columns to export to the final list
                        output_cols = [
                            'beam',
                            'latitude',
                            'longitude',
                            'rh98',
                            'solar_elevation',
                            'sensitivity'
                        ]

                        # Use to_dict('records') for fast conversion
                        partitions[key].extend(group[output_cols].to_dict('records'))

                except Exception as e:
                    # Skip problematic beams but keep processing others
                    print(f"    Error processing beam {beam}: {e}")
                    continue

        return partitions

    except Exception as e:
        print(f"Error in process_single_granule: {e}")
        return None

def load_checkpoint(checkpoint_path):
    """Load checkpoint file if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        'processed_granules': [],
        'failed_granules': [],
        'last_update': None
    }


def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save checkpoint file."""
    checkpoint_data['last_update'] = datetime.now().isoformat()
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def get_granule_id(granule_meta):
    """Extract unique ID from granule metadata."""
    return granule_meta['meta']['native-id']


def append_to_partition(partition_dir, data_list):
    """
    Append data to an existing partition or create a new one.

    Args:
        partition_dir: Path to partition directory
        data_list: List of dicts to append
    """
    partition_dir = Path(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = partition_dir / 'part.parquet'

    # Convert to DataFrame
    new_df = pd.DataFrame(data_list)

    # If file exists, read, append, and save
    if parquet_file.exists():
        existing_df = pd.read_parquet(parquet_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(parquet_file, index=False, compression='snappy')
    else:
        new_df.to_parquet(parquet_file, index=False, compression='snappy')


def download_region(region_key, n_workers=None, batch_size=BATCH_SIZE, reset=False):
    """
    Download GEDI data for a single region.

    Args:
        region_key: Key from REGIONS dict
        n_workers: Number of parallel workers
        batch_size: Granules per batch
        reset: Reset checkpoint and start over
    """
    if region_key not in REGIONS:
        print(f"❌ Unknown region: {region_key}")
        print(f"   Use --list to see available regions")
        return False

    region = REGIONS[region_key]
    bbox = region['bbox']
    output_dir = region['output']
    region_name = region['name']

    print("\n" + "="*70)
    print(f"GEDI {region_name} 2022 - Download and Partition")
    print("="*70)
    print(f"\nBounding Box: {bbox}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Size: {batch_size} granules")
    print(f"Workers: {n_workers or 'auto'}")
    print("\n✨ RESUME ENABLED: Can be interrupted and resumed!")

    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    checkpoint_path = output_path / CHECKPOINT_FILE

    # Handle reset
    if reset:
        if checkpoint_path.exists():
            print(f"\n⚠ Resetting {region_name} checkpoint - will re-download all granules!")
            checkpoint_path.unlink()
            # Also remove all partition directories
            for partition_dir in output_path.iterdir():
                if partition_dir.is_dir() and partition_dir.name.startswith('lat_'):
                    shutil.rmtree(partition_dir)
        else:
            print(f"\nNo checkpoint found for {region_name}.")

    # Check credentials
    username = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')

    if not username or not password:
        print("\n❌ Error: EARTHDATA_USERNAME and EARTHDATA_PASSWORD not set.")
        print("Please set them in .env file:")
        print("  EARTHDATA_USERNAME=your_username")
        print("  EARTHDATA_PASSWORD=your_password")
        return False

    os.environ['EARTHDATA_USERNAME'] = username
    os.environ['EARTHDATA_PASSWORD'] = password

    # Login
    print("\n🔐 Logging into NASA EarthData...")
    try:
        earthaccess.login()
        print("✓ Login successful")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    processed_ids = set(checkpoint['processed_granules'])
    failed_ids = set(checkpoint['failed_granules'])

    if processed_ids:
        print(f"\n📋 RESUME: Found {len(processed_ids)} already processed granules")
        if failed_ids:
            print(f"⚠ Found {len(failed_ids)} previously failed granules (will retry)")

    # Search for granules
    print(f"\n🔍 Searching for GEDI L2A granules...")
    results = earthaccess.search_data(
        short_name='GEDI02_A',
        version='002',
        bounding_box=tuple(bbox),
        temporal=(START_DATE, END_DATE)
    )

    print(f"✓ Found {len(results)} total granules")

    # Filter out already processed granules
    pending_granules = []
    for r in results:
        granule_id = get_granule_id(r)
        if granule_id not in processed_ids:
            pending_granules.append(r)


    #pending_granules = pending_granules[:1]


    print(f"  Pending: {len(pending_granules)} granules")
    print(f"  Already processed: {len(processed_ids)} granules")

    if len(pending_granules) == 0:
        print("\n✅ All granules already processed for this region!")
        return True

    # Set number of workers
    if n_workers is None:
        n_workers = min(8, multiprocessing.cpu_count())

    print(f"\n⚙️ Using {n_workers} workers")

    # Create temp directory
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='gedi_download_')
    print(f"📁 Temporary directory: {temp_dir}")

    # Process in batches
    total_pending = len(pending_granules)
    start_time = time.time()

    for batch_start in range(0, total_pending, batch_size):
        batch_end = min(batch_start + batch_size, total_pending)
        batch = pending_granules[batch_start:batch_end]

        print("\n" + "="*70)
        print(f"BATCH {batch_start//batch_size + 1}/{(total_pending-1)//batch_size + 1}")
        print(f"Processing granules {batch_start+1}-{batch_end} of {total_pending}")
        print("="*70)

        # Download batch
        print(f"\n⬇️ Downloading {len(batch)} granules...")
        batch_files = []
        batch_granule_ids = []

        for granule in batch:
            granule_id = get_granule_id(granule)
            batch_granule_ids.append(granule_id)

        # Download with retry for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                downloaded = earthaccess.download(batch, temp_dir)
                batch_files = downloaded
                print(f"✓ Downloaded {len(batch_files)} files")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = RATE_LIMIT_DELAY * (attempt + 1)
                    print(f"⚠ Download failed (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Download failed after {max_retries} attempts: {e}")
                    # Mark as failed and continue
                    for gid in batch_granule_ids:
                        if gid not in processed_ids:
                            failed_ids.add(gid)
                    save_checkpoint(checkpoint_path, {
                        'processed_granules': list(processed_ids),
                        'failed_granules': list(failed_ids),
                        'last_update': datetime.now().isoformat()
                    })
                    continue

        # Process batch
        print(f"\n🔄 Processing {len(batch_files)} granules...")

        for granule_id, h5_file in zip(batch_granule_ids, batch_files):
            if granule_id in processed_ids:
                continue

            # Validate file size (GEDI files are typically >100MB, empty/corrupted files are <50MB)
            file_size_mb = os.path.getsize(h5_file) / (1024 * 1024)
            if file_size_mb < 1:
                print(f"  ⚠ Skipping {granule_id}: file too small ({file_size_mb:.1f}MB, likely corrupted)")
                failed_ids.add(granule_id)
                try:
                    os.remove(h5_file)
                except:
                    pass
                continue

            try:
                partitions = process_single_granule(h5_file, bbox)

                # If partitions is None, treat as failed (error opening/reading file, etc.)
                if partitions is None:
                    print(f"  ⚠ {granule_id}: processing returned None (treating as failed)")
                    failed_ids.add(granule_id)

                # If partitions is empty, treat as successfully processed but warn (no valid shots)
                elif len(partitions) == 0:
                    print(f"  ⚠ {granule_id}: no valid shots after filtering (marked as processed)")
                    processed_ids.add(granule_id)
                    failed_ids.discard(granule_id)

                else:
                    # Append each partition to existing data
                    for key, data in partitions.items():
                        partition_dir = output_path / key
                        append_to_partition(partition_dir, data)

                    processed_ids.add(granule_id)
                    # Remove from failed if it was there
                    failed_ids.discard(granule_id)

            except Exception as e:
                print(f"  ⚠ Error processing {granule_id}: {e}")
                failed_ids.add(granule_id)

            # Delete the HDF5 file to save space
            try:
                os.remove(h5_file)
            except:
                pass

        # Update checkpoint after each batch
        save_checkpoint(checkpoint_path, {
            'processed_granules': list(processed_ids),
            'failed_granules': list(failed_ids),
            'last_update': datetime.now().isoformat()
        })

        # Progress update
        elapsed = time.time() - start_time
        remaining = len(pending_granules) - batch_end
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_hours = eta_seconds / 3600

        print(f"\n📊 Progress: {batch_end}/{total_pending} granules ({batch_end/total_pending*100:.1f}%)")
        print(f"   Elapsed: {elapsed/3600:.1f}h | ETA: {eta_hours:.1f}h")

        # Small delay between batches to be nice to the API
        if batch_end < total_pending:
            print(f"   Pausing 5 seconds before next batch...")
            time.sleep(5)

    # Cleanup temp directory
    print(f"\n🧹 Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    # Final statistics
    print("\n" + "="*70)
    print(f"✅ {region_name.upper()} DOWNLOAD COMPLETE!")
    print("="*70)

    # Scan all partitions for final stats
    stats = []
    for partition_dir in output_path.iterdir():
        if partition_dir.is_dir() and not partition_dir.name.startswith('.'):
            parquet_file = partition_dir / 'part.parquet'
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                stats.append({
                    'partition': partition_dir.name,
                    'rows': len(df),
                    'size_mb': parquet_file.stat().st_size / (1024 * 1024)
                })

    if stats:
        stats_df = pd.DataFrame(stats)
        print(f"\nTotal partitions: {len(stats_df)}")
        print(f"Total rows: {stats_df['rows'].sum():,}")
        print(f"Total size: {stats_df['size_mb'].sum():.1f} MB ({stats_df['size_mb'].sum()/1024:.2f} GB)")

        # Save summary
        summary_file = output_path / 'PARTITION_SUMMARY.csv'
        stats_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved partition summary to: {summary_file}")

    if failed_ids:
        print(f"\n⚠ Warning: {len(failed_ids)} granules failed to process")
        print(f"   Run again to retry, or check .gedi_checkpoint.json for details")

    print(f"\n📁 Data saved to: {output_dir}/")

    return True


def download_all_regions(n_workers=None, batch_size=BATCH_SIZE):
    """Download all regions sequentially."""
    print("\n" + "="*70)
    print("Downloading ALL Continental Regions (2022)")
    print("="*70)

    total_regions = len(REGIONS)
    results = {}

    for i, (region_key, region) in enumerate(REGIONS.items(), 1):
        print(f"\n{'='*70}")
        print(f"REGION {i}/{total_regions}: {region['name']}")
        print(f"{'='*70}")

        success = download_region(region_key, n_workers=n_workers, batch_size=batch_size)

        if success:
            results[region_key] = "✅ Complete"
        else:
            results[region_key] = "❌ Failed"

        # Small delay between regions
        if i < total_regions:
            print(f"\n⏸ Waiting 10 seconds before next region...")
            time.sleep(10)

    # Final summary
    print("\n" + "="*70)
    print("ALL REGIONS - FINAL SUMMARY")
    print("="*70)
    for region_key, status in results.items():
        region_name = REGIONS[region_key]['name']
        print(f"  {region_name:20} | {status}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download and partition GEDI data by continental regions (RESUME CAPABLE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available regions
  python download_gedi_by_region.py --list

  # Download specific region
  python download_gedi_by_region.py --region north_america

  # Download multiple regions
  python download_gedi_by_region.py --region north_america south_america

  # Download all regions sequentially
  python download_gedi_by_region.py --all

  # Resume interrupted region
  python download_gedi_by_region.py --region africa

  # Reset and re-download a region
  python download_gedi_by_region.py --region europe --reset
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List all available regions')
    parser.add_argument('--region', nargs='+',
                        help='Region name(s) to download')
    parser.add_argument('--all', action='store_true',
                        help='Download all regions sequentially')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: 8, max: CPU count)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Granules per batch (default: {BATCH_SIZE})')
    parser.add_argument('--reset', action='store_true',
                        help='Reset checkpoint and start over (WARNING: re-downloads everything)')

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_regions()
        sys.exit(0)

    # Handle --all
    if args.all:
        download_all_regions(n_workers=args.workers, batch_size=args.batch_size)
        sys.exit(0)

    # Handle --region
    if args.region:
        for region_key in args.region:
            download_region(region_key, n_workers=args.workers, batch_size=args.batch_size, reset=args.reset)
        sys.exit(0)

    # No arguments: show help
    parser.print_help()
    print("\n💡 Use --list to see available regions")
