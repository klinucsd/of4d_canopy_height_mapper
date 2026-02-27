#!/usr/bin/env python3
"""
Merge GEDI Regional Partitions into Single Global Dataset

Combines multiple regional GEDI datasets (e.g., gedi_na_2024_2025, gedi_eu_2024_2025)
into a single unified dataset with matching partitions merged.

Usage:
    # Merge all regional folders matching pattern
    python merge_gedi_regions.py --source "gedi_*_2024_2025" --output gedi_global_2024_2025

    # Merge specific folders
    python merge_gedi_regions.py --source gedi_na_2024_2025 gedi_eu_2024_2025 --output gedi_global_2024_2025

    # Dry run (show what would be merged)
    python merge_gedi_regions.py --source "gedi_*_2024_2025" --output gedi_global_2024_2025 --dry-run
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime


def find_source_folders(pattern=None, folders=None):
    """
    Find source GEDI folders to merge.

    Args:
        pattern: Glob pattern to match folders (e.g., "gedi_*_2024_2025")
        folders: Explicit list of folder paths

    Returns:
        list: List of Path objects for source folders
    """
    if folders:
        return [Path(f) for f in folders if Path(f).exists() and Path(f).is_dir()]

    if pattern:
        matched = glob.glob(pattern)
        return [Path(f) for f in matched if Path(f).is_dir()]

    return []


def get_partitions_from_folder(folder_path):
    """
    Get all partition directories from a GEDI folder.

    Args:
        folder_path: Path to GEDI region folder

    Returns:
        dict: {partition_key: Path to part.parquet}
    """
    partitions = {}
    folder = Path(folder_path)

    if not folder.exists():
        return partitions

    for item in folder.iterdir():
        if item.is_dir() and item.name.startswith('lat_'):
            parquet_file = item / 'part.parquet'
            if parquet_file.exists():
                partitions[item.name] = parquet_file

    return partitions


def merge_partitions(source_folders, output_folder, dry_run=False):
    """
    Merge all source partitions into output folder.

    Args:
        source_folders: List of source folder Paths
        output_folder: Output folder Path
        dry_run: If True, only print what would be done
    """
    print("="*70)
    print(f"MERGE GEDI REGIONS → {output_folder}")
    print("="*70)

    # Collect all partitions from all sources
    all_partitions = {}  # {partition_key: [source_paths]}

    print(f"\n📂 Scanning {len(source_folders)} source folders...")
    for source in source_folders:
        partitions = get_partitions_from_folder(source)
        print(f"  {source.name}: {len(partitions)} partitions")

        for key, parquet_path in partitions.items():
            if key not in all_partitions:
                all_partitions[key] = []
            all_partitions[key].append(parquet_path)

    print(f"\n✓ Found {len(all_partitions)} unique partitions")

    # Check for partitions that need merging (multiple sources)
    needs_merge = {k: v for k, v in all_partitions.items() if len(v) > 1}
    single_source = {k: v for k, v in all_partitions.items() if len(v) == 1}

    print(f"  Partitions from single source: {len(single_source)}")
    print(f"  Partitions needing merge: {len(needs_merge)}")

    if dry_run:
        print("\n" + "="*70)
        print("DRY RUN - Would merge the following:")
        print("="*70)
        for key in sorted(needs_merge.keys())[:10]:  # Show first 10
            sources = needs_merge[key]
            print(f"  {key}: {len(sources)} sources")
            for s in sources:
                print(f"    - {s.parent.name}/")
        if len(needs_merge) > 10:
            print(f"  ... and {len(needs_merge) - 10} more")
        print("="*70)
        return

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge statistics
    stats = {
        'partitions_created': 0,
        'partitions_merged': 0,
        'total_rows': 0,
        'total_size_mb': 0
    }

    print("\n" + "="*70)
    print("MERGING PARTITIONS")
    print("="*70)

    # Process each partition
    for i, (key, sources) in enumerate(sorted(all_partitions.items()), 1):
        output_partition_dir = output_path / key

        if len(sources) == 1:
            # Just copy
            if i % 100 == 0 or i == 1:
                print(f"\n[{i}/{len(all_partitions)}] Copying: {key}")

            output_partition_dir.mkdir(parents=True, exist_ok=True)
            output_parquet = output_partition_dir / 'part.parquet'

            # Read and write (to get stats)
            df = pd.read_parquet(sources[0])
            df.to_parquet(output_parquet, index=False, compression='snappy')

            stats['partitions_created'] += 1
            stats['total_rows'] += len(df)
            stats['total_size_mb'] += output_parquet.stat().st_size / (1024 * 1024)

        else:
            # Merge multiple sources
            if i % 100 == 0 or i == 1:
                print(f"\n[{i}/{len(all_partitions)}] Merging: {key} ({len(sources)} sources)")

            output_partition_dir.mkdir(parents=True, exist_ok=True)
            output_parquet = output_partition_dir / 'part.parquet'

            # Read and combine all sources
            dataframes = []
            for source_path in sources:
                df = pd.read_parquet(source_path)
                dataframes.append(df)

            merged_df = pd.concat(dataframes, ignore_index=True)
            merged_df.to_parquet(output_parquet, index=False, compression='snappy')

            stats['partitions_merged'] += 1
            stats['total_rows'] += len(merged_df)
            stats['total_size_mb'] += output_parquet.stat().st_size / (1024 * 1024)

    # Copy checkpoint from first source (optional, for reference)
    for source in source_folders:
        checkpoint = source / '.gedi_checkpoint.json'
        if checkpoint.exists():
            shutil.copy2(checkpoint, output_path / '.checkpoint_from_regions.json')
            break

    # Print final statistics
    print("\n" + "="*70)
    print("✅ MERGE COMPLETE!")
    print("="*70)
    print(f"\nOutput folder: {output_path}")
    print(f"\nStatistics:")
    print(f"  Total partitions: {stats['partitions_created'] + stats['partitions_merged']}")
    print(f"    - Copied (single source): {stats['partitions_created']}")
    print(f"    - Merged (multiple sources): {stats['partitions_merged']}")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB ({stats['total_size_mb']/1024:.2f} GB)")

    # Save summary
    summary_data = []
    for partition_dir in sorted(output_path.iterdir()):
        if partition_dir.is_dir() and partition_dir.name.startswith('lat_'):
            parquet_file = partition_dir / 'part.parquet'
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                summary_data.append({
                    'partition': partition_dir.name,
                    'rows': len(df),
                    'size_mb': parquet_file.stat().st_size / (1024 * 1024)
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / 'PARTITION_SUMMARY.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved partition summary to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge GEDI regional partitions into single dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all regional folders
  python merge_gedi_regions.py --source "gedi_*_2024_2025" --output gedi_global_2024_2025

  # Merge specific folders
  python merge_gedi_regions.py --source gedi_na_2024_2025 gedi_eu_2024_2025 --output gedi_merged

  # Dry run first
  python merge_gedi_regions.py --source "gedi_*_2024_2025" --output gedi_global_2024_2025 --dry-run
        """
    )

    parser.add_argument('--source', nargs='+',
                        help='Source folder(s) or glob pattern (use quotes for wildcards)')
    parser.add_argument('--output', required=True,
                        help='Output folder for merged dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be merged without actually doing it')

    args = parser.parse_args()

    # Find source folders
    if len(args.source) == 1 and '*' in args.source[0]:
        source_folders = find_source_folders(pattern=args.source[0])
    else:
        source_folders = find_source_folders(folders=args.source)

    if not source_folders:
        print("❌ No source folders found!")
        print("   Use --source with folder names or a glob pattern")
        print("   Example: --source 'gedi_*_2024_2025'")
        exit(1)

    print(f"Found {len(source_folders)} source folders:")
    for f in source_folders:
        print(f"  - {f}")

    # Merge
    merge_partitions(source_folders, args.output, dry_run=args.dry_run)
