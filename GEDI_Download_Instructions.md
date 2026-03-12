# GEDI Global Download Instructions

This guide explains how to download the global GEDI L2A dataset (2024–2025)
using two scripts in the project root:

| Script | Purpose |
|---|---|
| `download_gedi_by_region.py` | Download GEDI by continent, partitioned into 1°×1° tiles |
| `merge_gedi_regions.py` | Merge all continental outputs into one global dataset |

---

## Prerequisites

### 1. Python environment
```bash
python
```

### 2. NASA EarthData account
Register for free at: https://urs.earthdata.nasa.gov

### 3. Credentials file
Create a `.env` file in the project root:
```
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
```

---

## Step 1 — Download by Continent

### List available regions
```bash
python download_gedi_by_region.py --list
```

Available continental regions (within GEDI coverage: 51.6°S – 51.6°N):

| Key | Region | Coverage |
|---|---|---|
| `north_america` | North America | Canada, USA, Mexico, Central America |
| `south_america` | South America | South American continent |
| `europe` | Europe | European continent |
| `africa` | Africa | African continent + Madagascar |
| `asia` | Asia | Middle East, India, Japan, SE Asia |
| `oceania` | Oceania | Australia, New Zealand, Pacific Islands |

### Download all continents sequentially
```bash
python download_gedi_by_region.py --all
```

### Download one or more specific continents
```bash
python download_gedi_by_region.py --region north_america
python download_gedi_by_region.py --region north_america europe asia
```

### Resume an interrupted download
Simply re-run the same command — the script automatically skips already
processed granules using a checkpoint file:
```bash
python download_gedi_by_region.py --region africa
```

### Reset and re-download a continent from scratch
```bash
python download_gedi_by_region.py --region europe --reset
```

### Output structure
Each continent produces a separate folder partitioned by 1°×1° grid cell:
```
gedi_na_2024_2025/
├── lat_34_lon_-118/
│   └── part.parquet
├── lat_47_lon_-122/
│   └── part.parquet
├── .gedi_checkpoint.json   ← resume state (do not delete)
└── PARTITION_SUMMARY.csv   ← row counts and sizes per tile
gedi_eu_2024_2025/
...
```

### Fields extracted per shot
| Field | Description |
|---|---|
| `latitude` | Shot latitude (lowestmode) |
| `longitude` | Shot longitude (lowestmode) |
| `rh98` | 98th percentile canopy height (m) |
| `rh50` | 50th percentile height (m) |
| `sensitivity` | Beam sensitivity (0–1) |
| `degrade_flag` | Degradation flag (0 = good) |

### Adding more fields or filters
Open `download_gedi_by_region.py` and edit `process_single_granule()`.

**To add a field** — add it inside the `with f:` beam loop alongside the
existing fields, then include it in the `df = pd.DataFrame({...})` block:
```python
cover = f[f'{beam}/cover'][:]          # canopy cover fraction
elev  = f[f'{beam}/elev_lowestmode'][:] # ground elevation (m)
```

**To add a filter** — extend the `valid` mask:
```python
valid = (quality == 1) & (rh98 > 0) & (rh98 < 130)  # existing
valid &= (sensitivity >= 0.95)    # add: sensitivity threshold
valid &= (degrade == 0)           # add: exclude degraded shots
```

---

## Step 2 — Merge into One Global Dataset

After all continents have been downloaded, merge them into a single unified
dataset with consistent 1°×1° partitioning.

### Dry run first (recommended)
Check what will be merged without writing any files:
```bash
python merge_gedi_regions.py \
    --source "gedi_*_2024_2025" \
    --output gedi_global_2024_2025 \
    --dry-run
```

### Run the merge
```bash
python merge_gedi_regions.py \
    --source "gedi_*_2024_2025" \
    --output gedi_global_2024_2025
```

### Merge specific continents only
```bash
python merge_gedi_regions.py \
    --source gedi_na_2024_2025 gedi_eu_2024_2025 gedi_af_2024_2025 \
    --output gedi_global_2024_2025
```

### Output structure
```
gedi_global_2024_2025/
├── lat_34_lon_-118/
│   └── part.parquet
├── lat_47_lon_-122/
│   └── part.parquet
├── ...
└── PARTITION_SUMMARY.csv   ← total rows and size per tile
```

Tiles that appear in only one continent are copied directly. Tiles that
overlap between continental bboxes are concatenated and deduplicated.

---

## Full Workflow Summary

```bash
# 1. Set up credentials
echo "EARTHDATA_USERNAME=your_username" >> .env
echo "EARTHDATA_PASSWORD=your_password" >> .env

# 2. Download all continents (resumable — safe to interrupt and re-run)
python download_gedi_by_region.py --all

# 3. Preview the merge
python merge_gedi_regions.py \
    --source "gedi_*_2024_2025" \
    --output gedi_global_2024_2025 \
    --dry-run

# 4. Run the merge
python merge_gedi_regions.py \
    --source "gedi_*_2024_2025" \
    --output gedi_global_2024_2025
```

---

## Notes

- **GEDI coverage**: latitude 51.6°S to 51.6°N only (ISS orbit limit). Higher
  latitudes (e.g. northern Canada, Scandinavia above 51.6°N) are not covered.
- **Date range**: 2024-01-01 to 2025-12-31. To change, edit `START_DATE` and
  `END_DATE` at the top of `download_gedi_by_region.py`.
- **Disk space**: expect roughly 50–200 GB per continent depending on forest
  density. Check `PARTITION_SUMMARY.csv` in each continental folder for an
  exact count after download.
- **Corrupted files**: granules smaller than 50 MB are automatically skipped
  and marked as failed in the checkpoint. They will be retried on the next run.
- **Rate limits**: if NASA's API rate-limits the download, the script waits
  60 seconds and retries automatically (up to 3 attempts per batch).
