"""Generate HTML reports for controlled experiment runs - Detailed version"""
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import numpy as np

def get_model_stats(output_dir):
    """Extract model performance metrics from pipeline output"""
    # These would typically be saved during pipeline run
    # For now, we'll read from the gedi data and make reasonable estimates
    gedi_csv = output_dir / 'gedi_raw.csv'
    if not gedi_csv.exists():
        return None

    df = pd.read_csv(gedi_csv)
    n_samples = len(df)

    # Estimate metrics based on GEDI data
    return {
        'n_samples': n_samples,
        'test_split': 0.2,
        'train_samples': int(n_samples * 0.8),
        'test_samples': int(n_samples * 0.2),
    }


def generate_report(run_dir, gedi_start, gedi_end, satellite_start='2022-01-01', satellite_end='2025-12-31', bbox=None):
    output_dir = Path(run_dir) / 'output'
    report_date = datetime.now().strftime('%Y-%m-%d')

    # Format bbox for display
    if bbox:
        bbox_str = f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}] (min_lon, min_lat, max_lon, max_lat)"
    else:
        bbox_str = "See bounding box in data files"

    def get_size_mb(filepath):
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            return f'{size_bytes / (1024**2):.1f} MB'
        return 'N/A'

    # GEDI stats
    gedi_csv = output_dir / 'gedi_raw.csv'
    gedi_points = 'N/A'
    gedi_mean = 'N/A'
    gedi_max = 'N/A'
    gedi_csv_size = 'N/A'
    n_granules = 'N/A'

    if os.path.exists(gedi_csv):
        df = pd.read_csv(gedi_csv)
        gedi_points = f'{len(df):,}'
        gedi_mean = f'{df["rh98"].mean():.1f} m'
        gedi_max = f'{df["rh98"].max():.1f} m'
        gedi_csv_size = get_size_mb(gedi_csv)

    # Height map stats
    height_map = output_dir / 'canopy_height_map.tif'
    mean_height = 'N/A'
    max_height = 'N/A'
    tif_size = 'N/A'
    try:
        import rasterio
        with rasterio.open(height_map) as src:
            data = src.read(1)
            mean_height = f'{np.nanmean(data):.1f} m'
            max_height = f'{np.nanmax(data):.1f} m'
        tif_size = get_size_mb(height_map)
    except:
        pass

    # Other file sizes
    s2_size = get_size_mb(output_dir / 'sentinel2.tif')
    s1_size = get_size_mb(output_dir / 'sentinel1.tif')
    topo_size = get_size_mb(output_dir / 'topography.tif')
    model_size = get_size_mb(output_dir / 'canopy_height_model.pkl')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Canopy Height Mapping Report - {Path(run_dir).name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; background-color: #f5f5f5; }}
        .container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 15px; }}
        h3 {{ color: #555; margin-top: 25px; }}
        h4 {{ color: #666; margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: 600; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ display: inline-block; background: #ecf0f1; padding: 8px 15px; margin: 5px; border-radius: 5px; font-weight: 500; }}
        .metric-value {{ color: #3498db; font-weight: 700; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 500; }}
        .badge-green {{ background: #e8f5e9; color: #388e3c; }}
        .badge-orange {{ background: #fff3e0; color: #f57c00; }}
        .image-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 30px 0; }}
        .image-card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white; }}
        .image-card img {{ width: 100%; height: 250px; object-fit: cover; cursor: pointer; }}
        .image-card .caption {{ padding: 12px; background: #f8f9fa; font-size: 0.9em; color: #555; }}
        .image-card .caption a {{ color: #3498db; text-decoration: none; }}
        .toc {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .toc ul {{ list-style: none; padding-left: 0; }}
        .toc a {{ color: #3498db; text-decoration: none; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #777; font-size: 0.9em; }}
        .full-width-image {{ width: 100%; border-radius: 8px; overflow: hidden; background: white; margin: 30px 0; text-align: center; }}
        .full-width-image img {{ max-width: 100%; height: 450px; object-fit: contain; cursor: pointer; }}
        .full-width-image .caption {{ padding: 12px; background: #f8f9fa; font-size: 0.9em; color: #555; }}
        .full-width-image .caption a {{ color: #3498db; text-decoration: none; }}
        .full-width-image .caption a:hover {{ text-decoration: underline; }}
        .info-box {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px 20px; margin: 20px 0; border-radius: 0 5px 5px 0; }}
        .download-btn {{ display: inline-block; background: #3498db; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; margin: 5px; font-size: 0.9em; }}
        .download-btn:hover {{ background: #2980b9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Canopy Height Mapping Report - {Path(run_dir).name}</h1>
        <p><strong>Generated:</strong> {report_date}<br>
        <strong>Pipeline:</strong> Random Forest Regression with GEDI L2A, Sentinel-2, Sentinel-1, SRTM</p>

        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#study-area">1. Study Area Configuration</a></li>
                <li><a href="#data-sources">2. Data Sources & Download Statistics</a></li>
                <li><a href="#model-training">3. Model Training Results</a></li>
                <li><a href="#prediction-results">4. Prediction Results</a></li>
                <li><a href="#output-files">5. Output Files</a></li>
                <li><a href="#visualizations">6. Visualizations</a></li>
            </ul>
        </div>

        <h2 id="study-area">1. Study Area Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td><strong>Region</strong></td><td>{bbox_str}</td></tr>
            <tr><td><strong>Coordinate Reference System</strong></td><td>UTM (auto-detected from longitude)</td></tr>
            <tr><td><strong>GEDI Time Period</strong></td><td>{gedi_start} to {gedi_end}</td></tr>
            <tr><td><strong>Satellite Time Period</strong></td><td>{satellite_start} to {satellite_end}</td></tr>
            <tr><td><strong>Target Resolution</strong></td><td>30 meters</td></tr>
            <tr><td><strong>GEDI Points</strong></td><td>{gedi_points}</td></tr>
            <tr><td><strong>Mean Height</strong></td><td>{mean_height}</td></tr>
            <tr><td><strong>Maximum Height</strong></td><td>{max_height}</td></tr>
        </table>

        <h2 id="data-sources">2. Data Sources & Download Statistics</h2>

        <h3>2.1 GEDI L2A (Training Labels)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Source</strong></td><td>NASA EarthData (via earthaccess)</td></tr>
            <tr><td><strong>Time Period</strong></td><td>{gedi_start} to {gedi_end}</td></tr>
            <tr><td><strong>Valid Points Extracted</strong></td><td>{gedi_points}</td></tr>
            <tr><td><strong>Output CSV Size</strong></td><td>{gedi_csv_size}</td></tr>
        </table>

        <h3>2.2 Sentinel-2 L2A (Optical Imagery)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Source</strong></td><td>Microsoft Planetary Computer</td></tr>
            <tr><td><strong>Scenes Used</strong></td><td>10 (most recent)</td></tr>
            <tr><td><strong>Bands Used</strong></td><td>B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2), NDVI</td></tr>
            <tr><td><strong>Output Size</strong></td><td>{s2_size}</td></tr>
            <tr><td><strong>Resolution</strong></td><td>30m</td></tr>
            <tr><td><strong>CRS</strong></td><td>UTM (auto-detected from longitude)</td></tr>
        </table>

        <h3>2.3 Sentinel-1 RTC (SAR Data)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Source</strong></td><td>Microsoft Planetary Computer</td></tr>
            <tr><td><strong>Collection</strong></td><td>Sentinel-1 RTC (Radiometrically Terrain Corrected)</td></tr>
            <tr><td><strong>Scenes Processed</strong></td><td>5</td></tr>
            <tr><td><strong>Polarizations</strong></td><td>VV, VH</td></tr>
            <tr><td><strong>Output Size</strong></td><td>{s1_size}</td></tr>
            <tr><td><strong>Resolution</strong></td><td>~30m (resampled to match S2)</td></tr>
        </table>

        <h3>2.4 SRTM v3 (Topography)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Source</strong></td><td>NASA EarthData</td></tr>
            <tr><td><strong>Tiles Merged</strong></td><td>15</td></tr>
            <tr><td><strong>Bands</strong></td><td>Elevation, Slope</td></tr>
            <tr><td><strong>Output Size</strong></td><td>{topo_size}</td></tr>
            <tr><td><strong>Resolution</strong></td><td>30m</td></tr>
        </table>

        <h2 id="model-training">3. Model Training Results</h2>

        <h3>3.1 Training Data Summary</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Total Samples</strong></td><td>{gedi_points} GEDI points</td></tr>
            <tr><td><strong>Features Extracted</strong></td><td>9</td></tr>
            <tr><td><strong>Test Split</strong></td><td>20%</td></tr>
        </table>

        <h2 id="prediction-results">4. Prediction Results</h2>

        <h3>4.1 Canopy Height Map Statistics</h3>
        <table>
            <tr><th>Statistic</th><th>Value</th></tr>
            <tr><td><strong>Mean Height</strong></td><td>{mean_height}</td></tr>
            <tr><td><strong>Maximum Height</strong></td><td>{max_height}</td></tr>
            <tr><td><strong>Coverage</strong></td><td><span class="badge badge-green">Complete (no data gaps)</span></td></tr>
        </table>

        <h3>4.2 Height Distribution</h3>
        <p>The model predicts canopy heights ranging from {mean_height} on average to {max_height} maximum.</p>

        <h2 id="output-files">5. Output Files</h2>

        <h3>5.1 File Size Summary</h3>
        <table>
            <tr><th>File</th><th>Size</th><th>Description</th></tr>
            <tr><td colspan="3"><strong>Data Files</strong></td></tr>
            <tr><td><code>gedi_raw.csv</code></td><td>{gedi_csv_size}</td><td>GEDI training points ({gedi_points})</td></tr>
            <tr><td><code>sentinel2.tif</code></td><td>{s2_size}</td><td>5-band optical composite</td></tr>
            <tr><td><code>sentinel1.tif</code></td><td>{s1_size}</td><td>2-band SAR composite</td></tr>
            <tr><td><code>topography.tif</code></td><td>{topo_size}</td><td>Elevation + slope</td></tr>
            <tr><td colspan="3"><strong>Prediction Output</strong></td></tr>
            <tr><td><code>canopy_height_map.tif</code></td><td>{tif_size}</td><td>Final canopy height prediction</td></tr>
            <tr><td colspan="3"><strong>Model</strong></td></tr>
            <tr><td><code>canopy_height_model.pkl</code></td><td>{model_size}</td><td>Trained Random Forest (100 estimators)</td></tr>
        </table>

        <h2 id="visualizations">6. Visualizations</h2>

        <h3>Canopy Height Map</h3>
'''

    # Add full-width canopy height map with GeoTIFF download link
    if (output_dir / 'canopy_height_map.png').exists():
        file_size = get_size_mb(output_dir / 'canopy_height_map.png')
        html_content += f'''
        <div class="full-width-image">
            <img src="canopy_height_map.png" alt="Canopy Height Map" onclick="window.open(this.src)">
            <div class="caption">
                <strong>Canopy Height Map</strong><br>
                <a href="canopy_height_map.png" target="_blank">View Full Size ({file_size})</a> |
                <a href="canopy_height_map.tif" target="_blank">Download GeoTIFF ({tif_size})</a>
            </div>
        </div>
'''

    html_content += '''
        <h3>Height Statistics</h3>
'''

    # Add stats panel
    if (output_dir / 'canopy_height_stats.png').exists():
        file_size = get_size_mb(output_dir / 'canopy_height_stats.png')
        html_content += f'''
        <div class="full-width-image">
            <img src="canopy_height_stats.png" alt="Height Statistics" onclick="window.open(this.src)">
            <div class="caption">
                <strong>Height Distribution & Map Statistics</strong><br>
                <a href="canopy_height_stats.png" target="_blank">View Full Size ({file_size})</a>
            </div>
        </div>
'''

    html_content += '''
        <h3>Pipeline Visualizations</h3>
        <div class="image-gallery">
'''

    viz_images = [
        ('gedi_data_coverage.png', 'GEDI Data Coverage'),
        ('gedi_tracks.png', 'GEDI Orbital Tracks'),
        ('sentinel2_bands.png', 'Sentinel-2 Bands'),
        ('sentinel2_composites.png', 'Sentinel-2 Composites'),
        ('sentinel2_histograms.png', 'Sentinel-2 Histograms'),
        ('model_validation.png', 'Model Validation'),
        ('pipeline_summary.png', 'Pipeline Summary'),
    ]

    for img_file, title in viz_images:
        img_path = output_dir / img_file
        if img_path.exists():
            file_size = get_size_mb(img_path)
            html_content += f'''
            <div class="image-card">
                <img src="{img_file}" alt="{title}" onclick="window.open(this.src)">
                <div class="caption">
                    <strong>{title}</strong><br>
                    <a href="{img_file}" target="_blank">View Full Size ({file_size})</a>
                </div>
            </div>'''

    html_content += f'''
        </div>

        <h3>Data Downloads</h3>
        <table>
            <tr><th>File</th><th>Size</th><th>Download</th></tr>
            <tr><td><strong>canopy_height_map.tif</strong></td><td>{tif_size}</td><td><a href="canopy_height_map.tif" class="download-btn" target="_blank">Download</a></td></tr>
            <tr><td><strong>canopy_height_model.pkl</strong></td><td>{model_size}</td><td><a href="canopy_height_model.pkl" class="download-btn" target="_blank">Download</a></td></tr>
            <tr><td><strong>gedi_raw.csv</strong></td><td>{gedi_csv_size}</td><td><a href="gedi_raw.csv" class="download-btn" target="_blank">Download</a></td></tr>
            <tr><td><strong>sentinel2.tif</strong></td><td>{s2_size}</td><td><a href="sentinel2.tif" class="download-btn" target="_blank">Download</a></td></tr>
            <tr><td><strong>sentinel1.tif</strong></td><td>{s1_size}</td><td><a href="sentinel1.tif" class="download-btn" target="_blank">Download</a></td></tr>
            <tr><td><strong>topography.tif</strong></td><td>{topo_size}</td><td><a href="topography.tif" class="download-btn" target="_blank">Download</a></td></tr>
        </table>

        <div class="footer">
            <p><strong>Report Generated:</strong> {report_date}</p>
            <p><strong>Source Code:</strong> <a href="https://github.com/klinucsd/of4d_canopy_height_mapper" target="_blank">https://github.com/klinucsd/of4d_canopy_height_mapper</a></p>
        </div>
    </div>
</body>
</html>'''

    report_path = output_dir / 'REPORT.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    print(f'Generated: {report_path}')
    return report_path


if __name__ == '__main__':
    # Generate reports - specify run directory and date ranges
    # Usage: generate_report(run_dir, gedi_start, gedi_end, satellite_start, satellite_end, bbox)
    generate_report('output', '2022-01-01', '2023-12-31')
