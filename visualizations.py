"""
Visualization Module for Canopy Height Mapping Pipeline
Provides plotting functions for each part of the pipeline similar to ICESat2VegR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# PART 1: GEDI DATA VISUALIZATION
# ============================================================================

def plot_gedi_data_coverage(gedi_csv, bbox=None, output_dir=None):
    """
    Visualize GEDI point data coverage and distribution

    Parameters:
    -----------
    gedi_csv : str
        Path to GEDI CSV file
    bbox : list, optional
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    output_dir : str, optional
        Directory to save plots

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    gedi = pd.read_csv(gedi_csv)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Spatial distribution map
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(gedi['longitude'], gedi['latitude'],
                         c=gedi['rh98'], cmap='RdYlGn',
                         s=1, alpha=0.5, vmin=0, vmax=50)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GEDI Point Distribution (colored by height)')
    plt.colorbar(scatter, ax=ax1, label='RH98 (m)')

    if bbox:
        rect = mpatches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2]-bbox[0], bbox[3]-bbox[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)

    # 2. RH98 histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(gedi['rh98'], bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('RH98 (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Canopy Height Distribution\n(n={len(gedi)} points, mean={gedi["rh98"].mean():.2f}m)')
    ax2.axvline(gedi['rh98'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gedi["rh98"].mean():.2f}m')
    ax2.legend()

    # 3. RH98 box plot by synthetic flag
    ax3 = fig.add_subplot(gs[0, 2])
    if 'synthetic' in gedi.columns:
        gedi.boxplot(column='rh98', by='synthetic', ax=ax3)
        ax3.set_title('Real vs Synthetic Points')
        ax3.set_xlabel('Synthetic')
        ax3.set_ylabel('RH98 (m)')
    else:
        ax3.text(0.5, 0.5, 'No synthetic augmentation',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Synthetic Augmentation')
    plt.suptitle('')  # Remove automatic suptitle

    # 4. Longitude distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(gedi['longitude'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Longitude Distribution')

    # 5. Latitude distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(gedi['latitude'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Latitude')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Latitude Distribution')

    # 6. Density heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    from scipy.stats import gaussian_kde
    x = gedi['longitude']
    y = gedi['latitude']
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    scatter = ax6.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    ax6.set_title('Point Density')
    plt.colorbar(scatter, ax=ax6, label='Density')

    fig.suptitle('GEDI L2A Data Overview', fontsize=16, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/gedi_data_coverage.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/gedi_data_coverage.png")

    return fig


def plot_gedi_tracks(gedi_csv, output_dir=None):
    """
    Plot GEDI tracks colored by beam
    """
    gedi = pd.read_csv(gedi_csv)

    fig, ax = plt.subplots(figsize=(12, 8))

    if 'beam' in gedi.columns:
        beams = gedi['beam'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(beams)))

        for i, beam in enumerate(beams):
            beam_data = gedi[gedi['beam'] == beam]
            ax.scatter(beam_data['longitude'], beam_data['latitude'],
                      c=[colors[i]], s=1, alpha=0.5, label=beam)
        ax.legend()
    else:
        ax.scatter(gedi['longitude'], gedi['latitude'],
                  c=gedi['rh98'], cmap='RdYlGn', s=1, alpha=0.5)
        plt.colorbar(ax.collections[0], label='RH98 (m)')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GEDI Track Visualization')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/gedi_tracks.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/gedi_tracks.png")

    return fig


# ============================================================================
# PART 2: SATELLITE DATA VISUALIZATION
# ============================================================================

def plot_sentinel2_bands(s2_tif, output_dir=None):
    """
    Visualize Sentinel-2 bands and indices

    Parameters:
    -----------
    s2_tif : str
        Path to Sentinel-2 TIFF file
    output_dir : str, optional
        Directory to save plots
    """
    import rasterio

    with rasterio.open(s2_tif) as src:
        data = src.read()
        descriptions = src.descriptions

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Plot each band
    for i in range(min(data.shape[0], 8)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        im = ax.imshow(data[i], cmap='viridis')
        ax.set_title(descriptions[i] if i < len(descriptions) else f'Band {i+1}')
        plt.colorbar(im, ax=ax)

    fig.suptitle('Sentinel-2 Bands and Indices', fontsize=16, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/sentinel2_bands.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/sentinel2_bands.png")

    return fig


def plot_sentinel2_composite(s2_tif, bbox=None, output_dir=None):
    """
    Create RGB and False Color composites from Sentinel-2

    Parameters:
    -----------
    s2_tif : str
        Path to Sentinel-2 TIFF file
    bbox : list, optional
        Bounding box for cropping
    output_dir : str, optional
        Directory to save plots
    """
    import rasterio

    with rasterio.open(s2_tif) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB composite (if we have B04, B03, B02 equivalent)
    # Using B04, B08, B11 for visualization as alternative
    if data.shape[0] >= 3:
        # Natural color-ish
        rgb = np.stack([data[0], data[1], data[2]])  # B04, B08, B11

        for i, band in enumerate(rgb):
            rgb[i] = (band - band.min()) / (band.max() - band.min() + 1e-10)

        # Apply gamma correction
        rgb = np.power(rgb, 1/2.2)

        axes[0].imshow(np.moveaxis(rgb, 0, -1))
        axes[0].set_title('RGB Composite (B04, B08, B11)')

        # False color NIR
        nir_rgb = np.stack([data[1], data[0], data[2]])  # B08, B04, B11

        for i, band in enumerate(nir_rgb):
            nir_rgb[i] = np.clip(band / 0.5, 0, 1)

        axes[1].imshow(np.moveaxis(nir_rgb, 0, -1))
        axes[1].set_title('False Color (NIR, Red, Green)')

        # NDVI
        if data.shape[0] >= 5:  # NDVI band exists
            ndvi = data[4]
            im = axes[2].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1)
            axes[2].set_title('NDVI')
            plt.colorbar(im, ax=axes[2])
        else:
            # Calculate NDVI
            ndvi = (data[1] - data[0]) / (data[1] + data[0] + 1e-10)
            im = axes[2].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1)
            axes[2].set_title('NDVI (calculated)')
            plt.colorbar(im, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')

    fig.suptitle(f'Sentinel-2 Composites\nCRS: {crs}', fontsize=14, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/sentinel2_composites.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/sentinel2_composites.png")

    return fig


def plot_band_histograms(s2_tif, output_dir=None):
    """
    Plot histograms of all Sentinel-2 bands
    """
    import rasterio

    with rasterio.open(s2_tif) as src:
        data = src.read()
        descriptions = src.descriptions

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(data.shape[0], 6)):
        ax = axes[i]
        band_data = data[i].flatten()
        band_data = band_data[np.isfinite(band_data)]

        ax.hist(band_data, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(descriptions[i] if i < len(descriptions) else f'Band {i+1}')
        ax.set_xlabel('Reflectance')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sentinel-2 Band Histograms', fontsize=14, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/sentinel2_histograms.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/sentinel2_histograms.png")

    return fig


# ============================================================================
# PART 3: MODEL TRAINING VISUALIZATION
# ============================================================================

def plot_model_validation(y_true, y_pred, feature_names=None, feature_importance=None,
                          model_name="Random Forest", output_dir=None):
    """
    Generate comprehensive model validation plots

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    feature_names : list, optional
        Names of features
    feature_importance : array-like, optional
        Feature importance scores
    model_name : str
        Name of the model
    output_dir : str, optional
        Directory to save plots
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Predicted vs Observed scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Add 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax1.plot(x_line, p(x_line), 'b-', linewidth=2, label=f'Regression (y={z[0]:.2f}x+{z[1]:.2f})')

    ax1.set_xlabel('Observed Height (m)', fontsize=12)
    ax1.set_ylabel('Predicted Height (m)', fontsize=12)
    ax1.set_title(f'Predicted vs Observed\nR² = {r2:.3f} | RMSE = {rmse:.2f}m | MAE = {mae:.2f}m',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals plot
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Height (m)', fontsize=12)
    ax2.set_ylabel('Residuals (m)', fontsize=12)
    ax2.set_title(f'Residuals Plot\nBias = {bias:.2f}m', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Residuals histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(residuals):.2f}m')
    ax3.set_xlabel('Residual (m)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Observed vs Predicted by height bins
    ax4 = fig.add_subplot(gs[1, 0])
    # Create bins
    n_bins = 10
    bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)

    valid_bin_centers = []
    obs_means = []
    pred_means = []
    stds = []

    for i in range(n_bins):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() > 0:
            bin_center = (bins[i] + bins[i + 1]) / 2
            valid_bin_centers.append(bin_center)
            obs_means.append(y_true[mask].mean())
            pred_means.append(y_pred[mask].mean())
            stds.append(y_pred[mask].std())

    ax4.errorbar(valid_bin_centers, pred_means, yerr=stds, fmt='o-', capsize=5,
                color='blue', label='Predicted ± 1 SD')
    ax4.plot(valid_bin_centers, valid_bin_centers, 'r--', linewidth=2, label='1:1 line')
    ax4.set_xlabel('Observed Height (m)', fontsize=12)
    ax4.set_ylabel('Predicted Height (m)', fontsize=12)
    ax4.set_title('Binned Predictions', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Error by height
    ax5 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(residuals)
    ax5.scatter(y_true, abs_errors, alpha=0.5, s=10)

    # Add trend line
    from scipy.stats import binned_statistic
    bin_stat = binned_statistic(y_true, abs_errors, statistic='mean', bins=15)
    bin_centers = (bin_stat.bin_edges[:-1] + bin_stat.bin_edges[1:]) / 2
    ax5.plot(bin_centers, bin_stat.statistic, 'r-', linewidth=2, label='Mean error')

    ax5.set_xlabel('Observed Height (m)', fontsize=12)
    ax5.set_ylabel('Absolute Error (m)', fontsize=12)
    ax5.set_title('Error by Height Class', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Feature importance
    ax6 = fig.add_subplot(gs[1, 2])
    if feature_importance is not None and feature_names is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)

        # Top 15 features
        importance_df = importance_df.tail(15)

        y_pos = np.arange(len(importance_df))
        ax6.barh(y_pos, importance_df['importance'], color='forestgreen')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(importance_df['feature'], fontsize=9)
        ax6.set_xlabel('Importance', fontsize=12)
        ax6.set_title('Feature Importance (Top 15)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
    else:
        ax6.text(0.5, 0.5, 'Feature importance\nnot provided',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Feature Importance', fontsize=12, fontweight='bold')

    fig.suptitle(f'{model_name} Model Validation', fontsize=16, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/model_validation.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/model_validation.png")

    return fig


def plot_cross_validation(cv_scores, model_name="Random Forest", output_dir=None):
    """
    Plot cross-validation results

    Parameters:
    -----------
    cv_scores : dict
        Dictionary with 'test_r2', 'test_rmse', etc. arrays
    model_name : str
        Name of the model
    output_dir : str, optional
        Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² scores
    ax1 = axes[0]
    ax1.bar(range(len(cv_scores['test_r2'])), cv_scores['test_r2'],
            color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axhline(y=np.mean(cv_scores['test_r2']), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(cv_scores["test_r2"]):.3f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('R²')
    ax1.set_title(f'Cross-Validation R² Scores\nMean: {np.mean(cv_scores["test_r2"]):.3f} ± {np.std(cv_scores["test_r2"]):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # RMSE scores
    ax2 = axes[1]
    rmse = np.sqrt(cv_scores['test_neg_mean_squared_error'] * -1)
    ax2.bar(range(len(rmse)), rmse, color='coral', edgecolor='black', alpha=0.7)
    ax2.axhline(y=np.mean(rmse), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(rmse):.2f}m')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('RMSE (m)')
    ax2.set_title(f'Cross-Validation RMSE\nMean: {np.mean(rmse):.2f} ± {np.std(rmse):.2f}m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} Cross-Validation Results', fontsize=14, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/cross_validation.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/cross_validation.png")

    return fig


# ============================================================================
# PART 4: FINAL MAP VISUALIZATION
# ============================================================================

def plot_canopy_height_map(height_map_path, gedi_csv=None, title="Canopy Height Map",
                           cmap='YlGnBu', output_dir=None):  # Changed default to YlGnBu (Yellow-Green-Blue)
    """
    Visualize the final canopy height map

    Reprojects height map to WGS84 for proper lat/lon display with GEDI overlay.

    Parameters:
    -----------
    height_map_path : str
        Path to canopy height GeoTIFF
    gedi_csv : str, optional
        Path to GEDI CSV for overlay
    title : str
        Map title
    cmap : str
        Colormap name
    output_dir : str, optional
        Directory to save plots
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from pyproj import CRS

    with rasterio.open(height_map_path) as src:
        height_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height

    # Reproject to WGS84 for lat/lon display
    dst_crs = CRS.from_epsg(4326)  # WGS84
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src.bounds)

    # Create destination array for reprojected data
    reprojected = np.zeros((height, width), dtype=np.float32)

    # Reproject the height map
    with rasterio.open(height_map_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear)

    # Calculate extent in lat/lon
    # Transform pixel coordinates to geographic coordinates
    min_lon, min_lat = transform * (0, 0)
    max_lon, max_lat = transform * (width, height)
    extent = [min_lon, max_lon, min_lat, max_lat]

    # Create full-width map only
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(reprojected, extent=extent, cmap=cmap, vmin=0, vmax=50, origin='lower')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{title}\nCRS: WGS84 (Lat/Lon)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Canopy Height (m)', fontsize=12)
    ax.grid(True, alpha=0.3)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/canopy_height_map.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/canopy_height_map.png")
    plt.close(fig)

    return fig


def plot_canopy_height_stats(height_map_path, title="Canopy Height Statistics", output_dir=None):
    """
    Create a separate statistics panel with histogram and map statistics side by side.

    Parameters:
    -----------
    height_map_path : str
        Path to canopy height GeoTIFF
    title : str
        Panel title
    output_dir : str, optional
        Directory to save plots
    """
    import rasterio

    with rasterio.open(height_map_path) as src:
        height_data = src.read(1)

    # Wider figure for better right padding on stats
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram
    ax1 = axes[0]
    valid_data = height_data[np.isfinite(height_data) & (height_data > 0)]
    ax1.hist(valid_data, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax1.axvline(np.nanmean(height_data), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.nanmean(height_data):.2f}m')
    ax1.axvline(np.nanmedian(height_data), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.nanmedian(height_data):.2f}m')
    ax1.set_xlabel('Canopy Height (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Height Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Statistics
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    stats_text = f"""MAP STATISTICS
==============
Mean Height:      {np.nanmean(height_data):.2f} m
Median Height:    {np.nanmedian(height_data):.2f} m
Std Deviation:    {np.nanstd(height_data):.2f} m
Min Height:       {np.nanmin(height_data):.2f} m
Max Height:       {np.nanmax(height_data):.2f} m

COVERAGE
========
Valid Pixels:     {np.sum(np.isfinite(height_data)):,}
NoData Pixels:    {np.sum(~np.isfinite(height_data)):,}
Total Pixels:     {height_data.size:,}

PERCENTILES
===========
25th percentile:  {np.nanpercentile(height_data, 25):.2f} m
50th percentile:  {np.nanpercentile(height_data, 50):.2f} m
75th percentile:  {np.nanpercentile(height_data, 75):.2f} m
95th percentile:  {np.nanpercentile(height_data, 95):.2f} m"""

    # Place text with proper margins on all sides
    ax2.text(0.05, 0.92, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.3))

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/canopy_height_stats.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/canopy_height_stats.png")
    plt.close(fig)

    return fig


def plot_comparison_maps(predicted_path, reference_path=None, gedi_csv=None,
                         output_dir=None):
    """
    Create comparison plots between predicted and reference data

    Parameters:
    -----------
    predicted_path : str
        Path to predicted height map
    reference_path : str, optional
        Path to reference height map
    gedi_csv : str, optional
        Path to GEDI validation data
    output_dir : str, optional
        Directory to save plots
    """
    import rasterio

    with rasterio.open(predicted_path) as src:
        pred_data = src.read(1)
        pred_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    n_plots = 1
    if reference_path:
        n_plots += 1
    if gedi_csv:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Predicted map
    im = axes[idx].imshow(pred_data, extent=pred_extent, cmap='RdYlGn',
                         vmin=0, vmax=50, origin='lower')
    axes[idx].set_title('Predicted Canopy Height', fontweight='bold')
    plt.colorbar(im, ax=axes[idx], label='Height (m)')
    idx += 1

    # Reference map
    if reference_path:
        with rasterio.open(reference_path) as src:
            ref_data = src.read(1)
        im = axes[idx].imshow(ref_data, extent=pred_extent, cmap='RdYlGn',
                             vmin=0, vmax=50, origin='lower')
        axes[idx].set_title('Reference Canopy Height', fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Height (m)')
        idx += 1

    # Validation scatter
    if gedi_csv:
        gedi = pd.read_csv(gedi_csv)
        # Sample predicted values at GEDI locations
        from rasterio.sample import sample_gen

        with rasterio.open(predicted_path) as src:
            coords = [(lon, lat) for lon, lat in zip(gedi['longitude'], gedi['latitude'])]
            pred_values = list(src.sample(coords))
            pred_values = [v[0] for v in pred_values]

        valid_mask = (np.isfinite(pred_values)) & (gedi['rh98'] > 0)

        axes[idx].scatter(gedi.loc[valid_mask, 'rh98'],
                         np.array(pred_values)[valid_mask],
                         alpha=0.5, s=10)
        axes[idx].plot([0, 50], [0, 50], 'r--', linewidth=2, label='1:1 line')

        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(gedi.loc[valid_mask, 'rh98'], np.array(pred_values)[valid_mask])
        rmse = np.sqrt(mean_squared_error(gedi.loc[valid_mask, 'rh98'],
                                          np.array(pred_values)[valid_mask]))

        axes[idx].set_xlabel('GEDI RH98 (m)')
        axes[idx].set_ylabel('Predicted Height (m)')
        axes[idx].set_title(f'Validation\nR²={r2:.3f}, RMSE={rmse:.2f}m', fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle('Canopy Height Map Comparison', fontsize=14, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/comparison_maps.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/comparison_maps.png")

    return fig


# ============================================================================
# SUMMARY FIGURE
# ============================================================================

def create_pipeline_summary(gedi_csv, s2_tif, height_map_path, model_stats,
                           output_dir=None):
    """
    Create a summary figure showing all pipeline products

    Parameters:
    -----------
    gedi_csv : str
        Path to GEDI CSV
    s2_tif : str
        Path to Sentinel-2 TIFF
    height_map_path : str
        Path to final canopy height map
    model_stats : dict
        Dictionary with 'r2', 'rmse', 'mae'
    output_dir : str, optional
        Directory to save plot
    """
    import rasterio

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # GEDI points
    ax1 = fig.add_subplot(gs[0, 0])
    gedi = pd.read_csv(gedi_csv)
    scatter = ax1.scatter(gedi['longitude'], gedi['latitude'],
                         c=gedi['rh98'], cmap='RdYlGn', s=2, alpha=0.5, vmin=0, vmax=50)
    ax1.set_title('GEDI Training Data', fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax1, label='Height (m)')

    # Sentinel-2 RGB
    ax2 = fig.add_subplot(gs[0, 1])
    with rasterio.open(s2_tif) as src:
        s2_data = src.read()
        ndvi = s2_data[4] if s2_data.shape[0] >= 5 else (s2_data[1] - s2_data[0]) / (s2_data[1] + s2_data[0] + 1e-10)
    im = ax2.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1)
    ax2.set_title('Sentinel-2 NDVI', fontweight='bold')
    plt.colorbar(im, ax=ax2)

    # Model statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    stats_text = f"""
    MODEL PERFORMANCE
    =================
    R²:               {model_stats['r2']:.3f}
    RMSE:             {model_stats['rmse']:.2f} m
    MAE:              {model_stats['mae']:.2f} m

    TRAINING DATA
    ==============
    GEDI Points:      {len(gedi):,}
    Study Area:       {gedi['longitude'].max()-gedi['longitude'].min():.2f}° x {gedi['latitude'].max()-gedi['latitude'].min():.2f}°
    """
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Feature importance
    ax4 = fig.add_subplot(gs[0, 3])
    if 'feature_importance' in model_stats:
        imp_df = pd.DataFrame(model_stats['feature_importance'])
        imp_df = imp_df.sort_values('importance', ascending=True).tail(10)
        ax4.barh(range(len(imp_df)), imp_df['importance'], color='forestgreen')
        ax4.set_yticks(range(len(imp_df)))
        ax4.set_yticklabels(imp_df['feature'], fontsize=8)
        ax4.set_title('Top Features', fontweight='bold')
        ax4.set_xlabel('Importance')
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnot available',
                ha='center', va='center', transform=ax4.transAxes)

    # Final canopy height map - reproject to WGS84 for lat/lon display
    ax5 = fig.add_subplot(gs[1, :])
    with rasterio.open(height_map_path) as src:
        height_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height

    # Reproject to WGS84
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from pyproj import CRS
    import rasterio
    dst_crs = CRS.from_epsg(4326)
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src.bounds)
    reprojected = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(height_map_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear)

    # Calculate extent in lat/lon
    min_lon, min_lat = transform * (0, 0)
    max_lon, max_lat = transform * (width, height)
    extent = [min_lon, max_lon, min_lat, max_lat]

    im = ax5.imshow(reprojected, extent=extent, cmap='YlGnBu', vmin=0, vmax=50, origin='lower')
    ax5.set_title('Final Canopy Height Map', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax5, fraction=0.02, pad=0.04)
    cbar.set_label('Height (m)', fontsize=12)
    # No GEDI points overlay

    fig.suptitle('Canopy Height Mapping Pipeline Summary',
                fontsize=16, fontweight='bold')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{output_dir}/pipeline_summary.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/pipeline_summary.png")

    return fig


if __name__ == "__main__":
    # Test the module
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("- GEDI: plot_gedi_data_coverage, plot_gedi_tracks")
    print("- Sentinel-2: plot_sentinel2_bands, plot_sentinel2_composite, plot_band_histograms")
    print("- Model: plot_model_validation, plot_cross_validation")
    print("- Maps: plot_canopy_height_map, plot_comparison_maps, create_pipeline_summary")
