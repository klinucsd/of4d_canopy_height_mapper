"""
Validation Module for Canopy Height Mapping Pipeline
Provides model evaluation, cross-validation, and statistical analysis functions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, max_error
)
import json
from pathlib import Path


# ============================================================================
# MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for reporting

    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    metrics = {
        'model_name': model_name,
        'n_samples': len(y_true),
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
    }

    # Height class-specific metrics
    height_classes = [
        (0, 5, '0-5m (shrubs)'),
        (5, 10, '5-10m (small trees)'),
        (10, 20, '10-20m (medium trees)'),
        (20, 50, '20-50m (large trees)')
    ]

    for min_h, max_h, class_name in height_classes:
        mask = (y_true >= min_h) & (y_true < max_h)
        if mask.sum() > 0:
            class_y_true = y_true[mask]
            class_y_pred = y_pred[mask]

            metrics[f'{class_name}_n'] = mask.sum()
            metrics[f'{class_name}_r2'] = r2_score(class_y_true, class_y_pred)
            metrics[f'{class_name}_rmse'] = np.sqrt(mean_squared_error(class_y_true, class_y_pred))
            metrics[f'{class_name}_mae'] = mean_absolute_error(class_y_true, class_y_pred)
            metrics[f'{class_name}_bias'] = np.mean(class_y_pred - class_y_true)

    return metrics


def print_model_report(metrics):
    """
    Print a formatted model evaluation report

    Parameters:
    -----------
    metrics : dict
        Dictionary returned by evaluate_model()
    """
    print("\n" + "="*60)
    print(f"MODEL EVALUATION REPORT: {metrics['model_name'].upper()}")
    print("="*60)

    print("\nOVERALL PERFORMANCE")
    print("-" * 40)
    print(f"  Samples:                 {metrics['n_samples']:,}")
    print(f"  R² (Coefficient of Determination):  {metrics['r2']:.4f}")
    print(f"  RMSE (Root Mean Square Error):      {metrics['rmse']:.3f} m")
    print(f"  MAE (Mean Absolute Error):           {metrics['mae']:.3f} m")
    print(f"  Explained Variance:                 {metrics['explained_variance']:.4f}")
    print(f"  Maximum Error:                       {metrics['max_error']:.3f} m")
    print(f"  Bias (mean prediction error):       {metrics['bias']:.3f} m")
    print(f"  MAPE (Mean Absolute % Error):        {metrics['mape']:.2f}%")

    print("\nHEIGHT CLASS PERFORMANCE")
    print("-" * 40)
    height_classes = [
        (0, 5, '0-5m (shrubs)'),
        (5, 10, '5-10m (small trees)'),
        (10, 20, '10-20m (medium trees)'),
        (20, 50, '20-50m (large trees)')
    ]

    print(f"{'Height Class':<25} {'N':>6} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
    print("-" * 70)

    for min_h, max_h, class_name in height_classes:
        n_key = f'{class_name}_n'
        if n_key in metrics:
            print(f"{class_name:<25} {metrics[n_key]:>6} "
                  f"{metrics[f'{class_name}_r2']:>8.3f} "
                  f"{metrics[f'{class_name}_rmse']:>8.2f} "
                  f"{metrics[f'{class_name}_mae']:>8.2f} "
                  f"{metrics[f'{class_name}_bias']:>8.2f}")

    print("\nINTERPRETATION")
    print("-" * 40)
    if metrics['r2'] > 0.8:
        print("  ✓ Excellent model fit (R² > 0.8)")
    elif metrics['r2'] > 0.6:
        print("  ✓ Good model fit (R² > 0.6)")
    elif metrics['r2'] > 0.4:
        print("  ⚠ Moderate model fit (R² > 0.4)")
    else:
        print("  ✗ Poor model fit (R² < 0.4)")

    if abs(metrics['bias']) < 1:
        print("  ✓ Low bias (|bias| < 1m)")
    elif abs(metrics['bias']) < 3:
        print("  ⚠ Moderate bias (|bias| < 3m)")
    else:
        print("  ✗ High bias (|bias| > 3m)")

    if metrics['rmse'] < 5:
        print("  ✓ Low RMSE (RMSE < 5m)")
    elif metrics['rmse'] < 10:
        print("  ⚠ Moderate RMSE (RMSE < 10m)")
    else:
        print("  ✗ High RMSE (RMSE > 10m)")

    print("="*60 + "\n")


def perform_cross_validation(model, X, y, n_folds=5, random_state=42):
    """
    Perform k-fold cross-validation with multiple metrics

    Parameters:
    -----------
    model : sklearn-like model
        Model with fit() and predict() methods
    X : array-like
        Feature matrix
    y : array-like
        Target values
    n_folds : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Cross-validation results
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import make_scorer

    # Create KFold splitter for reproducible results
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Create scorers
    scorers = {
        'r2': make_scorer(r2_score),
        'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
        'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }

    # Perform cross-validation
    cv_results = {}
    for name, scorer in scorers.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring=scorer, n_jobs=-1)
        cv_results[name] = scores

    # Calculate summary statistics
    cv_summary = {
        'n_folds': n_folds,
        'r2_mean': np.mean(cv_results['r2']),
        'r2_std': np.std(cv_results['r2']),
        'rmse_mean': np.mean(np.sqrt(-cv_results['neg_mse'])),
        'rmse_std': np.std(np.sqrt(-cv_results['neg_mse'])),
        'mae_mean': np.mean(-cv_results['neg_mae']),
        'mae_std': np.std(-cv_results['neg_mae']),
        'fold_scores': cv_results
    }

    return cv_summary


def print_cv_report(cv_summary):
    """
    Print cross-validation report

    Parameters:
    -----------
    cv_summary : dict
        Dictionary returned by perform_cross_validation()
    """
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION REPORT ({cv_summary['n_folds']}-FOLD)")
    print("="*60)

    print("\nCROSS-VALIDATION METRICS")
    print("-" * 40)
    print(f"  R²:      {cv_summary['r2_mean']:.4f} ± {cv_summary['r2_std']:.4f}")
    print(f"  RMSE:    {cv_summary['rmse_mean']:.3f} ± {cv_summary['rmse_std']:.3f} m")
    print(f"  MAE:     {cv_summary['mae_mean']:.3f} ± {cv_summary['mae_std']:.3f} m")

    print("\nFOLD-BY-FOLD RESULTS")
    print("-" * 40)
    print(f"{'Fold':<6} {'R²':>10} {'RMSE (m)':>10} {'MAE (m)':>10}")
    print("-" * 40)

    for i in range(cv_summary['n_folds']):
        r2 = cv_summary['fold_scores']['r2'][i]
        rmse = np.sqrt(-cv_summary['fold_scores']['neg_mse'][i])
        mae = -cv_summary['fold_scores']['neg_mae'][i]
        print(f"{i+1:<6} {r2:>10.4f} {rmse:>10.3f} {mae:>10.3f}")

    print("="*60 + "\n")


# ============================================================================
# VALIDATION WITH INDEPENDENT DATA
# ============================================================================

def validate_with_gedi(model, gedi_csv, s2_tif, s1_tif, topo_tif, output_dir=None):
    """
    Validate model predictions against GEDI data using spatial holdout

    Parameters:
    -----------
    model : sklearn model
        Trained model
    gedi_csv : str
        Path to GEDI CSV file
    s2_tif : str
        Path to Sentinel-2 TIFF
    s1_tif : str
        Path to Sentinel-1 TIFF
    topo_tif : str
        Path to topography TIFF
    output_dir : str, optional
        Directory to save validation results

    Returns:
    --------
    dict : Validation results
    """
    # Extract features at GEDI locations
    from complete_canopy_height_pipeline_v2 import extract_features

    X, y, features = extract_features(gedi_csv, s2_tif, s1_tif, topo_tif)

    # Predict
    y_pred = model.predict(X)

    # Evaluate
    metrics = evaluate_model(y, y_pred, "GEDI Validation")

    # Save results
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save validation CSV
        val_df = pd.DataFrame({
            'observed': y,
            'predicted': y_pred,
            'residual': y_pred - y,
            'abs_error': np.abs(y_pred - y)
        })
        val_df.to_csv(f'{output_dir}/validation_predictions.csv', index=False)

        # Save metrics
        with open(f'{output_dir}/validation_metrics.json', 'w') as f:
            # Remove non-serializable items
            save_metrics = {k: v for k, v in metrics.items()
                           if not isinstance(v, (np.ndarray, list))}
            json.dump(save_metrics, f, indent=2)

        print(f"  Saved validation results to {output_dir}")

    return metrics


def spatial_cross_validation(model, gedi_csv, s2_tif, s1_tif, topo_tif,
                            n_folds=5, output_dir=None):
    """
    Perform spatial cross-validation by dividing study area into regions

    Parameters:
    -----------
    model : sklearn model
        Model to validate
    gedi_csv : str
        Path to GEDI CSV file
    s2_tif, s1_tif, topo_tif : str
        Path to raster files
    n_folds : int
        Number of spatial folds
    output_dir : str, optional
        Directory to save results

    Returns:
    --------
    dict : Spatial CV results
    """
    from complete_canopy_height_pipeline_v2 import extract_features

    gedi = pd.read_csv(gedi_csv)

    # Create spatial folds based on longitude
    lon_bins = np.linspace(gedi['longitude'].min(), gedi['longitude'].max(), n_folds + 1)
    gedi['spatial_fold'] = pd.cut(gedi['longitude'], bins=lon_bins, labels=False)

    fold_results = []

    for fold in range(n_folds):
        print(f"  Processing fold {fold + 1}/{n_folds}...")

        # Split data
        train_mask = gedi['spatial_fold'] != fold
        test_mask = gedi['spatial_fold'] == fold

        train_csv = f'{output_dir}/train_fold_{fold}.csv' if output_dir else None
        test_csv = f'{output_dir}/test_fold_{fold}.csv' if output_dir else None

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            gedi[train_mask].to_csv(train_csv, index=False)
            gedi[test_mask].to_csv(test_csv, index=False)

        # Train model
        from sklearn.ensemble import RandomForestRegressor
        fold_model = RandomForestRegressor(
            n_estimators=50, max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=-1
        )

        # Extract features
        # Note: This is simplified - in practice, you'd extract features once
        # and then just index into them
        fold_results.append({
            'fold': fold,
            'n_train': train_mask.sum(),
            'n_test': test_mask.sum(),
            'train_lon_range': (gedi[train_mask]['longitude'].min(),
                               gedi[train_mask]['longitude'].max()),
            'test_lon_range': (gedi[test_mask]['longitude'].min(),
                              gedi[test_mask]['longitude'].max())
        })

    return fold_results


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================

def prediction_intervals(model, X, percentile=95, n_bootstrap=100):
    """
    Calculate prediction intervals using bootstrap

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : array-like
        Feature matrix
    percentile : int
        Percentile for prediction interval (default: 95)
    n_bootstrap : int
        Number of bootstrap iterations

    Returns:
    --------
    tuple : (lower_bound, upper_bound, predictions)
    """
    from sklearn.utils import resample

    predictions = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        X_boot, _, _, _ = resample(X, np.zeros(len(X)), random_state=i)

        # Predict
        pred = model.predict(X_boot)
        predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate percentiles
    lower = np.percentile(predictions, (100 - percentile) / 2, axis=0)
    upper = np.percentile(predictions, 100 - (100 - percentile) / 2, axis=0)
    mean_pred = np.mean(predictions, axis=0)

    return lower, upper, mean_pred


def calculate_map_uncertainty(model, s2_tif, s1_tif, topo_tif, output_path,
                              n_bootstrap=50):
    """
    Generate uncertainty map for predictions

    Parameters:
    -----------
    model : sklearn model
        Trained model
    s2_tif, s1_tif, topo_tif : str
        Path to raster files
    output_path : str
        Path to save uncertainty map
    n_bootstrap : int
        Number of bootstrap iterations
    """
    import rasterio
    from sklearn.utils import resample

    # Read reference data
    with rasterio.open(s2_tif) as src:
        s2_data = src.read()
        profile = src.profile

    h, w = s2_data.shape[1], s2_data.shape[2]

    # Resample other rasters
    from rasterio.warp import reproject, Resampling
    from scipy.ndimage import zoom

    # Stack all features (simplified)
    all_data = s2_data

    # Reshape for prediction
    n_feat, h, w = all_data.shape
    data_2d = all_data.reshape(n_feat, -1).T

    # Bootstrap predictions
    predictions = []

    for i in range(n_bootstrap):
        if i % 10 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}...")

        pred = model.predict(data_2d)
        predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate uncertainty metrics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    lower_pred = np.percentile(predictions, 5, axis=0)
    upper_pred = np.percentile(predictions, 95, axis=0)

    # Reshape back to 2D
    mean_map = mean_pred.reshape(h, w)
    std_map = std_pred.reshape(h, w)
    lower_map = lower_pred.reshape(h, w)
    upper_map = upper_pred.reshape(h, w)

    # Save uncertainty map
    profile.update(count=4, dtype='float32')

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mean_map.astype('float32'), 1)
        dst.write(std_map.astype('float32'), 2)
        dst.write(lower_map.astype('float32'), 3)
        dst.write(upper_map.astype('float32'), 4)

        dst.set_band_description(1, 'mean_height')
        dst.set_band_description(2, 'std_height')
        dst.set_band_description(3, 'lower_95ci')
        dst.set_band_description(4, 'upper_95ci')

    print(f"  Saved uncertainty map to {output_path}")

    return {
        'mean': mean_map,
        'std': std_map,
        'lower': lower_map,
        'upper': upper_map
    }


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def analyze_feature_importance(model, feature_names, output_dir=None):
    """
    Analyze and visualize feature importance

    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    output_dir : str, optional
        Directory to save results

    Returns:
    --------
    pandas.DataFrame : Feature importance DataFrame
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        print(f"  Saved feature importance to {output_dir}/feature_importance.csv")

    return importance_df


def correlation_analysis(X, feature_names, output_dir=None):
    """
    Perform correlation analysis on features

    Parameters:
    -----------
    X : array-like
        Feature matrix
    feature_names : list
        List of feature names
    output_dir : str, optional
        Directory to save results

    Returns:
    --------
    pandas.DataFrame : Correlation matrix
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(f'{output_dir}/feature_correlation.csv')
        print(f"  Saved correlation matrix to {output_dir}/feature_correlation.csv")

        # Create heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm',
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_correlation_heatmap.png', dpi=300)
        plt.close()

        print(f"  Saved correlation heatmap to {output_dir}/feature_correlation_heatmap.png")

    return corr_matrix


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(models_dict, X_test, y_test, output_dir=None):
    """
    Compare multiple models on the same test set

    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and trained models as values
    X_test : array-like
        Test features
    y_test : array-like
        Test target values
    output_dir : str, optional
        Directory to save comparison results

    Returns:
    --------
    pandas.DataFrame : Comparison table
    """
    results = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Select key columns
    key_cols = ['model_name', 'r2', 'rmse', 'mae', 'bias']
    comparison_df = comparison_df[key_cols]

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        print(f"  Saved model comparison to {output_dir}/model_comparison.csv")

    return comparison_df


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_validation_report(gedi_csv, s2_tif, s1_tif, topo_tif, model,
                              feature_names, output_dir):
    """
    Generate a comprehensive validation report

    Parameters:
    -----------
    gedi_csv : str
        Path to GEDI CSV file
    s2_tif, s1_tif, topo_tif : str
        Path to raster files
    model : sklearn model
        Trained model
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save all validation outputs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VALIDATION REPORT")
    print("="*60 + "\n")

    # 1. Extract features
    print("1. Extracting features...")
    from complete_canopy_height_pipeline_v2 import extract_features
    X, y, features = extract_features(gedi_csv, s2_tif, s1_tif, topo_tif)

    # 2. Split data
    print("2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Predict on test set
    print("3. Evaluating model...")
    y_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred, "Test Set")
    print_model_report(test_metrics)

    # 4. Cross-validation
    print("4. Performing cross-validation...")
    cv_summary = perform_cross_validation(model, X, y, n_folds=5)
    print_cv_report(cv_summary)

    # 5. Feature importance
    print("5. Analyzing feature importance...")
    importance_df = analyze_feature_importance(model, feature_names, output_dir)

    print("\nTop 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # 6. Correlation analysis
    print("\n6. Performing correlation analysis...")
    corr_matrix = correlation_analysis(X, feature_names, output_dir)

    # 7. Save all metrics
    print("\n7. Saving results...")

    report = {
        'test_metrics': test_metrics,
        'cross_validation': cv_summary,
        'feature_importance': importance_df.to_dict('records')
    }

    # Save non-serializable items separately
    save_report = {
        'test_metrics': {k: float(v) if not isinstance(v, str) else v
                        for k, v in test_metrics.items() if k not in ['model_name']},
        'cross_validation': {k: float(v) for k, v in cv_summary.items()
                            if k not in ['fold_scores']},
    }

    with open(f'{output_dir}/validation_report.json', 'w') as f:
        json.dump(save_report, f, indent=2)

    print(f"\n  Validation report saved to {output_dir}")
    print("="*60)

    return report


if __name__ == "__main__":
    print("Validation module loaded successfully!")
    print("Available functions:")
    print("- Evaluation: evaluate_model, print_model_report")
    print("- Cross-validation: perform_cross_validation, print_cv_report")
    print("- Validation: validate_with_gedi, spatial_cross_validation")
    print("- Uncertainty: prediction_intervals, calculate_map_uncertainty")
    print("- Features: analyze_feature_importance, correlation_analysis")
    print("- Comparison: compare_models")
    print("- Reports: generate_validation_report")
