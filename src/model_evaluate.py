"""
Model evaluation utilities for SmartRent Manhattan project.
Handles evaluation metrics and model performance assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data and compute performance metrics.
    
    Parameters
    ----------
    model : sklearn model
        Trained regression model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test target values.
    
    Returns
    -------
    dict
        Dictionary containing RMSE, MAE, and R² scores.
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print("Model Evaluation Metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return metrics


def plot_residuals(model, X_test, y_test):
    """
    Create residual plots to assess model performance.
    
    Generates two plots:
    1. Scatter plot of actual values vs residuals
    2. Histogram of residuals
    
    Parameters
    ----------
    model : sklearn model
        Trained regression model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test target values.
    
    Returns
    -------
    None
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    axes[0].scatter(y_test, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Actual Rent ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title('Residuals vs Actual Values', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def shap_summary_plot(model, X_train, feature_names, save_path=None):
    """
    Generate SHAP summary plot showing feature importance.
    
    Parameters
    ----------
    model : sklearn model
        Trained tree-based model (RandomForest, XGBoost, or DecisionTree).
    X_train : pd.DataFrame or np.ndarray
        Training features used to compute SHAP values.
    feature_names : list
        List of feature names corresponding to X_train columns.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed only.
    
    Returns
    -------
    None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 8), dpi=100)
    
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    
    plt.title('SHAP Summary Plot - Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.show()


def shap_bar_plot(model, X_train, feature_names, save_path=None):
    """
    Generate SHAP bar plot showing mean absolute SHAP values.
    
    Parameters
    ----------
    model : sklearn model
        Trained tree-based model (RandomForest, XGBoost, or DecisionTree).
    X_train : pd.DataFrame or np.ndarray
        Training features used to compute SHAP values.
    feature_names : list
        List of feature names corresponding to X_train columns.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed only.
    
    Returns
    -------
    None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 8), dpi=100)
    
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    
    plt.title('SHAP Bar Plot - Mean Absolute Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP bar plot saved to {save_path}")
    
    plt.show()


def shap_dependence_plots(model, X_train, feature_names, top_n=5, save_path=None):
    """
    Generate SHAP dependence plots for top N most important features.
    
    Parameters
    ----------
    model : sklearn model
        Trained tree-based model (RandomForest, XGBoost, or DecisionTree).
    X_train : pd.DataFrame or np.ndarray
        Training features used to compute SHAP values.
    feature_names : list
        List of feature names corresponding to X_train columns.
    top_n : int, default=5
        Number of top features to plot.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed only.
    
    Returns
    -------
    None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    mean_abs_shap = np.abs(shap_values).mean(0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    n_cols = 2
    n_rows = (top_n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), dpi=100)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_indices):
        ax = axes[idx]
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_train,
            feature_names=feature_names,
            ax=ax,
            show=False
        )
        ax.set_title(f'SHAP Dependence: {feature_names[feat_idx]}', fontsize=12, fontweight='bold')
    
    for idx in range(top_n, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('SHAP Dependence Plots - Top Features', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP dependence plots saved to {save_path}")
    
    plt.show()


def generate_all_shap_plots(model, X_train, feature_names, output_dir='models/shap'):
    """
    Generate all SHAP plots and save them to the specified directory.
    
    Parameters
    ----------
    model : sklearn model
        Trained tree-based model.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    feature_names : list
        List of feature names.
    output_dir : str, default='models/shap'
        Directory to save all SHAP plots.
    
    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    shap_summary_plot(
        model, X_train, feature_names,
        save_path=os.path.join(output_dir, 'shap_summary.png')
    )
    
    shap_bar_plot(
        model, X_train, feature_names,
        save_path=os.path.join(output_dir, 'shap_bar.png')
    )
    
    shap_dependence_plots(
        model, X_train, feature_names, top_n=5,
        save_path=os.path.join(output_dir, 'shap_dependence.png')
    )
    
    print(f"All SHAP plots generated and saved to {output_dir}")
