"""
Streamlit Model Interpretation page for SmartRent Manhattan dashboard.
Displays model interpretability insights and feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predict import load_model


def load_processed_data():
    """
    Load the processed dataset from the data/processed directory.
    
    Returns
    -------
    pd.DataFrame
        Processed Manhattan rental dataset.
    """
    data_path = os.path.join(project_root, 'data', 'processed', 'cleaned_manhattan.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    else:
        st.error(f"Dataset not found at {data_path}")
        return None


def prepare_feature_matrix(df):
    """
    Prepare feature matrix X by dropping the target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with target column.
    
    Returns
    -------
    tuple
        (X, feature_names) - Feature matrix and list of feature names.
    """
    X = df.copy()
    
    # Drop target column if it exists
    if 'rent' in X.columns:
        X = X.drop(columns=['rent'])
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Ensure numeric types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_names = [name for name in feature_names if name in numeric_cols]
    
    # Fill any missing values
    X = X.fillna(0)
    
    return X, feature_names


def create_shap_summary_plot_streamlit(explainer, X, feature_names):
    """
    Create SHAP summary plot for Streamlit display.
    
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    feature_names : list
        List of feature names.
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.
    """
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 8), dpi=100)
    
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    plt.title('SHAP Summary Plot - Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    fig = plt.gcf()
    return fig


def create_shap_bar_plot_streamlit(explainer, X, feature_names):
    """
    Create SHAP bar plot for Streamlit display.
    
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    feature_names : list
        List of feature names.
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.
    """
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 8), dpi=100)
    
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    
    plt.title('SHAP Bar Plot - Mean Absolute Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    fig = plt.gcf()
    return fig


def create_shap_dependence_plots_streamlit(explainer, X, feature_names, top_n=5):
    """
    Create SHAP dependence plots for Streamlit display.
    
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    feature_names : list
        List of feature names.
    top_n : int, default=5
        Number of top features to plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.
    """
    shap_values = explainer.shap_values(X)
    
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
            X,
            feature_names=feature_names,
            ax=ax,
            show=False
        )
        ax.set_title(f'SHAP Dependence: {feature_names[feat_idx]}', fontsize=12, fontweight='bold')
    
    for idx in range(top_n, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('SHAP Dependence Plots - Top Features', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def get_top_features_insights(explainer, X, feature_names, top_n=5):
    """
    Get insights about top features from SHAP values.
    
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    feature_names : list
        List of feature names.
    top_n : int, default=5
        Number of top features to analyze.
    
    Returns
    -------
    dict
        Dictionary with top features and their mean absolute SHAP values.
    """
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    
    top_features = {}
    for idx in top_indices:
        top_features[feature_names[idx]] = mean_abs_shap[idx]
    
    return top_features


def main():
    """
    Main function to render the Streamlit Interpretability page.
    """
    st.title("Model Interpretability")
    st.subheader("Understand how the model determines Manhattan rental prices.")
    
    st.divider()
    
    # Load data and model
    with st.spinner("Loading data and model..."):
        df = load_processed_data()
        
        if df is None:
            st.stop()
        
        model_path = os.path.join(project_root, 'models', 'best_model.pkl')
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please train the model first.")
            st.stop()
        
        model = load_model(model_path)
        
        # Prepare feature matrix
        X, feature_names = prepare_feature_matrix(df)
    
    st.success("Data and model loaded successfully!")
    
    st.divider()
    
    # Create SHAP explainer
    st.header("SHAP Explainer")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values explain the output of machine learning models 
    by showing how each feature contributes to the prediction. This helps us understand which 
    factors most influence rental prices in Manhattan.
    """)
    
    with st.spinner("Computing SHAP values (this may take a moment)..."):
        # Use a sample for faster computation
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    
    st.info(f"SHAP values computed for {len(X_sample):,} properties. Using a sample for faster computation.")
    
    st.divider()
    
    # Feature Importance Summary
    st.header("Feature Importance Summary")
    st.markdown("""
    The summary plot below shows how each feature impacts rental price predictions. Features are 
    ranked by their average impact, with red indicating higher feature values and blue indicating 
    lower values.
    """)
    
    try:
        fig_summary = create_shap_summary_plot_streamlit(explainer, X_sample, feature_names)
        st.pyplot(fig_summary)
        plt.close(fig_summary)
    except Exception as e:
        st.error(f"Error creating summary plot: {str(e)}")
    
    st.divider()
    
    # Top Feature Impact (Bar Plot)
    st.header("Top Feature Impact (Bar Plot)")
    st.markdown("""
    This bar plot shows the mean absolute SHAP value for each feature, indicating the average 
    magnitude of each feature's impact on predictions.
    """)
    
    try:
        fig_bar = create_shap_bar_plot_streamlit(explainer, X_sample, feature_names)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    except Exception as e:
        st.error(f"Error creating bar plot: {str(e)}")
    
    st.divider()
    
    # Feature Interaction Exploration
    st.header("Feature Interaction Exploration")
    st.markdown("""
    Dependence plots show how individual features interact with the model's predictions. Each plot 
    displays the relationship between a feature's value and its SHAP value, revealing non-linear 
    relationships and interactions.
    """)
    
    top_n = st.slider(
        "Number of top features",
        min_value=1,
        max_value=min(10, len(feature_names)),
        value=5,
        help="Select how many top features to display in dependence plots"
    )
    
    try:
        fig_dependence = create_shap_dependence_plots_streamlit(explainer, X_sample, feature_names, top_n=top_n)
        st.pyplot(fig_dependence)
        plt.close(fig_dependence)
    except Exception as e:
        st.error(f"Error creating dependence plots: {str(e)}")
    
    st.divider()
    
    # Insights Section
    st.header("Key Insights")
    
    # Get top features for insights
    top_features = get_top_features_insights(explainer, X_sample, feature_names, top_n=5)
    
    st.markdown("""
    ### Which Features Impact Rent the Most?
    
    Based on the SHAP analysis, the following features have the strongest impact on rental price predictions:
    """)
    
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        st.write(f"{i}. **{feature}**: {importance:.2f} average absolute SHAP value")
    
    st.divider()
    
    st.markdown("""
    ### How Price per Square Foot Influences Predictions
    
    The `price_per_sqft` feature is typically one of the strongest predictors of rental prices. 
    Higher price per square foot values directly increase the predicted rent, as this metric 
    captures both the size and quality premium of a property. Properties in prime locations with 
    better amenities tend to have higher price per square foot values.
    """)
    
    st.divider()
    
    st.markdown("""
    ### How Neighborhood Affects Rent
    
    Neighborhood features (one-hot encoded) show significant variation in their impact. Premium 
    neighborhoods like Upper East Side, SoHo, and Tribeca typically have positive SHAP values, 
    indicating they add to the predicted rent. More affordable neighborhoods show negative or 
    smaller positive contributions. The model learns these neighborhood premiums from the training data.
    """)
    
    st.divider()
    
    st.markdown("""
    ### How Amenities Shift Predicted Values
    
    Amenities like elevators, doormen, gyms, and roof decks generally have positive SHAP values, 
    meaning they increase the predicted rent. The `amenity_count` feature aggregates these benefits, 
    and properties with more amenities typically command higher rents. However, the impact varies 
    - premium amenities in luxury buildings have a stronger effect than basic amenities.
    """)
    
    st.divider()
    
    st.markdown("""
    ### What SHAP Reveals About Expensive vs. Affordable Areas
    
    SHAP values reveal that expensive areas (high rent predictions) typically have:
    - High price per square foot
    - Premium neighborhood locations
    - Multiple luxury amenities
    - Larger unit sizes
    - Newer or well-maintained buildings
    
    Affordable areas show the opposite pattern - lower price per square foot, fewer amenities, 
    and locations in more budget-friendly neighborhoods. The model captures these patterns through 
    the combination of all features, with location and size being the primary drivers.
    """)
    
    st.info("""
    **Understanding SHAP Values**: 
    - Positive SHAP values increase the predicted rent
    - Negative SHAP values decrease the predicted rent
    - The magnitude indicates the strength of the impact
    - Feature interactions are captured through the model's structure
    """)


if __name__ == "__main__":
    main()
