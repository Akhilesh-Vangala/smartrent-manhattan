"""
Prediction utilities for SmartRent Manhattan project.
Handles making predictions using trained models.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os


def load_model(path: str):
    """
    Load a trained model from a pickle file.
    
    Parameters
    ----------
    path : str
        Path to the pickle file containing the trained model.
    
    Returns
    -------
    model
        Loaded machine learning model.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_feature_metadata(path: str = None):
    """
    Load feature metadata (feature names and order) from JSON file.
    If path is None or file doesn't exist, returns None (will use reference_df).
    
    Parameters
    ----------
    path : str, optional
        Path to JSON file containing feature metadata.
        Expected format: {"feature_names": ["feature1", "feature2", ...]}
    
    Returns
    -------
    list or None
        List of feature names in the correct order, or None if not available.
    """
    if path is None:
        return None
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('feature_names', None)
    return None


def prepare_input(user_input: dict, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert user input dictionary into a single-row dataframe with all
    transformations applied (matching preprocessing and feature engineering).
    
    Parameters
    ----------
    user_input : dict
        Dictionary containing user-provided features. Expected keys include:
        - bedrooms, bathrooms, floor, size_sqft, building_age_yrs
        - neighborhood (string)
        - amenity flags: no_fee, has_roofdeck, has_washer_dryer, has_doorman,
          has_elevator, has_dishwasher, has_patio, has_gym
        - min_to_subway (optional)
        - rent (optional, for validation only)
    reference_df : pd.DataFrame
        Reference dataframe from training data to ensure feature alignment.
        Used to get all possible feature columns and their order.
    
    Returns
    -------
    pd.DataFrame
        Single-row dataframe with all transformations applied, matching
        the exact feature order and structure used during training.
    """
    df = pd.DataFrame([user_input])
    
    if 'bedrooms' in df.columns:
        df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(0).astype(int)
    else:
        df['bedrooms'] = 0
    
    if 'bathrooms' in df.columns:
        df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').fillna(0).astype(int)
    else:
        df['bathrooms'] = 0
    
    if 'floor' in df.columns:
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce').fillna(0).astype(int)
    else:
        df['floor'] = 0
    
    if 'size_sqft' in df.columns:
        df['size_sqft'] = pd.to_numeric(df['size_sqft'], errors='coerce').fillna(0)
    else:
        df['size_sqft'] = 0
    
    if 'building_age_yrs' in df.columns:
        df['building_age_yrs'] = pd.to_numeric(df['building_age_yrs'], errors='coerce').fillna(0)
    else:
        df['building_age_yrs'] = 0
    
    if 'min_to_subway' in df.columns:
        df['min_to_subway'] = pd.to_numeric(df['min_to_subway'], errors='coerce').fillna(0)
    else:
        df['min_to_subway'] = 0
    
    if 'rent' in df.columns and 'size_sqft' in df.columns and df['size_sqft'].iloc[0] > 0:
        df['price_per_sqft'] = df['rent'] / df['size_sqft']
    elif 'size_sqft' in df.columns and df['size_sqft'].iloc[0] > 0:
        df['price_per_sqft'] = 0.0
    else:
        df['price_per_sqft'] = 0.0
    
    amenity_columns = ['no_fee', 'has_roofdeck', 'has_washer_dryer',
                       'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']
    
    for col in amenity_columns:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].clip(0, 1)
    
    df['amenity_count'] = df[amenity_columns].sum(axis=1)
    
    amenity_cols_all = ['no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman',
                        'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']
    for col in amenity_cols_all:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].clip(0, 1)
    
    if 'neighborhood' in df.columns:
        neighborhood_value = str(df['neighborhood'].iloc[0])
        neighborhood_dummies = pd.get_dummies([neighborhood_value], prefix='neighborhood')
        
        reference_neighborhood_cols = [col for col in reference_df.columns if col.startswith('neighborhood_')]
        for col in reference_neighborhood_cols:
            if col in neighborhood_dummies.columns:
                df[col] = neighborhood_dummies[col].iloc[0]
            else:
                df[col] = 0
        
        df = df.drop('neighborhood', axis=1)
    else:
        reference_neighborhood_cols = [col for col in reference_df.columns if col.startswith('neighborhood_')]
        for col in reference_neighborhood_cols:
            df[col] = 0
    
    if 'building_age_yrs' in df.columns:
        age = df['building_age_yrs'].iloc[0]
        if age <= 10:
            age_bucket = 'new'
        elif age <= 30:
            age_bucket = 'mid'
        elif age <= 60:
            age_bucket = 'old'
        else:
            age_bucket = 'historic'
        
        reference_age_cols = [col for col in reference_df.columns if col.startswith('age_bucket_')]
        for col in reference_age_cols:
            if col == f'age_bucket_{age_bucket}':
                df[col] = 1
            else:
                df[col] = 0
    else:
        reference_age_cols = [col for col in reference_df.columns if col.startswith('age_bucket_')]
        for col in reference_age_cols:
            df[col] = 0
    
    if 'size_sqft' in df.columns:
        size = df['size_sqft'].iloc[0]
        if size < 400:
            size_category = 'tiny'
        elif size < 650:
            size_category = 'small'
        elif size < 900:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        reference_size_cols = [col for col in reference_df.columns if col.startswith('size_category_')]
        for col in reference_size_cols:
            if col == f'size_category_{size_category}':
                df[col] = 1
            else:
                df[col] = 0
    else:
        reference_size_cols = [col for col in reference_df.columns if col.startswith('size_category_')]
        for col in reference_size_cols:
            df[col] = 0
    
    if 'rent' in df.columns and 'bedrooms' in df.columns:
        df['price_per_br'] = df['rent'] / np.maximum(df['bedrooms'], 1)
    elif 'bedrooms' in df.columns:
        df['price_per_br'] = 0.0
    else:
        df['price_per_br'] = 0.0
    
    df['rent_to_neighborhood_ratio'] = 1.0
    
    if 'amenity_count' in df.columns:
        amenity_min = 0
        amenity_max = len(amenity_columns)
        if amenity_max > amenity_min:
            df['amenity_density_score'] = (df['amenity_count'] - amenity_min) / (amenity_max - amenity_min)
        else:
            df['amenity_density_score'] = 0.0
    else:
        df['amenity_density_score'] = 0.0
    
    if 'rent' in df.columns:
        df['log_rent'] = np.log1p(df['rent'])
    else:
        df['log_rent'] = 0.0
    
    df['neighborhood_avg_rent'] = 0.0
    
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'rent' in numeric_cols:
        numeric_cols.remove('rent')
    
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    
    df = df.fillna(0)
    
    df = df[numeric_cols]
    
    return df


def predict_rent(model, X: pd.DataFrame) -> float:
    """
    Make a rental price prediction using the trained model.
    
    Parameters
    ----------
    model : sklearn model
        Trained regression model.
    X : pd.DataFrame
        Single-row dataframe with features prepared by prepare_input().
    
    Returns
    -------
    float
        Predicted rent value, rounded to the nearest dollar.
    """
    predictions = model.predict(X)
    if isinstance(predictions, np.ndarray):
        prediction = float(predictions.flat[0])
    else:
        prediction = float(predictions)
    return round(prediction)
