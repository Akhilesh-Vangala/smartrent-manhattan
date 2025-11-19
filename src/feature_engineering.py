"""
Feature engineering utilities for SmartRent Manhattan project.
Creates derived features and transformations for modeling.
"""

import pandas as pd
import numpy as np
import os


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the preprocessed dataset.
    
    Creates the following features:
    - log_rent: log transformation of rent
    - age_bucket: categorical age groups (new, mid, old, historic)
    - size_category: categorical size groups (tiny, small, medium, large)
    - neighborhood_avg_rent: average rent per neighborhood
    - rent_to_neighborhood_ratio: rent relative to neighborhood average
    - price_per_br: rent per bedroom
    - amenity_density_score: normalized amenity_count (0-1)
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset with cleaned data.
    
    Returns
    -------
    pd.DataFrame
        Dataset with engineered features added.
    """
    df = df.copy()
    
    df['log_rent'] = np.log1p(df['rent'])
    
    df['age_bucket'] = pd.cut(
        df['building_age_yrs'],
        bins=[0, 10, 30, 60, float('inf')],
        labels=['new', 'mid', 'old', 'historic'],
        include_lowest=True
    )
    df['age_bucket'] = df['age_bucket'].astype(str)
    
    df['size_category'] = pd.cut(
        df['size_sqft'],
        bins=[0, 400, 650, 900, float('inf')],
        labels=['tiny', 'small', 'medium', 'large'],
        include_lowest=True
    )
    df['size_category'] = df['size_category'].astype(str)
    
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    
    if neighborhood_cols:
        df['neighborhood_name'] = df[neighborhood_cols].idxmax(axis=1)
        df['neighborhood_name'] = df['neighborhood_name'].str.replace('neighborhood_', '', regex=False)
        
        neighborhood_avg = df.groupby('neighborhood_name')['rent'].mean().reset_index()
        neighborhood_avg.columns = ['neighborhood_name', 'neighborhood_avg_rent']
        df = df.merge(neighborhood_avg, on='neighborhood_name', how='left')
        df['neighborhood_avg_rent'] = df['neighborhood_avg_rent'].fillna(df['rent'].mean())
        
        df['rent_to_neighborhood_ratio'] = df['rent'] / df['neighborhood_avg_rent']
        
        df = df.drop('neighborhood_name', axis=1)
    else:
        df['neighborhood_avg_rent'] = df['rent'].mean()
        df['rent_to_neighborhood_ratio'] = 1.0
    
    df['price_per_br'] = df['rent'] / np.maximum(df['bedrooms'], 1)
    
    if 'amenity_count' in df.columns:
        amenity_min = df['amenity_count'].min()
        amenity_max = df['amenity_count'].max()
        if amenity_max > amenity_min:
            df['amenity_density_score'] = (df['amenity_count'] - amenity_min) / (amenity_max - amenity_min)
        else:
            df['amenity_density_score'] = 0.0
    else:
        df['amenity_density_score'] = 0.0
    
    return df


def encode_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical engineered features.
    
    Encodes the following categorical features:
    - age_bucket
    - size_category
    
    Removes the original categorical columns after encoding to prevent leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with engineered features.
    
    Returns
    -------
    pd.DataFrame
        Dataset with categorical features one-hot encoded.
    """
    df = df.copy()
    
    if 'age_bucket' in df.columns:
        age_dummies = pd.get_dummies(df['age_bucket'], prefix='age_bucket')
        df = pd.concat([df, age_dummies], axis=1)
        df = df.drop('age_bucket', axis=1)
    
    if 'size_category' in df.columns:
        size_dummies = pd.get_dummies(df['size_category'], prefix='size_category')
        df = pd.concat([df, size_dummies], axis=1)
        df = df.drop('size_category', axis=1)
    
    return df


def save_engineered_data(df: pd.DataFrame, path: str) -> None:
    """
    Save feature-engineered dataset to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with engineered features to save.
    path : str
        Path where the engineered CSV file will be saved.
    
    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
