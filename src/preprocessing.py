"""
Data preprocessing utilities for SmartRent Manhattan project.
Handles cleaning, transformation, and preparation of raw data.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    
    Returns
    -------
    pd.DataFrame
        Raw dataset as a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataset by cleaning, transforming, and engineering features.
    
    Steps performed:
    1. Remove duplicate rows
    2. Convert columns to correct data types
    3. Create price_per_sqft feature
    4. Create amenity_count feature
    5. Remove outliers using 1st and 99th percentiles
    6. One-hot encode neighborhood column
    7. Handle missing values
    8. Ensure amenity columns are integers
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset to preprocess.
    
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset.
    """
    df = df.copy()
    
    df = df.drop_duplicates()
    
    df['bedrooms'] = df['bedrooms'].astype(int)
    df['bathrooms'] = df['bathrooms'].astype(int)
    df['floor'] = df['floor'].astype(int)
    
    df['price_per_sqft'] = df['rent'] / df['size_sqft']
    
    amenity_columns = ['no_fee', 'has_roofdeck', 'has_washer_dryer',
                       'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']
    df['amenity_count'] = df[amenity_columns].sum(axis=1)
    
    rent_lower = df['rent'].quantile(0.01)
    rent_upper = df['rent'].quantile(0.99)
    df = df[(df['rent'] >= rent_lower) & (df['rent'] <= rent_upper)]
    
    size_lower = df['size_sqft'].quantile(0.01)
    size_upper = df['size_sqft'].quantile(0.99)
    df = df[(df['size_sqft'] >= size_lower) & (df['size_sqft'] <= size_upper)]
    
    neighborhood_dummies = pd.get_dummies(df['neighborhood'], prefix='neighborhood')
    df = pd.concat([df, neighborhood_dummies], axis=1)
    df = df.drop('neighborhood', axis=1)
    
    amenity_cols = ['no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman',
                    'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']
    for col in amenity_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    df = df.dropna()
    
    return df


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Save processed dataset to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset to save.
    path : str
        Path where the processed CSV file will be saved.
    
    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
