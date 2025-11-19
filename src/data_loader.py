"""
Data loading utilities for SmartRent Manhattan project.
Handles loading raw data from various sources.
"""

import pandas as pd
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    This function provides a centralized way to load raw datasets.
    Supports CSV files with automatic encoding detection.
    
    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    
    Returns
    -------
    pd.DataFrame
        Raw dataset as a pandas DataFrame.
    
    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')
    
    return df


def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Parameters
    ----------
    path : str
        Path to the processed CSV file.
    
    Returns
    -------
    pd.DataFrame
        Processed dataset as a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found at: {path}")
    
    return pd.read_csv(path)


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    
    Returns
    -------
    dict
        Dictionary containing dataset information:
        - shape: (rows, columns)
        - columns: list of column names
        - dtypes: data types
        - memory_usage: memory usage in bytes
        - null_counts: count of null values per column
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict()
    }
    
    return info
