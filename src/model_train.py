"""
Model training utilities for SmartRent Manhattan project.
Handles training machine learning models for rental price prediction.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

try:
    from .feature_engineering import add_engineered_features, encode_engineered_features
except ImportError:
    from feature_engineering import add_engineered_features, encode_engineered_features


def split_data(df: pd.DataFrame, target: str):
    """
    Split dataset into features and target, then split into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target column.
    target : str
        Name of the target column.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) - Training and testing sets.
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling by dropping non-numeric columns and handling data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with engineered features.
    
    Returns
    -------
    pd.DataFrame
        Dataset ready for modeling with only numeric features.
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'rent' in df.columns:
        numeric_cols.append('rent')
    
    df = df[numeric_cols]
    
    for col in df.columns:
        col_dtype = str(df.dtypes[col])
        if col_dtype == 'bool' or 'bool' in col_dtype.lower():
            df[col] = df[col].astype(int)
    
    df = df.fillna(0)
    
    return df


def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree Regressor with hyperparameter tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target values.
    
    Returns
    -------
    DecisionTreeRegressor
        Trained and tuned decision tree model.
    """
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = DecisionTreeRegressor(random_state=42)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor with hyperparameter tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target values.
    
    Returns
    -------
    RandomForestRegressor
        Trained and tuned random forest model.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost Regressor with hyperparameter tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target values.
    
    Returns
    -------
    XGBRegressor
        Trained and tuned XGBoost model.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=kfold,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
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
    
    results = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    return results


def save_model(model, path: str) -> None:
    """
    Save a trained model to disk using pickle.
    
    Parameters
    ----------
    model : sklearn model
        Trained model to save.
    path : str
        File path where the model will be saved.
    
    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    data_path = 'data/processed/cleaned_manhattan.csv'
    df = pd.read_csv(data_path)
    
    # Check if features are already engineered (neighborhood_avg_rent exists)
    if 'neighborhood_avg_rent' not in df.columns:
        df = add_engineered_features(df)
        df = encode_engineered_features(df)
    
    df = prepare_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df, target='rent')
    
    print("Training Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train)
    dt_results = evaluate_model(dt_model, X_test, y_test)
    print(f"Decision Tree Results: {dt_results}")
    save_model(dt_model, 'models/decision_tree.pkl')
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest Results: {rf_results}")
    save_model(rf_model, 'models/random_forest.pkl')
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost Results: {xgb_results}")
    save_model(xgb_model, 'models/xgboost.pkl')
    
    results_dict = {
        'decision_tree': dt_results,
        'random_forest': rf_results,
        'xgboost': xgb_results
    }
    
    best_model_name = min(results_dict, key=lambda x: results_dict[x]['RMSE'])
    
    if best_model_name == 'decision_tree':
        best_model = dt_model
    elif best_model_name == 'random_forest':
        best_model = rf_model
    else:
        best_model = xgb_model
    
    save_model(best_model, 'models/best_model.pkl')
    print(f"\nBest model: {best_model_name} (RMSE: {results_dict[best_model_name]['RMSE']:.2f})")
    print("All models saved successfully.")
