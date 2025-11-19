"""
Complete pipeline runner for SmartRent Manhattan project.
Runs preprocessing, feature engineering, and model training in sequence.
"""

import os
import sys
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.preprocessing import load_raw_data, preprocess_data, save_processed_data
from src.feature_engineering import add_engineered_features, encode_engineered_features, save_engineered_data
from src.model_train import split_data, prepare_features, train_decision_tree, train_random_forest, train_xgboost, evaluate_model, save_model


def run_complete_pipeline():
    """
    Run the complete data science pipeline:
    1. Load and preprocess raw data
    2. Feature engineering
    3. Model training and evaluation
    """
    print("=" * 60)
    print("SmartRent Manhattan - Complete Pipeline")
    print("=" * 60)
    
    # Step 1: Preprocessing
    print("\n[1/3] Preprocessing Data...")
    print("-" * 60)
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'manhattan.csv')
    
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw data not found at {raw_data_path}")
        print("Please ensure manhattan.csv is in data/raw/ directory")
        return False
    
    try:
        df = load_raw_data(raw_data_path)
        print(f"Loaded {len(df):,} records")
        
        df = preprocess_data(df)
        print(f"Preprocessed: {len(df):,} records remaining")
        
        processed_path = os.path.join(project_root, 'data', 'processed', 'cleaned_manhattan.csv')
        save_processed_data(df, processed_path)
        print(f"Saved to {processed_path}")
        
    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Feature Engineering
    print("\n[2/3] Feature Engineering...")
    print("-" * 60)
    
    try:
        df = pd.read_csv(processed_path)
        print(f"Loaded processed data: {len(df):,} records")
        
        df = add_engineered_features(df)
        print(f"Added engineered features")
        
        df = encode_engineered_features(df)
        print(f"Encoded categorical features")
        
        save_engineered_data(df, processed_path)
        print(f"Saved feature-engineered data")
        
    except Exception as e:
        print(f"ERROR in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Model Training
    print("\n[3/3] Training Models...")
    print("-" * 60)
    
    try:
        df = pd.read_csv(processed_path)
        df = prepare_features(df)
        
        X_train, X_test, y_train, y_test = split_data(df, target='rent')
        print(f"Split data: Train={len(X_train):,}, Test={len(X_test):,}")
        
        print("\nTraining Decision Tree...")
        dt_model = train_decision_tree(X_train, y_train)
        dt_results = evaluate_model(dt_model, X_test, y_test)
        print(f"Results: {dt_results}")
        save_model(dt_model, os.path.join(project_root, 'models', 'decision_tree.pkl'))
        print("Saved decision_tree.pkl")
        
        print("\nTraining Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        rf_results = evaluate_model(rf_model, X_test, y_test)
        print(f"Results: {rf_results}")
        save_model(rf_model, os.path.join(project_root, 'models', 'random_forest.pkl'))
        print("Saved random_forest.pkl")
        
        print("\nTraining XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        xgb_results = evaluate_model(xgb_model, X_test, y_test)
        print(f"Results: {xgb_results}")
        save_model(xgb_model, os.path.join(project_root, 'models', 'xgboost.pkl'))
        print("Saved xgboost.pkl")
        
        # Select best model
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
        
        save_model(best_model, os.path.join(project_root, 'models', 'best_model.pkl'))
        print(f"\nBest model: {best_model_name} (RMSE: {results_dict[best_model_name]['RMSE']:.2f})")
        print(f"Saved best_model.pkl")
        
    except Exception as e:
        print(f"ERROR in model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run Streamlit app: streamlit run streamlit_app/Home.py")
    print("2. Or deploy to Streamlit Cloud")
    
    return True


if __name__ == "__main__":
    success = run_complete_pipeline()
    sys.exit(0 if success else 1)

