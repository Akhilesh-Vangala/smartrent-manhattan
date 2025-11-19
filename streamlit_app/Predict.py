"""
Streamlit Prediction page for SmartRent Manhattan dashboard.
Allows users to input features and get rental price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predict import load_model, prepare_input, predict_rent


def load_reference_dataset():
    """
    Load the processed dataset to use as reference for feature preparation.
    
    Returns
    -------
    pd.DataFrame
        Processed dataset for reference.
    """
    data_path = os.path.join(project_root, 'data', 'processed', 'cleaned_manhattan.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    else:
        st.error(f"Reference dataset not found at {data_path}")
        return None


def get_neighborhood_options(df):
    """
    Extract neighborhood names from the processed dataset.
    Since neighborhoods are one-hot encoded, extract from column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset with one-hot encoded neighborhoods.
    
    Returns
    -------
    list
        List of neighborhood names.
    """
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    neighborhoods = [col.replace('neighborhood_', '') for col in neighborhood_cols]
    return sorted(neighborhoods)


def calculate_neighborhood_avg_rent(df, neighborhood):
    """
    Calculate average rent for a given neighborhood from the reference dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Reference dataset with one-hot encoded neighborhoods.
    neighborhood : str
        Neighborhood name.
    
    Returns
    -------
    float
        Average rent for the neighborhood, or 0 if not found.
    """
    neighborhood_col = f'neighborhood_{neighborhood}'
    
    if neighborhood_col in df.columns:
        neighborhood_df = df[df[neighborhood_col] == 1]
        if len(neighborhood_df) > 0 and 'rent' in neighborhood_df.columns:
            return neighborhood_df['rent'].mean()
    
    return 0.0


def main():
    """
    Main function to render the Streamlit Prediction page.
    """
    st.title("Predict Manhattan Rent")
    st.subheader("Enter listing details to estimate the monthly rent.")
    
    st.divider()
    
    # Load reference dataset
    reference_df = load_reference_dataset()
    
    if reference_df is None:
        st.stop()
    
    # Get neighborhood options
    neighborhoods = get_neighborhood_options(reference_df)
    
    # Create prediction form
    with st.form(key="prediction_form"):
        st.header("Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
                help="Number of bedrooms"
            )
            
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="Number of bathrooms"
            )
            
            size_sqft = st.number_input(
                "Size (sqft)",
                min_value=0,
                max_value=10000,
                value=500,
                step=50,
                help="Square footage of the property"
            )
        
        with col2:
            min_to_subway = st.number_input(
                "Minutes to Subway",
                min_value=0.0,
                max_value=60.0,
                value=5.0,
                step=0.5,
                help="Walking time to nearest subway station"
            )
            
            floor = st.number_input(
                "Floor",
                min_value=0,
                max_value=100,
                value=1,
                step=1,
                help="Floor number"
            )
            
            building_age_yrs = st.number_input(
                "Building Age (years)",
                min_value=0,
                max_value=200,
                value=20,
                step=1,
                help="Age of the building in years"
            )
        
        st.divider()
        
        neighborhood = st.selectbox(
            "Neighborhood",
            options=neighborhoods,
            help="Select the Manhattan neighborhood"
        )
        
        st.divider()
        
        st.header("Amenities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            no_fee = st.checkbox("No Fee", help="No broker fee")
            has_roofdeck = st.checkbox("Roof Deck", help="Has roof deck")
            has_washer_dryer = st.checkbox("Washer/Dryer", help="Has washer/dryer")
        
        with col2:
            has_elevator = st.checkbox("Elevator", help="Has elevator")
            has_dishwasher = st.checkbox("Dishwasher", help="Has dishwasher")
            has_patio = st.checkbox("Patio", help="Has patio")
        
        with col3:
            has_gym = st.checkbox("Gym", help="Has gym")
            has_doorman = st.checkbox("Doorman", help="Has doorman")
        
        submitted = st.form_submit_button("Predict Rent", use_container_width=True)
    
    # Handle form submission
    if submitted:
        # Gather all input values into a dictionary
        user_input = {
            'bedrooms': int(bedrooms),
            'bathrooms': float(bathrooms),
            'size_sqft': int(size_sqft),
            'min_to_subway': float(min_to_subway),
            'floor': int(floor),
            'building_age_yrs': int(building_age_yrs),
            'neighborhood': neighborhood,
            'no_fee': 1 if no_fee else 0,
            'has_roofdeck': 1 if has_roofdeck else 0,
            'has_washer_dryer': 1 if has_washer_dryer else 0,
            'has_doorman': 1 if has_doorman else 0,
            'has_elevator': 1 if has_elevator else 0,
            'has_dishwasher': 1 if has_dishwasher else 0,
            'has_patio': 1 if has_patio else 0,
            'has_gym': 1 if has_gym else 0
        }
        
        try:
            # Load the best model
            model_path = os.path.join(project_root, 'models', 'best_model.pkl')
            
            if not os.path.exists(model_path):
                st.error(f"Model not found at {model_path}. Please train the model first.")
                st.stop()
            
            with st.spinner("Loading model and making prediction..."):
                model = load_model(model_path)
                
                # Prepare input for model
                X = prepare_input(user_input, reference_df)
                
                # Make prediction
                predicted_rent = predict_rent(model, X)
            
            st.divider()
            
            # Display prediction result
            st.success(f"## Predicted Monthly Rent: ${predicted_rent:,.0f}")
            
            st.divider()
            
            # Calculate and display additional metrics
            st.header("Property Metrics")
            
            price_per_sqft = predicted_rent / size_sqft if size_sqft > 0 else 0
            amenity_count = sum([
                user_input['no_fee'],
                user_input['has_roofdeck'],
                user_input['has_washer_dryer'],
                user_input['has_elevator'],
                user_input['has_dishwasher'],
                user_input['has_patio'],
                user_input['has_gym']
            ])
            neighborhood_avg = calculate_neighborhood_avg_rent(reference_df, neighborhood)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Rent", f"${predicted_rent:,.0f}")
            
            with col2:
                st.metric("Price per sqft", f"${price_per_sqft:,.2f}")
            
            with col3:
                st.metric("Amenity Count", f"{amenity_count}")
            
            with col4:
                st.metric("Neighborhood Avg", f"${neighborhood_avg:,.0f}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
