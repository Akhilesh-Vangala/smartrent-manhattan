"""
Streamlit Home page for SmartRent Manhattan dashboard.
Main landing page of the application.
"""

import streamlit as st
import pandas as pd
import os


def load_processed_data():
    """
    Load the processed dataset from the data/processed directory.
    
    Returns
    -------
    pd.DataFrame
        Processed Manhattan rental dataset.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed', 'cleaned_manhattan.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    else:
        st.error(f"Dataset not found at {data_path}")
        return None


def display_dataset_summary(df):
    """
    Display dataset summary information including records, features, sample, and statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize.
    """
    st.subheader("Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Records", f"{len(df):,}")
    
    with col2:
        st.metric("Number of Features", f"{len(df.columns):,}")
    
    st.subheader("Sample Data (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)


def display_key_insights(df):
    """
    Display key insights from the dataset including average rent, price per sqft,
    and top/bottom neighborhoods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    """
    st.subheader("Key Insights")
    
    if 'rent' in df.columns:
        avg_rent = df['rent'].mean()
        st.metric("Average Rent", f"${avg_rent:,.2f}")
    
    if 'price_per_sqft' in df.columns:
        avg_price_per_sqft = df['price_per_sqft'].mean()
        st.metric("Average Price per Square Foot", f"${avg_price_per_sqft:,.2f}")
    
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    
    if neighborhood_cols:
        neighborhood_rents = {}
        
        for col in neighborhood_cols:
            neighborhood_name = col.replace('neighborhood_', '')
            neighborhood_df = df[df[col] == 1]
            
            if len(neighborhood_df) > 0 and 'rent' in neighborhood_df.columns:
                avg_rent = neighborhood_df['rent'].mean()
                neighborhood_rents[neighborhood_name] = avg_rent
        
        if neighborhood_rents:
            sorted_neighborhoods = sorted(neighborhood_rents.items(), key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Neighborhoods by Average Rent**")
                top_5 = sorted_neighborhoods[:5]
                for i, (neighborhood, rent) in enumerate(top_5, 1):
                    st.write(f"{i}. {neighborhood}: ${rent:,.2f}")
            
            with col2:
                st.write("**Bottom 5 Neighborhoods by Average Rent**")
                bottom_5 = sorted_neighborhoods[-5:]
                for i, (neighborhood, rent) in enumerate(bottom_5, 1):
                    st.write(f"{i}. {neighborhood}: ${rent:,.2f}")


def display_navigation():
    """
    Display navigation links to other pages in the Streamlit app.
    """
    st.subheader("Navigation")
    
    st.markdown("""
    Navigate to other pages:
    
    - **[Predict Rent](Predict)** - Get rental price predictions for your property
    - **[Map Visualization](Map)** - Explore rental prices across Manhattan neighborhoods  
    - **[Model Interpretability](Interpret)** - Understand feature importance and SHAP values
    """)


def main():
    """
    Main function to render the Streamlit Home page.
    """
    st.title("SmartRent Manhattan")
    st.subheader("Rental Price Modeling & Affordability Insights Across Manhattan")
    
    st.markdown("""
    This application provides comprehensive analysis and prediction of rental prices across Manhattan 
    using the StreetEasy dataset. The project aims to help renters, property managers, and real estate 
    professionals understand rental market trends and make informed decisions.
    
    **Dataset:** StreetEasy Manhattan Rental Data
    **Goal:** Build accurate rental price prediction models and provide actionable insights into 
    Manhattan's rental market dynamics.
    """)
    
    st.divider()
    
    df = load_processed_data()
    
    if df is not None:
        display_dataset_summary(df)
        
        st.divider()
        
        display_key_insights(df)
        
        st.divider()
        
        display_navigation()
    else:
        st.warning("Please ensure the processed dataset is available at data/processed/cleaned_manhattan.csv")


if __name__ == "__main__":
    main()
