"""
Streamlit Map visualization page for SmartRent Manhattan dashboard.
Displays interactive maps and location-based insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
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


def get_neighborhood_names(df):
    """
    Extract neighborhood names from one-hot encoded columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with one-hot encoded neighborhoods.
    
    Returns
    -------
    list
        List of neighborhood names.
    """
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    neighborhoods = [col.replace('neighborhood_', '') for col in neighborhood_cols]
    return sorted(neighborhoods)


def add_coordinates(df):
    """
    Add latitude and longitude coordinates to the dataframe.
    If coordinates don't exist, simulate them using neighborhood centroids.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to add coordinates to.
    
    Returns
    -------
    pd.DataFrame
        Dataset with latitude and longitude columns.
    """
    df = df.copy()
    
    # Check if lat/lon already exist
    if 'latitude' in df.columns and 'longitude' in df.columns:
        return df
    
    # Manhattan approximate center coordinates
    manhattan_center_lat = 40.7831
    manhattan_center_lon = -73.9712
    
    # Get neighborhood columns
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    
    if neighborhood_cols:
        # Create a mapping of neighborhoods to approximate coordinates
        # Using a grid pattern centered on Manhattan
        neighborhoods = [col.replace('neighborhood_', '') for col in neighborhood_cols]
        
        # Generate coordinates for each neighborhood
        np.random.seed(42)
        neighborhood_coords = {}
        
        for i, neighborhood in enumerate(neighborhoods):
            # Spread coordinates around Manhattan center
            lat_offset = (np.random.random() - 0.5) * 0.15
            lon_offset = (np.random.random() - 0.5) * 0.15
            neighborhood_coords[neighborhood] = {
                'latitude': manhattan_center_lat + lat_offset,
                'longitude': manhattan_center_lon + lon_offset
            }
        
        # Assign coordinates based on neighborhood
        df['latitude'] = manhattan_center_lat
        df['longitude'] = manhattan_center_lon
        
        for col in neighborhood_cols:
            neighborhood = col.replace('neighborhood_', '')
            if neighborhood in neighborhood_coords:
                mask = df[col] == 1
                df.loc[mask, 'latitude'] = neighborhood_coords[neighborhood]['latitude']
                df.loc[mask, 'longitude'] = neighborhood_coords[neighborhood]['longitude']
    else:
        # If no neighborhoods, use center coordinates with small random offsets
        np.random.seed(42)
        df['latitude'] = manhattan_center_lat + (np.random.random(len(df)) - 0.5) * 0.1
        df['longitude'] = manhattan_center_lon + (np.random.random(len(df)) - 0.5) * 0.1
    
    return df


def apply_filters(df, selected_bedrooms, selected_neighborhoods, rent_range):
    """
    Apply filters to the dataframe based on user selections.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    selected_bedrooms : list
        List of selected bedroom counts.
    selected_neighborhoods : list
        List of selected neighborhoods.
    rent_range : tuple
        Tuple of (min_rent, max_rent).
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    filtered_df = df.copy()
    
    # Filter by bedrooms
    if selected_bedrooms:
        filtered_df = filtered_df[filtered_df['bedrooms'].isin(selected_bedrooms)]
    
    # Filter by neighborhoods
    if selected_neighborhoods:
        neighborhood_cols = [f'neighborhood_{n}' for n in selected_neighborhoods]
        available_cols = [col for col in neighborhood_cols if col in filtered_df.columns]
        
        if available_cols:
            mask = filtered_df[available_cols].sum(axis=1) > 0
            filtered_df = filtered_df[mask]
    
    # Filter by rent range
    if 'rent' in filtered_df.columns:
        min_rent, max_rent = rent_range
        filtered_df = filtered_df[
            (filtered_df['rent'] >= min_rent) & 
            (filtered_df['rent'] <= max_rent)
        ]
    
    return filtered_df


def create_scatter_map(df):
    """
    Create a PyDeck scatter plot map visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe with coordinates and rent data.
    
    Returns
    -------
    pydeck.Deck
        PyDeck deck object for rendering.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    if len(df) == 0:
        return None
    
    # Prepare data for PyDeck
    map_data = df[['latitude', 'longitude', 'rent']].copy()
    
    if 'price_per_sqft' in df.columns:
        map_data['price_per_sqft'] = df['price_per_sqft']
    else:
        map_data['price_per_sqft'] = map_data['rent'] / 500
    
    # Normalize rent for radius scaling
    if map_data['rent'].max() > map_data['rent'].min():
        map_data['radius'] = 50 + (map_data['rent'] - map_data['rent'].min()) / (map_data['rent'].max() - map_data['rent'].min()) * 200
    else:
        map_data['radius'] = 100
    
    # Create color gradient based on rent
    if map_data['rent'].max() > map_data['rent'].min():
        normalized_rent = (map_data['rent'] - map_data['rent'].min()) / (map_data['rent'].max() - map_data['rent'].min())
    else:
        normalized_rent = 0.5
    
    # Color mapping: blue (low) to red (high)
    map_data['r'] = (normalized_rent * 255).astype(int)
    map_data['g'] = ((1 - normalized_rent) * 255).astype(int)
    map_data['b'] = 50
    
    # Calculate center of the map
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    # Create PyDeck layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=['longitude', 'latitude'],
        get_radius='radius',
        get_fill_color='[r, g, b, 180]',
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=50,
    )
    
    # Create deck
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0,
        ),
        layers=[scatter_layer],
        tooltip={
            "html": "<b>Rent:</b> ${rent:,.0f}<br><b>Price/sqft:</b> ${price_per_sqft:,.2f}",
            "style": {"color": "white"}
        }
    )
    
    return deck


def create_heatmap(df):
    """
    Create a PyDeck heatmap visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe with coordinates and rent data.
    
    Returns
    -------
    pydeck.Deck
        PyDeck deck object for rendering.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    if len(df) == 0:
        return None
    
    # Prepare data for PyDeck
    map_data = df[['latitude', 'longitude', 'rent']].copy()
    
    if 'price_per_sqft' in df.columns:
        map_data['weight'] = df['price_per_sqft']
    else:
        map_data['weight'] = map_data['rent'] / 500
    
    # Calculate center of the map
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    # Create PyDeck heatmap layer
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=map_data,
        get_position=['longitude', 'latitude'],
        get_weight='weight',
        radius_pixels=60,
        intensity=1,
        threshold=0.05,
    )
    
    # Create deck
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0,
        ),
        layers=[heatmap_layer],
    )
    
    return deck


def compute_neighborhood_summary(df):
    """
    Compute neighborhood-level summary statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with one-hot encoded neighborhoods.
    
    Returns
    -------
    pd.DataFrame
        Summary dataframe with neighborhood statistics.
    """
    neighborhood_cols = [col for col in df.columns if col.startswith('neighborhood_')]
    
    if not neighborhood_cols:
        return pd.DataFrame()
    
    summaries = []
    
    for col in neighborhood_cols:
        neighborhood_name = col.replace('neighborhood_', '')
        neighborhood_df = df[df[col] == 1]
        
        if len(neighborhood_df) > 0:
            summary = {
                'Neighborhood': neighborhood_name,
                'Count': len(neighborhood_df)
            }
            
            if 'rent' in neighborhood_df.columns:
                summary['Avg Rent'] = neighborhood_df['rent'].mean()
            
            if 'price_per_sqft' in neighborhood_df.columns:
                summary['Avg Price/sqft'] = neighborhood_df['price_per_sqft'].mean()
            
            if 'amenity_count' in neighborhood_df.columns:
                summary['Avg Amenity Count'] = neighborhood_df['amenity_count'].mean()
            
            summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    
    if 'Avg Rent' in summary_df.columns:
        summary_df = summary_df.sort_values('Avg Rent', ascending=False)
    
    return summary_df


def main():
    """
    Main function to render the Streamlit Map page.
    """
    st.title("Manhattan Rental Maps")
    st.subheader("Visualize rental patterns, neighborhood trends, and affordability.")
    
    st.divider()
    
    # Load data
    df = load_processed_data()
    
    if df is None:
        st.stop()
    
    # Add coordinates if missing
    df = add_coordinates(df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Bedroom filter
    if 'bedrooms' in df.columns:
        unique_bedrooms = sorted(df['bedrooms'].unique())
        selected_bedrooms = st.sidebar.multiselect(
            "Bedrooms",
            options=unique_bedrooms,
            default=unique_bedrooms,
            help="Select bedroom counts to display"
        )
    else:
        selected_bedrooms = []
    
    # Neighborhood filter
    neighborhoods = get_neighborhood_names(df)
    selected_neighborhoods = st.sidebar.multiselect(
        "Neighborhoods",
        options=neighborhoods,
        default=neighborhoods,
        help="Select neighborhoods to display"
    )
    
    # Rent range filter
    if 'rent' in df.columns:
        min_rent = int(df['rent'].min())
        max_rent = int(df['rent'].max())
        rent_range = st.sidebar.slider(
            "Rent Range ($)",
            min_value=min_rent,
            max_value=max_rent,
            value=(min_rent, max_rent),
            help="Filter by rent range"
        )
    else:
        rent_range = (0, 100000)
    
    # Heatmap toggle
    show_heatmap = st.sidebar.checkbox("Show Heatmap", help="Toggle between scatter plot and heatmap")
    
    st.divider()
    
    # Apply filters
    filtered_df = apply_filters(df, selected_bedrooms, selected_neighborhoods, rent_range)
    
    st.write(f"**Displaying {len(filtered_df):,} properties**")
    
    # Map visualization
    st.header("Interactive Map")
    
    if show_heatmap:
        deck = create_heatmap(filtered_df)
    else:
        deck = create_scatter_map(filtered_df)
    
    if deck is not None:
        st.pydeck_chart(deck)
        st.caption("Map shows rental properties with color and size indicating rent levels.")
    else:
        st.warning("Unable to create map visualization. Check if coordinates are available.")
    
    st.divider()
    
    # Neighborhood summary
    st.header("Neighborhood Summary")
    st.markdown("""
    Average rental statistics by neighborhood. Use the filters in the sidebar to explore 
    specific neighborhoods and price ranges.
    """)
    
    summary_df = compute_neighborhood_summary(filtered_df)
    
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No neighborhood data available with current filters.")
    
    st.divider()
    
    # Insights section
    st.header("Insights")
    
    if 'rent' in filtered_df.columns and len(filtered_df) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_rent = filtered_df['rent'].mean()
            st.metric("Average Rent", f"${avg_rent:,.0f}")
        
        with col2:
            if 'price_per_sqft' in filtered_df.columns:
                avg_price_sqft = filtered_df['price_per_sqft'].mean()
                st.metric("Avg Price/sqft", f"${avg_price_sqft:,.2f}")
        
        with col3:
            if 'amenity_count' in filtered_df.columns:
                avg_amenities = filtered_df['amenity_count'].mean()
                st.metric("Avg Amenities", f"{avg_amenities:.1f}")


if __name__ == "__main__":
    main()
