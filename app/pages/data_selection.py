"""
Data Selection & Loading Page
==============================

Handles:
- Plant selection
- Data loading and validation
- Data window configuration
- Data gap visualization
- Data pivoting

Maintainer: Project Team
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config_loader import load_config
from src.data.preprocessing import pivot_fsp_data, get_fsp_forecast_columns
from app.utils.page_summary import render_page_summary, get_page_context

config = load_config()
DATA_PROCESSED = PROJECT_DIR / 'data' / 'processed' / 'parquet'
DATA_INTERIM = PROJECT_DIR / config.get('data.interim_dir', 'data/interim')


def show():
    """Main function for data selection page."""
    st.header(" Data Selection & Loading")
    st.markdown("Select your power plant and configure data parameters")

    # AI-generated page summary
    render_page_summary("Data Selection", get_page_context("Data Selection"))

    # Step 1: Plant Selection
    st.markdown("### Step 1: Plant Selection")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Discover available plants from data files
        available_plants = discover_available_plants()

        if not available_plants:
            st.error(" No plant data found in `data/processed/` or `data/interim/` directories!")
            st.info("Please ensure your data files are in the correct location.")
            return

        plant = st.selectbox(
            "Select Power Plant",
            options=available_plants,
            index=0 if not st.session_state.plant_selected else available_plants.index(st.session_state.plant_selected),
            help="Choose the power plant to analyze"
        )

        st.session_state.plant_selected = plant

    with col2:
        st.metric("Available Plants", len(available_plants))

    # Step 2: Load Data
    st.markdown("### Step 2: Load and Inspect Data")

    if st.button(" Load Plant Data", type="primary"):
        with st.spinner("Loading data..."):
            success, df = load_plant_data(plant)

            if success:
                st.session_state.df_raw = df
                st.session_state.data_loaded = True
                st.success(f" Successfully loaded {len(df):,} rows of data")

                # Display basic metadata
                display_data_metadata(df)
            else:
                st.error(" Failed to load data. Please check file paths.")
                return

    # If data is loaded, show additional options
    if st.session_state.data_loaded and st.session_state.df_raw is not None:
        df = st.session_state.df_raw

        # Step 3: Data Window Selection
        st.markdown("### Step 3: Configure Data Window")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Get date range from data
            date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            total_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month

            data_months = st.slider(
                "Select number of latest months to use",
                min_value=3,
                max_value=max(total_months, 36),
                value=st.session_state.data_months,
                step=1,
                help="Use recent data for training. More data = better patterns, but may include outdated trends."
            )
            st.session_state.data_months = data_months

        with col2:
            st.metric("Total Data Range", f"{total_months} months")

        with col3:
            st.metric("Selected Range", f"{data_months} months")

        # Filter data by selected months
        cutoff_date = max_date - pd.DateOffset(months=data_months)
        df_filtered = df[df[date_col] >= cutoff_date].copy()

        st.info(f" Data Range: {df_filtered[date_col].min().strftime('%Y-%m-%d')} to {df_filtered[date_col].max().strftime('%Y-%m-%d')}")

        # Step 4: Data Gap Analysis
        st.markdown("### Step 4: Data Quality & Gap Analysis")

        with st.expander(" View Data Gaps and Missing Values", expanded=False):
            analyze_data_gaps(df_filtered)

        # Step 5: Pivot Data
        st.markdown("### Step 5: Pivot FSP Data")

        if st.button(" Pivot Data (Transform to Wide Format)", type="secondary"):
            with st.spinner("Pivoting FSP data..."):
                df_pivoted = pivot_fsp_data(df_filtered)
                st.session_state.df_pivoted = df_pivoted
                st.success(f" Data pivoted: {len(df_filtered):,}  {len(df_pivoted):,} rows")

                # Display pivoted data preview
                st.markdown("#### Pivoted Data Preview")
                st.dataframe(df_pivoted.head(10))

                # Show FSP columns
                fsp_cols = get_fsp_forecast_columns(df_pivoted)
                st.info(f" Found {len(fsp_cols)} FSP forecast columns: {', '.join([col.replace('forecast_power_', '').upper() for col in fsp_cols])}")

        # Step 6: Data Preview
        if st.session_state.df_pivoted is not None:
            st.markdown("### Step 6: Data Preview")

            preview_type = st.radio(
                "Select preview type:",
                ["First 50 rows", "Last 50 rows", "Random sample"],
                horizontal=True
            )

            df_preview = st.session_state.df_pivoted

            if preview_type == "First 50 rows":
                st.dataframe(df_preview.head(50))
            elif preview_type == "Last 50 rows":
                st.dataframe(df_preview.tail(50))
            else:
                st.dataframe(df_preview.sample(min(50, len(df_preview))))

            # Download option
            csv = df_preview.to_csv(index=False)
            st.download_button(
                label=" Download Pivoted Data (CSV)",
                data=csv,
                file_name=f"{plant}_pivoted_data.csv",
                mime="text/csv"
            )

            st.success(" Data is ready! Proceed to **Feature Engineering** ")


def discover_available_plants():
    """Discover available plants from data files."""
    plants = []

    # Check processed parquet directory (new structure)
    if DATA_PROCESSED.exists():
        parquet_files = list(DATA_PROCESSED.glob("*_dataset.parquet"))
        if parquet_files:
            # Extract plant names from filenames like "sample_pss_dataset.parquet"
            plants = [f.stem.replace('_dataset', '').upper() for f in parquet_files]

    # Check interim directory (fallback for old structure)
    if DATA_INTERIM.exists():
        parquet_files = list(DATA_INTERIM.glob("*.parquet"))
        if parquet_files:
            interim_plants = [f.stem.upper() for f in parquet_files
                            if not f.stem.startswith('eda_')]
            plants.extend(interim_plants)

    # Remove duplicates and sort
    plants = sorted(list(set(plants)))

    # Fallback to default if nothing found
    if not plants:
        plants = ["SAMPLE_PSS"]

    return plants


def load_plant_data(plant_name):
    """Load plant data from parquet files."""
    try:
        # Convert plant name to lowercase with underscores for filename
        plant_filename = plant_name.lower().replace(' ', '_')

        # Try processed directory first (new structure)
        processed_file = DATA_PROCESSED / f"{plant_filename}_dataset.parquet"
        if processed_file.exists():
            df = pd.read_parquet(processed_file)
            return True, df

        # Try interim directory (processed EDA data)
        interim_file = DATA_INTERIM / f"{plant_name}.parquet"
        if interim_file.exists():
            df = pd.read_parquet(interim_file)
            return True, df

        # Try standard name in interim
        interim_file = DATA_INTERIM / "eda_processed_data.parquet"
        if interim_file.exists():
            df = pd.read_parquet(interim_file)
            return True, df

        # Try old processed structure
        old_processed_file = DATA_PROCESSED.parent / f"{plant_name}.parquet"
        if old_processed_file.exists():
            df = pd.read_parquet(old_processed_file)
            return True, df

        return False, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False, None


def display_data_metadata(df):
    """Display basic metadata about the loaded data."""
    col1, col2, col3, col4 = st.columns(4)

    date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col])

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Values", f"{missing_pct:.2f}%")

    with col4:
        # Count FSP providers
        if 'forecast_facode' in df.columns:
            n_fsps = df['forecast_facode'].nunique()
            st.metric("FSP Providers", n_fsps)
        else:
            st.metric("Date Range", f"{(df[date_col].max() - df[date_col].min()).days} days")


def analyze_data_gaps(df):
    """Analyze and visualize data gaps."""
    date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col])

    # Missing values per column
    st.markdown("#### Missing Values per Column")
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)

    missing_stats = missing_stats[missing_stats['Missing Count'] > 0]

    if len(missing_stats) > 0:
        fig = px.bar(
            missing_stats.head(20),
            x='Missing %',
            y='Column',
            orientation='h',
            title='Top 20 Columns with Missing Values',
            labels={'Missing %': 'Missing Percentage (%)'},
            color='Missing %',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

        # Show table
        st.dataframe(missing_stats)
    else:
        st.success(" No missing values detected!")

    # Time series gap analysis
    st.markdown("#### Time Series Gap Analysis")

    # Calculate expected time steps (15-minute intervals)
    df_sorted = df.sort_values(date_col).copy()
    df_sorted['time_diff'] = df_sorted[date_col].diff()

    # Expected diff is 15 minutes
    expected_diff = pd.Timedelta(minutes=15)
    gaps = df_sorted[df_sorted['time_diff'] > expected_diff]

    if len(gaps) > 0:
        st.warning(f" Found {len(gaps)} time gaps larger than 15 minutes")

        # Visualize gaps
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gaps[date_col],
            y=gaps['time_diff'].dt.total_seconds() / 60,
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Time Gaps'
        ))
        fig.update_layout(
            title='Time Gaps in Data (>15 minutes)',
            xaxis_title='Date',
            yaxis_title='Gap Duration (minutes)',
            height=400
        )
        st.plotly_chart(fig, width='stretch')

        # Show gap details
        with st.expander("View Gap Details"):
            gap_details = gaps[[date_col, 'time_diff']].copy()
            gap_details['gap_minutes'] = gap_details['time_diff'].dt.total_seconds() / 60
            st.dataframe(gap_details)
    else:
        st.success(" No significant time gaps detected!")

    # FSP availability analysis
    if 'forecast_facode' in df.columns:
        st.markdown("#### FSP Data Availability")

        fsp_counts = df.groupby('forecast_facode').size().sort_values(ascending=False)

        fig = px.bar(
            x=fsp_counts.values,
            y=fsp_counts.index,
            orientation='h',
            title='Data Points per FSP Provider',
            labels={'x': 'Number of Records', 'y': 'FSP Provider'},
            color=fsp_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
