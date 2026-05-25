"""
Feature Engineering Page
=========================

Handles:
- Data splitting (train/val/test)
- Temporal split visualization
- Feature creation (rolling, time-based, categorical)
- Feature inspection and statistics

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

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.features.feature_engineering import (
    create_temporal_split,
    create_rolling_features,
    create_time_features,
    encode_categorical_features,
    get_feature_columns
)
from src.data.preprocessing import calculate_fsp_errors
from app.utils.page_summary import render_page_summary, get_page_context


def show():
    """Main function for feature engineering page."""
    st.header(" Feature Engineering & Data Splitting")
    st.markdown("Configure data splits and create features for model training")

    # AI-generated page summary
    render_page_summary("Feature Engineering", get_page_context("Feature Engineering"))

    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_pivoted is None:
        st.warning(" Please load and pivot data first in **Data Selection** page")
        return

    df = st.session_state.df_pivoted

    # Step 1: Data Splitting Configuration
    st.markdown("### Step 1: Configure Data Splits")
    st.info(" Use temporal splits to prevent data leakage. Train on past, validate on recent, test on future.")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_ratio = st.slider(
            "Train Ratio",
            min_value=0.50,
            max_value=0.85,
            value=st.session_state.train_ratio,
            step=0.05,
            help="Percentage of data for training"
        )
        st.session_state.train_ratio = train_ratio

    with col2:
        val_ratio = st.slider(
            "Validation Ratio",
            min_value=0.10,
            max_value=0.30,
            value=st.session_state.val_ratio,
            step=0.05,
            help="Percentage of data for validation"
        )
        st.session_state.val_ratio = val_ratio

    with col3:
        test_ratio = 1.0 - train_ratio - val_ratio
        st.metric("Test Ratio", f"{test_ratio:.2%}")
        st.session_state.test_ratio = test_ratio

    # Validate splits sum to 1
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        st.error(f" Splits must sum to 100%. Current: {total*100:.1f}%")
        return

    # Show split visualization
    st.markdown("#### Split Visualization")
    visualize_split_ratios(train_ratio, val_ratio, test_ratio, len(df))

    # Step 2: Perform Split
    st.markdown("### Step 2: Perform Temporal Split")

    if st.button(" Split Data", type="primary"):
        with st.spinner("Splitting data temporally..."):
            # First, handle missing values and create target
            df_clean = prepare_data_for_split(df)

            if df_clean is None:
                st.error(" Data preparation failed")
                return

            # Perform temporal split
            train_df, val_df, test_df = create_temporal_split(
                df_clean, train_ratio, val_ratio, test_ratio
            )

            # Store in session state
            st.session_state.train_df = train_df
            st.session_state.val_df = val_df
            st.session_state.test_df = test_df
            st.session_state.split_complete = True

            st.success(f" Split complete: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

            # Show split details
            display_split_details(train_df, val_df, test_df)

    # If split is complete, show feature engineering options
    if st.session_state.split_complete:
        st.markdown("---")
        st.markdown("### Step 3: Feature Engineering")

        # Feature creation options
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Rolling Window Features")

            window_1 = st.checkbox("1-block lag (15 min)", value=True)
            window_6 = st.checkbox("6-block rolling (1.5 hours)", value=True)
            window_24 = st.checkbox("24-block rolling (6 hours)", value=True)
            window_96 = st.checkbox("96-block rolling (24 hours)", value=True)

            windows = []
            if window_1: windows.append(1)
            if window_6: windows.append(6)
            if window_24: windows.append(24)
            if window_96: windows.append(96)

        with col2:
            st.markdown("#### Time-based Features")

            create_hour = st.checkbox("Hour features (sin/cos)", value=True)
            create_dow = st.checkbox("Day of week features", value=True)
            create_month = st.checkbox("Month features (seasonal)", value=True)
            create_weekend = st.checkbox("Weekend indicator", value=True)

        # Create features button
        if st.button(" Create Features", type="primary"):
            with st.spinner("Creating features..."):
                success = create_all_features(windows)

                if success:
                    st.success(" Features created successfully!")

                    # Display feature statistics
                    display_feature_statistics()

                    st.success(" Data is ready for model training! Proceed to **FSP Selection** ")
                else:
                    st.error(" Feature creation failed")

        # Feature inspection
        if st.session_state.features_created:
            st.markdown("---")
            st.markdown("### Step 4: Feature Inspection")

            inspect_features()


def prepare_data_for_split(df):
    """Prepare data: handle missing values, create target."""
    try:
        TARGET = 'actual_power'
        PREDICTION_HORIZON = 6

        # Check target exists
        if TARGET not in df.columns:
            st.error(f" Target column '{TARGET}' not found")
            return None

        # Drop rows with missing target
        df_clean = df.dropna(subset=[TARGET]).copy()

        # Calculate FSP errors
        df_clean = calculate_fsp_errors(df_clean, TARGET)

        # Create forward-shifted target (6-block ahead)
        df_clean['target_horizon'] = df_clean[TARGET].shift(-PREDICTION_HORIZON)

        # Drop trailing rows with NaN target_horizon
        df_clean = df_clean.dropna(subset=['target_horizon'])

        st.info(f" Prepared {len(df_clean):,} rows for splitting (dropped {len(df) - len(df_clean):,} rows with missing values)")

        return df_clean

    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None


def visualize_split_ratios(train_ratio, val_ratio, test_ratio, total_rows):
    """Visualize data split ratios."""
    # Calculate row counts
    train_rows = int(total_rows * train_ratio)
    val_rows = int(total_rows * val_ratio)
    test_rows = total_rows - train_rows - val_rows

    # Create stacked bar
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Train',
        x=[train_ratio],
        y=['Split'],
        orientation='h',
        marker=dict(color='#2E86AB'),
        text=f'{train_ratio:.0%} ({train_rows:,} rows)',
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        name='Validation',
        x=[val_ratio],
        y=['Split'],
        orientation='h',
        marker=dict(color='#A23B72'),
        text=f'{val_ratio:.0%} ({val_rows:,} rows)',
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        name='Test',
        x=[test_ratio],
        y=['Split'],
        orientation='h',
        marker=dict(color='#F18F01'),
        text=f'{test_ratio:.0%} ({test_rows:,} rows)',
        textposition='inside'
    ))

    fig.update_layout(
        barmode='stack',
        height=200,
        showlegend=True,
        xaxis=dict(title='Proportion', tickformat='.0%'),
        yaxis=dict(showticklabels=False)
    )

    st.plotly_chart(fig, width='stretch')


def display_split_details(train_df, val_df, test_df):
    """Display detailed split information."""
    date_col = 'timestamp' if 'timestamp' in train_df.columns else 'date'

    # Date ranges
    splits_info = []
    for name, df_split in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        df_split[date_col] = pd.to_datetime(df_split[date_col])
        splits_info.append({
            'Split': name,
            'Rows': len(df_split),
            'Start Date': df_split[date_col].min().strftime('%Y-%m-%d'),
            'End Date': df_split[date_col].max().strftime('%Y-%m-%d'),
            'Days': (df_split[date_col].max() - df_split[date_col].min()).days
        })

    splits_df = pd.DataFrame(splits_info)
    st.dataframe(splits_df)

    # Timeline visualization
    fig = go.Figure()

    colors = {'Train': '#2E86AB', 'Validation': '#A23B72', 'Test': '#F18F01'}

    for _, row in splits_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Start Date'], row['End Date']],
            y=[row['Split'], row['Split']],
            mode='lines+markers',
            name=row['Split'],
            line=dict(color=colors[row['Split']], width=20),
            marker=dict(size=12)
        ))

    fig.update_layout(
        title='Temporal Split Timeline',
        xaxis_title='Date',
        yaxis_title='Split',
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, width='stretch')


def create_all_features(windows):
    """Create all features for train/val/test sets."""
    try:
        train_df = st.session_state.train_df.copy()
        val_df = st.session_state.val_df.copy()
        test_df = st.session_state.test_df.copy()

        TARGET = 'actual_power'

        # Time features
        train_df = create_time_features(train_df)
        val_df = create_time_features(val_df)
        test_df = create_time_features(test_df)

        st.info(" Created time-based features")

        # Rolling features
        if windows:
            train_df = create_rolling_features(train_df, TARGET, windows)
            val_df = create_rolling_features(val_df, TARGET, windows)
            test_df = create_rolling_features(test_df, TARGET, windows)
            st.info(f" Created rolling features for windows: {windows}")

        # Categorical encoding
        exclude_cols = [TARGET, 'target_horizon', 'date', 'timestamp', 'sscode']
        train_df, encoders = encode_categorical_features(train_df, exclude_cols)
        val_df, _ = encode_categorical_features(val_df, exclude_cols, encoders)
        test_df, _ = encode_categorical_features(test_df, exclude_cols, encoders)

        st.info(" Encoded categorical features")

        # Get feature columns
        feature_cols = get_feature_columns(
            train_df,
            target_col='target_horizon',
            exclude_patterns=['date', 'timestamp', 'index', 'actual_', 'schedule_', 'error_', 'sscode']
        )

        # Filter feature columns to only selected FSPs
        selected_fsps = st.session_state.get('selected_fsps', [])
        if selected_fsps:
            # Keep only forecast columns for selected FSPs
            selected_fsp_cols = []
            for col in feature_cols:
                # Check if this is a forecast or rolling error column
                if col.startswith('forecast_power_'):
                    fsp_name = col.replace('forecast_power_', '').upper()
                    if fsp_name in selected_fsps:
                        selected_fsp_cols.append(col)
                elif 'rolling_mae_forecast_power_' in col:
                    # Extract FSP name from rolling_mae_forecast_power_xxx_window
                    fsp_part = col.replace('rolling_mae_forecast_power_', '').rsplit('_', 1)[0]
                    fsp_name = fsp_part.upper()
                    if fsp_name in selected_fsps:
                        selected_fsp_cols.append(col)
                else:
                    # Keep non-FSP columns (rolling stats, time features, etc.)
                    selected_fsp_cols.append(col)

            feature_cols = selected_fsp_cols
            st.info(f" Filtered features to selected FSPs: {', '.join(selected_fsps)}")

        # Filter dataframes to only keep selected FSP columns (don't drop them, just exclude from features)
        # But ensure test/val/train data still have all FSP columns for visualization purposes

        # Update session state
        st.session_state.train_df = train_df
        st.session_state.val_df = val_df
        st.session_state.test_df = test_df
        st.session_state.feature_columns = feature_cols
        st.session_state.encoders = encoders
        st.session_state.features_created = True

        return True

    except Exception as e:
        st.error(f"Error creating features: {str(e)}")
        return False


def display_feature_statistics():
    """Display statistics about created features."""
    feature_cols = st.session_state.feature_columns
    train_df = st.session_state.train_df

    st.markdown("#### Feature Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Features", len(feature_cols))

    with col2:
        rolling_features = [f for f in feature_cols if 'rolling' in f]
        st.metric("Rolling Features", len(rolling_features))

    with col3:
        time_features = [f for f in feature_cols if any(x in f for x in ['hour', 'dow', 'month', 'weekend'])]
        st.metric("Time Features", len(time_features))

    # Show sample features
    with st.expander(" View All Features"):
        st.write(feature_cols)


def inspect_features():
    """Allow users to inspect created features."""
    st.markdown("Select a feature to inspect:")

    feature_cols = st.session_state.feature_columns
    train_df = st.session_state.train_df

    selected_feature = st.selectbox("Feature", options=feature_cols)

    col1, col2 = st.columns(2)

    with col1:
        # Statistics
        st.markdown("##### Statistics")
        stats = train_df[selected_feature].describe()
        st.dataframe(stats)

    with col2:
        # Distribution plot
        st.markdown("##### Distribution")
        fig = px.histogram(
            train_df,
            x=selected_feature,
            nbins=50,
            title=f'Distribution of {selected_feature}'
        )
        st.plotly_chart(fig, width='stretch')

    # Time series plot
    st.markdown("##### Time Series")
    date_col = 'timestamp' if 'timestamp' in train_df.columns else 'date'

    # Sample for performance (plot max 10000 points)
    plot_df = train_df.sample(min(10000, len(train_df))).sort_values(date_col)

    fig = px.line(
        plot_df,
        x=date_col,
        y=selected_feature,
        title=f'{selected_feature} over time (sample)'
    )
    st.plotly_chart(fig, width='stretch')
