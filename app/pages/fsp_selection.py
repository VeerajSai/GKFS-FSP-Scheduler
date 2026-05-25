"""
FSP Selection Page
==================

Handles:
- Dynamic FSP detection
- Historical MAE calculation for each FSP
- Interactive FSP selection with sorting
- FSP performance visualization

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

from src.data.preprocessing import get_fsp_forecast_columns, get_available_fsps, FSP_PROVIDERS
from app.utils.page_summary import render_page_summary, get_page_context


def show():
    """Main function for FSP selection page."""
    st.header(" FSP Selection & Performance Analysis")
    st.markdown("Select FSPs to include in model training based on historical performance")

    # AI-generated page summary
    render_page_summary("FSP Selection", get_page_context("FSP Selection"))

    # Check if features are created
    if not st.session_state.features_created:
        st.warning(" Please complete **Feature Engineering** first")
        return

    train_df = st.session_state.train_df

    # Step 1: Detect FSPs
    st.markdown("### Step 1: Detect Available FSPs")

    fsp_cols = get_fsp_forecast_columns(train_df)
    fsp_names = [col.replace('forecast_power_', '').upper() for col in fsp_cols]

    st.session_state.fsp_list = fsp_names

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric("Available FSPs", len(fsp_names))

    with col2:
        st.info(f" Detected FSPs: {', '.join(fsp_names)}")

    # Step 2: Calculate Historical MAE
    st.markdown("### Step 2: Calculate Historical MAE Scores")
    st.info(" MAE (Mean Absolute Error) measures average forecast error. Lower is better.")

    if st.button(" Calculate MAE Scores", type="primary"):
        with st.spinner("Calculating MAE for each FSP..."):
            mae_scores = calculate_fsp_mae_scores(train_df, fsp_cols)
            st.session_state.fsp_mae_scores = mae_scores

            # Display results
            display_fsp_performance(mae_scores, fsp_names)

    # Step 3: FSP Selection
    if st.session_state.fsp_mae_scores:
        st.markdown("---")
        st.markdown("### Step 3: Select FSPs for Training")

        mae_scores = st.session_state.fsp_mae_scores

        # Sort options
        col1, col2 = st.columns([1, 3])

        with col1:
            sort_by = st.selectbox(
                "Sort FSPs by:",
                ["MAE (Low to High)", "MAE (High to Low)", "Alphabetical"],
                help="Sort FSPs to make selection easier"
            )

        # Sort FSP names based on selection
        if sort_by == "MAE (Low to High)":
            sorted_fsps = sorted(fsp_names, key=lambda x: mae_scores.get(x, float('inf')))
        elif sort_by == "MAE (High to Low)":
            sorted_fsps = sorted(fsp_names, key=lambda x: mae_scores.get(x, float('inf')), reverse=True)
        else:
            sorted_fsps = sorted(fsp_names)

        # Handle quick actions BEFORE widget creation to avoid session_state conflicts
        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("##### Quick Actions")

            if st.button("Select All", key="btn_select_all"):
                for fsp in fsp_names:
                    st.session_state[f"fsp_select_{fsp}"] = True
                st.rerun()

            if st.button("Select None", key="btn_select_none"):
                for fsp in fsp_names:
                    st.session_state[f"fsp_select_{fsp}"] = False
                st.rerun()

            if st.button("Select Top 3", key="btn_select_top3"):
                sorted_by_mae = sorted(fsp_names, key=lambda x: mae_scores.get(x, float('inf')))
                for fsp in fsp_names:
                    st.session_state[f"fsp_select_{fsp}"] = fsp in sorted_by_mae[:3]
                st.rerun()

        # Selection interface
        st.markdown("#### Select FSPs to include:")

        with col1:
            # Create checkboxes for each FSP
            selected_fsps = []

            # Initialize default values in session state if not present
            for fsp in sorted_fsps:
                if f"fsp_select_{fsp}" not in st.session_state:
                    st.session_state[f"fsp_select_{fsp}"] = True  # Default: all selected

            for fsp in sorted_fsps:
                mae = mae_scores.get(fsp, 0)

                # Color coding based on performance
                if mae < 3.0:
                    indicator = ""  # Excellent
                elif mae < 5.0:
                    indicator = ""  # Good
                else:
                    indicator = ""  # Poor

                is_selected = st.checkbox(
                    f"{indicator} {fsp} (MAE: {mae:.3f} MW)",
                    value=st.session_state[f"fsp_select_{fsp}"],
                    key=f"fsp_select_{fsp}"
                )

                if is_selected:
                    selected_fsps.append(fsp)

        # Update session state
        st.session_state.selected_fsps = selected_fsps

        # Summary
        st.markdown("---")
        st.markdown("#### Selection Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total FSPs", len(fsp_names))

        with col2:
            st.metric("Selected FSPs", len(selected_fsps))

        with col3:
            if selected_fsps:
                avg_mae = np.mean([mae_scores.get(fsp, 0) for fsp in selected_fsps])
                st.metric("Avg MAE (Selected)", f"{avg_mae:.3f} MW")

        if selected_fsps:
            st.success(f" Selected: {', '.join(selected_fsps)}")
            st.info(" FSP selection complete! Proceed to **Model Training** ")
        else:
            st.warning(" Please select at least one FSP")

    # Step 4: Detailed Performance Analysis
    if st.session_state.fsp_mae_scores:
        st.markdown("---")
        st.markdown("### Step 4: Detailed Performance Analysis")

        analyze_fsp_performance_over_time(train_df, fsp_cols)


def calculate_fsp_mae_scores(df, fsp_cols):
    """Calculate MAE for each FSP compared to actual power."""
    TARGET = 'actual_power'
    mae_scores = {}

    for fsp_col in fsp_cols:
        fsp_name = fsp_col.replace('forecast_power_', '').upper()

        # Calculate MAE
        valid_mask = df[fsp_col].notna() & df[TARGET].notna()
        if valid_mask.sum() > 0:
            mae = np.abs(df.loc[valid_mask, fsp_col] - df.loc[valid_mask, TARGET]).mean()
            mae_scores[fsp_name] = mae
        else:
            mae_scores[fsp_name] = float('inf')

    return mae_scores


def display_fsp_performance(mae_scores, fsp_names):
    """Display FSP performance metrics and visualizations."""
    # Create DataFrame
    performance_df = pd.DataFrame({
        'FSP': list(mae_scores.keys()),
        'MAE (MW)': list(mae_scores.values())
    }).sort_values('MAE (MW)')

    # Add rank
    performance_df['Rank'] = range(1, len(performance_df) + 1)

    # Add performance category
    performance_df['Category'] = performance_df['MAE (MW)'].apply(
        lambda x: ' Excellent' if x < 3.0 else (' Good' if x < 5.0 else ' Needs Improvement')
    )

    # Reorder columns
    performance_df = performance_df[['Rank', 'FSP', 'MAE (MW)', 'Category']]

    # Display table
    st.markdown("#### FSP Performance Rankings")
    st.dataframe(performance_df)

    # Bar chart
    fig = px.bar(
        performance_df,
        x='MAE (MW)',
        y='FSP',
        orientation='h',
        title='FSP Performance Comparison (Lower MAE is Better)',
        color='MAE (MW)',
        color_continuous_scale='RdYlGn_r',
        text='MAE (MW)'
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=400)

    st.plotly_chart(fig, width='stretch')

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_fsp = performance_df.iloc[0]
        st.metric(" Best FSP", best_fsp['FSP'], f"{best_fsp['MAE (MW)']:.3f} MW")

    with col2:
        worst_fsp = performance_df.iloc[-1]
        st.metric(" Worst FSP", worst_fsp['FSP'], f"{worst_fsp['MAE (MW)']:.3f} MW")

    with col3:
        avg_mae = performance_df['MAE (MW)'].mean()
        st.metric(" Average MAE", f"{avg_mae:.3f} MW")

    with col4:
        std_mae = performance_df['MAE (MW)'].std()
        st.metric(" Std Dev", f"{std_mae:.3f} MW")


def analyze_fsp_performance_over_time(df, fsp_cols):
    """Analyze FSP performance over time - dynamically filtered by selected FSPs."""
    TARGET = 'actual_power'
    date_col = 'timestamp' if 'timestamp' in df.columns else 'date'

    # Get selected FSPs from session state
    selected_fsps = st.session_state.get('selected_fsps', [])

    if not selected_fsps:
        st.info("i Select FSPs above to view detailed performance analysis")
        return

    # Filter fsp_cols to only selected FSPs
    filtered_fsp_cols = [
        col for col in fsp_cols
        if col.replace('forecast_power_', '').upper() in selected_fsps
    ]

    if not filtered_fsp_cols:
        return

    with st.expander(" View Performance Over Time (Selected FSPs Only)", expanded=False):
        # Resample to daily for better visualization
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.set_index(date_col)

        # Calculate daily MAE for each SELECTED FSP
        st.markdown("#### Daily MAE Trends (Selected FSPs)")
        st.info(f" Showing: {', '.join(selected_fsps)}")

        fig = go.Figure()

        # Define unique colors for consistency
        fsp_colors = {
            'FA_PROVIDER_A': '#1f77b4',
            'FA_PROVIDER_B': '#ff7f0e',
            'FA_PROVIDER_C': '#2ca02c',
            'FA_TECHDEV': '#d62728',
            'FA_KONA': '#9467bd',
            'FA_CUSTOM': '#8c564b',
            'DF_PERSIST': '#e377c2',
        }

        for fsp_col in filtered_fsp_cols:
            fsp_name = fsp_col.replace('forecast_power_', '').upper()

            # Calculate daily MAE
            daily_mae = df_copy.resample('D').apply(
                lambda x: np.abs(x[fsp_col] - x[TARGET]).mean() if len(x) > 0 else np.nan
            )

            # Drop NaN values
            daily_mae = daily_mae.dropna()

            if len(daily_mae) > 0:
                color = fsp_colors.get(fsp_name, px.colors.qualitative.Set3[len(fig.data) % len(px.colors.qualitative.Set3)])
                fig.add_trace(go.Scatter(
                    x=daily_mae.index,
                    y=daily_mae.values,
                    mode='lines',
                    name=fsp_name,
                    line=dict(width=2, color=color)
                ))

        fig.update_layout(
            title='Daily MAE Trends by Selected FSP',
            xaxis_title='Date',
            yaxis_title='MAE (MW)',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, width='stretch')

        # Hourly performance
        st.markdown("#### Performance by Hour of Day (Selected FSPs)")

        df_hourly = df.copy()
        df_hourly[date_col] = pd.to_datetime(df_hourly[date_col])

        if 'hour' not in df_hourly.columns:
            df_hourly['hour'] = df_hourly[date_col].dt.hour

        hourly_mae = []

        for fsp_col in filtered_fsp_cols:
            fsp_name = fsp_col.replace('forecast_power_', '').upper()

            for hour in range(24):
                hour_data = df_hourly[df_hourly['hour'] == hour]
                if len(hour_data) > 0:
                    mae = np.abs(hour_data[fsp_col] - hour_data[TARGET]).mean()
                    hourly_mae.append({
                        'Hour': hour,
                        'FSP': fsp_name,
                        'MAE': mae
                    })

        hourly_df = pd.DataFrame(hourly_mae)

        fig = px.line(
            hourly_df,
            x='Hour',
            y='MAE',
            color='FSP',
            title='FSP Performance by Hour of Day (Selected FSPs)',
            markers=True
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
