"""
GKFS Auto Switch - Interactive ML Training & Forecasting Dashboard
===================================================================

Complete end-to-end Streamlit application for power plant forecasting with:
- Dynamic plant and data selection
- Interactive feature engineering
- Model training with hyperparameter tuning
- FSP selection with MAE scores
- Interactive visualizations with Plotly
- Prediction export and comparison

Maintainer: Project Team
Date: January 2026
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Import page modules
from app.pages import (
    data_selection,
    feature_engineering,
    fsp_selection,
    model_training,
    predictions_viz,
    model_comparison
)

# Page configuration
st.set_page_config(
    page_title="GKFS Auto Switch - FSP Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Data selection
        'plant_selected': None,
        'data_loaded': False,
        'df_raw': None,
        'df_pivoted': None,
        'data_months': 18,
        'data_date_range': None,

        # Data splitting
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'split_complete': False,
        'train_df': None,
        'val_df': None,
        'test_df': None,

        # Feature engineering
        'features_created': False,
        'feature_columns': [],
        'scaler': None,
        'encoders': None,
        'imputer': None,

        # FSP selection
        'fsp_list': [],
        'selected_fsps': [],
        'fsp_mae_scores': {},

        # Model training
        'models_trained': {},
        'model_predictions': {},
        'model_configs': {},
        'model_metrics': {},
        'selected_models': [],
        'ensemble_config': None,
        'loaded_model_bundle': None,
        'loaded_model_name': None,
        'loaded_model_test_metrics': None,
        'loaded_model_predictions': None,
        'loaded_model_source': None,
        'loaded_model_path': "",

        # Predictions
        'predictions_generated': False,
        'prediction_dfs': {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application entry point."""
    initialize_session_state()

    # Header
    st.title(" GKFS Auto Switch - FSP Forecasting Platform")
    st.markdown("### Interactive ML Training & Prediction Dashboard")
    st.markdown("---")

    # Sidebar navigation
    with st.sidebar:
        st.markdown("###  **GKFS Auto Switch**")
        st.markdown("### Navigation")

        page = st.radio(
            "Select Workflow Stage:",
            [
                " 1. Data Selection & Loading",
                " 2. Feature Engineering",
                " 3. FSP Selection",
                " 4. Model Training",
                " 5. Predictions & Visualization",
                " 6. Model Comparison"
            ],
            key="navigation"
        )

        st.markdown("---")

        # Quick status overview
        st.markdown("###  Pipeline Status")
        status_indicators = {
            "Data Loaded": st.session_state.data_loaded,
            "Features Created": st.session_state.features_created,
            "Models Trained": len(st.session_state.models_trained) > 0,
            "Predictions Generated": st.session_state.predictions_generated
        }

        for label, status in status_indicators.items():
            icon = "" if status else ""
            st.markdown(f"{icon} {label}")

        st.markdown("---")

        # Reset Button (Dev Tool)
        if st.button(" Reset Application", type="secondary", help="Clear all data and restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("**Version:** 1.0.0  \n**Updated:** Jan 2026  \n**Maintainer:** Project Team  \n**Team:** AI/ML Team")


    # Route to appropriate page
    if "1. Data Selection" in page:
        data_selection.show()
    elif "2. Feature Engineering" in page:
        feature_engineering.show()
    elif "3. FSP Selection" in page:
        fsp_selection.show()
    elif "4. Model Training" in page:
        model_training.show()
    elif "5. Predictions" in page:
        predictions_viz.show()
    elif "6. Model Comparison" in page:
        model_comparison.show()


if __name__ == "__main__":
    main()
