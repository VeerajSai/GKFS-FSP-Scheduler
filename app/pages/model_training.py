"""
Model Training Page
===================

Handles:
- Model selection (Classical ML, Deep Learning, Hybrid)
- Hyperparameter configuration
- Model training with progress tracking
- Performance evaluation and comparison
- Model saving

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
import pickle
import json
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.models.ensemble_model import RidgeLightGBMEnsemble
from app.utils.page_summary import render_page_summary, get_page_context

# Import model builders
try:
    from app.utils.model_builders import (
        build_ann_model, build_fcnn_model, build_lstm_model, build_gru_model,
        build_temporal_cnn_model, build_custom_architecture, build_cnn_bilstm_model,
        build_informer_model, reshape_for_rnn, get_callbacks,
        build_harmonic_regression_features, prepare_ceemdan_vmd_features,
        prepare_ivmd_fe_features
    )
    MODEL_BUILDERS_AVAILABLE = True
except ImportError:
    MODEL_BUILDERS_AVAILABLE = False


def show():
    """Main function for model training page."""
    st.header(" Model Training & Evaluation")
    st.markdown("Select models, configure hyperparameters, and train")

    # AI-generated page summary
    render_page_summary("Model Training", get_page_context("Model Training"))

    # Check if FSPs are selected
    if not st.session_state.selected_fsps:
        st.warning(" Please select FSPs in **FSP Selection** page first")
        return

    # Step 1: Model Selection
    st.markdown("### Step 1: Select Models or Load Pre-trained")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Classical ML",
        " Deep Learning",
        " Hybrid/Ensemble",
        " Load Model",
        " Summary"
    ])

    selected_models = {}

    with tab1:
        selected_models.update(select_classical_ml_models())

    with tab2:
        selected_models.update(select_deep_learning_models())

    with tab3:
        selected_models.update(select_ensemble_models())

    with tab4:
        load_pretrained_model_section()

    with tab5:
        display_model_selection_summary(
            selected_models,
            loaded_model_bundle=st.session_state.get('loaded_model_bundle')
        )

    st.session_state.selected_models = list(selected_models.keys())

    # Step 2: Training & Testing Controls
    st.markdown("---")
    st.markdown("### Step 2: Training & Testing Controls")
    train_disabled = not bool(selected_models)
    testing_disabled = st.session_state.get('loaded_model_bundle') is None
    col_train, col_test = st.columns(2)

    with col_train:
        if st.button(" Training & Validation", type="primary", disabled=train_disabled):
            train_all_models(selected_models)
        if train_disabled:
            st.caption("Select at least one model above to enable training.")

    with col_test:
        if st.button(" Testing Only", type="secondary", disabled=testing_disabled):
            evaluate_loaded_model_on_test()
        if testing_disabled:
            st.caption("Load a pre-trained bundle to unlock test-only evaluation.")
        else:
            st.caption("Runs the loaded model on the held-out test split without retraining.")

    # Step 3: View Results
    if st.session_state.models_trained and st.session_state.model_metrics:
        st.markdown("---")
        st.markdown("### Step 3: Training Results")

        display_training_results()

    if st.session_state.get('loaded_model_test_metrics'):
        st.markdown("---")
        render_loaded_model_test_results()


def select_classical_ml_models():
    """Interface for selecting classical ML models with hyperparameters."""
    st.markdown("#### Classical Machine Learning Models")

    selected = {}

    col1, col2 = st.columns(2)

    with col1:
        # Linear Regression
        st.markdown("#####  Linear Regression")
        use_ridge = st.checkbox("Ridge Regression", value=True, key="use_ridge")
        if use_ridge:
            # Adjusted min to 0.0 to avoid step conflict with value=1.0 (1.0 is multiple of 0.1)
            alpha = st.slider("Alpha (Regularization)", 0.0, 10.0, 1.0, 0.1, key="ridge_alpha")
            selected['ridge'] = {'alpha': alpha}

        use_lasso = st.checkbox("Lasso Regression", value=False, key="use_lasso")
        if use_lasso:
            # Adjusted min to 0.0 to avoid step conflict with value=1.0
            alpha = st.slider("Alpha (Regularization)", 0.0, 10.0, 1.0, 0.1, key="lasso_alpha")
            selected['lasso'] = {'alpha': alpha}

        st.markdown("#####  Harmonic Regression")
        use_harmonic = st.checkbox("Harmonic Regression", value=False, key="use_harmonic")
        if use_harmonic:
            periods = st.text_input("Periods (comma-separated)", "24,96", key="harmonic_periods")
            order = st.slider("Harmonic Order", 1, 5, 2, 1, key="harmonic_order")
            alpha = st.slider("Ridge Alpha", 0.0, 10.0, 0.5, 0.1, key="harmonic_alpha")
            use_original_features = st.checkbox(
                "Include Original Features",
                value=True,
                key="harmonic_use_original"
            )
            time_cols = ['timestamp', 'block']
            train_df = st.session_state.get('train_df')
            if train_df is not None:
                candidate_cols = [c for c in train_df.columns if c in ['timestamp', 'block']]
                if candidate_cols:
                    time_cols = candidate_cols
            time_col = st.selectbox("Time Column", time_cols, index=0, key="harmonic_time_col")

            selected['harmonic_regression'] = {
                'periods': [float(x.strip()) for x in periods.split(',') if x.strip()],
                'order': order,
                'alpha': alpha,
                'use_original_features': use_original_features,
                'time_col': time_col
            }

    with col2:
        # Random Forest
        st.markdown("#####  Random Forest")
        use_rf = st.checkbox("Random Forest", value=True, key="use_rf")
        if use_rf:
            n_estimators = st.slider("Number of Trees", 50, 500, 200, 50, key="rf_n_est")
            max_depth = st.slider("Max Depth", 5, 30, 15, 5, key="rf_depth")
            selected['random_forest'] = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'n_jobs': -1
            }

    col3, col4 = st.columns(2)

    with col3:
        # XGBoost
        st.markdown("#####  XGBoost")
        if XGB_AVAILABLE:
            use_xgb = st.checkbox("XGBoost", value=True, key="use_xgb")
            if use_xgb:
                n_estimators = st.slider("Number of Trees", 50, 500, 200, 50, key="xgb_n_est")
                max_depth = st.slider("Max Depth", 3, 15, 8, 1, key="xgb_depth")
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01, key="xgb_lr")
                selected['xgboost'] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
        else:
            st.warning(" XGBoost not installed")

    with col4:
        # LightGBM
        st.markdown("#####  LightGBM")
        if LGB_AVAILABLE:
            use_lgb = st.checkbox("LightGBM", value=True, key="use_lgb")
            if use_lgb:
                n_estimators = st.slider("Number of Trees", 50, 500, 200, 50, key="lgb_n_est")
                max_depth = st.slider("Max Depth", 3, 15, 10, 1, key="lgb_depth")
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01, key="lgb_lr")
                selected['lightgbm'] = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'verbose': -1
                }
        else:
            st.warning(" LightGBM not installed")

    return selected


def select_deep_learning_models():
    """Interface for selecting deep learning models."""
    st.markdown("#### Deep Learning Models")

    if not TF_AVAILABLE:
        st.warning(" TensorFlow not installed. Deep learning models unavailable.")
        return {}

    selected = {}

    # Row 1: ANN and Fully Connected NN
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#####  ANN (Artificial Neural Network)")
        use_ann = st.checkbox("ANN (Dense Layers)", value=False, key="use_ann")
        if use_ann:
            layers = st.text_input("Hidden Layers (comma-separated)", "256,128,64", key="ann_layers")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05, key="ann_dropout")
            activation = st.selectbox("Activation Function", ["relu", "tanh", "elu"], key="ann_activation")
            optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"], key="ann_optimizer")
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="ann_lr")
            dynamic_lr = st.checkbox("Dynamic Learning Rate (reduce on plateau)", value=True, key="ann_dynamic_lr")
            epochs = st.slider("Epochs", 50, 300, 100, 10, key="ann_epochs")
            batch_size = st.select_slider("Batch Size", [32, 64, 128, 256], value=64, key="ann_batch")

            selected['ann'] = {
                'layers': [int(x.strip()) for x in layers.split(',')],
                'dropout': dropout,
                'activation': activation,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'dynamic_lr': dynamic_lr,
                'epochs': epochs,
                'batch_size': batch_size
            }

    with col2:
        st.markdown("#####  Fully Connected NN")
        use_fcnn = st.checkbox("Deep Fully Connected NN", value=False, key="use_fcnn")
        if use_fcnn:
            num_layers = st.slider("Number of Hidden Layers", 2, 6, 4, key="fcnn_num_layers")
            layer_size = st.slider("Neurons per Layer", 32, 512, 128, 32, key="fcnn_layer_size")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05, key="fcnn_dropout")
            batch_norm = st.checkbox("Use Batch Normalization", value=True, key="fcnn_batchnorm")
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="fcnn_lr")
            dynamic_lr = st.checkbox("Dynamic Learning Rate", value=True, key="fcnn_dynamic_lr")

            selected['fcnn'] = {
                'num_layers': num_layers,
                'layer_size': layer_size,
                'dropout': dropout,
                'batch_norm': batch_norm,
                'learning_rate': learning_rate,
                'dynamic_lr': dynamic_lr,
                'epochs': 150,
                'batch_size': 64
            }

    # Row 2: LSTM and GRU
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#####  LSTM (Long Short-Term Memory)")
        use_lstm = st.checkbox("LSTM Network", value=False, key="use_lstm")
        if use_lstm:
            units = st.text_input("LSTM Units (comma-separated)", "128,64", key="lstm_units")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05, key="lstm_dropout")
            recurrent_dropout = st.slider("Recurrent Dropout", 0.0, 0.5, 0.2, 0.05, key="lstm_rec_dropout")
            bidirectional = st.checkbox("Use Bidirectional LSTM", value=False, key="lstm_bidirectional")
            timesteps = st.slider("Lookback Timesteps", 1, 10, 3, key="lstm_timesteps")
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="lstm_lr")
            dynamic_lr = st.checkbox("Dynamic Learning Rate", value=True, key="lstm_dynamic_lr")

            selected['lstm'] = {
                'units': [int(x.strip()) for x in units.split(',')],
                'dropout': dropout,
                'recurrent_dropout': recurrent_dropout,
                'bidirectional': bidirectional,
                'timesteps': timesteps,
                'learning_rate': learning_rate,
                'dynamic_lr': dynamic_lr,
                'epochs': 150,
                'batch_size': 64
            }

    with col4:
        st.markdown("#####  GRU (Gated Recurrent Unit)")
        use_gru = st.checkbox("GRU Network", value=False, key="use_gru")
        if use_gru:
            units = st.text_input("GRU Units (comma-separated)", "128,64", key="gru_units")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05, key="gru_dropout")
            recurrent_dropout = st.slider("Recurrent Dropout", 0.0, 0.5, 0.2, 0.05, key="gru_rec_dropout")
            bidirectional = st.checkbox("Use Bidirectional GRU", value=False, key="gru_bidirectional")
            timesteps = st.slider("Lookback Timesteps", 1, 10, 3, key="gru_timesteps")
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="gru_lr")
            dynamic_lr = st.checkbox("Dynamic Learning Rate", value=True, key="gru_dynamic_lr")

            selected['gru'] = {
                'units': [int(x.strip()) for x in units.split(',')],
                'dropout': dropout,
                'recurrent_dropout': recurrent_dropout,
                'bidirectional': bidirectional,
                'timesteps': timesteps,
                'learning_rate': learning_rate,
                'dynamic_lr': dynamic_lr,
                'epochs': 150,
                'batch_size': 64
            }

    # Row 3: Temporal CNN and Custom Architecture
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#####  Temporal CNN")
        use_tcnn = st.checkbox("Temporal Convolutional NN", value=False, key="use_tcnn")
        if use_tcnn:
            num_filters = st.slider("Number of Filters", 32, 256, 64, 32, key="tcnn_filters")
            kernel_size = st.slider("Kernel Size", 2, 7, 3, key="tcnn_kernel")
            num_conv_layers = st.slider("Conv Layers", 1, 4, 2, key="tcnn_conv_layers")
            pool_size = st.slider("Max Pool Size", 1, 3, 2, key="tcnn_pool")
            dense_units = st.text_input("Dense Layers", "128,64", key="tcnn_dense")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05, key="tcnn_dropout")
            timesteps = st.slider("Lookback Timesteps", 1, 10, 5, key="tcnn_timesteps")
            dynamic_lr = st.checkbox("Dynamic Learning Rate", value=True, key="tcnn_dynamic_lr")

            selected['temporal_cnn'] = {
                'num_filters': num_filters,
                'kernel_size': kernel_size,
                'num_conv_layers': num_conv_layers,
                'pool_size': pool_size,
                'dense_units': [int(x.strip()) for x in dense_units.split(',')],
                'dropout': dropout,
                'timesteps': timesteps,
                'learning_rate': 0.001,
                'dynamic_lr': dynamic_lr,
                'epochs': 100,
                'batch_size': 64
            }

    with col6:
        st.markdown("#####  Custom Architecture")
        use_custom = st.checkbox("Build Custom Architecture", value=False, key="use_custom_arch")
        if use_custom:
            st.info("Configure your custom neural network layer by layer")

            architecture_type = st.selectbox(
                "Base Architecture",
                ["Dense Only", "CNN + Dense", "LSTM + Dense", "GRU + Dense", "CNN + LSTM + Dense"],
                key="custom_arch_type"
            )

            if architecture_type == "Dense Only":
                layers_config = st.text_input("Dense Layers (comma-separated)", "256,128,64,32", key="custom_dense")
            elif architecture_type == "CNN + Dense":
                cnn_filters = st.text_input("CNN Filters (comma-separated)", "64,32", key="custom_cnn_filters")
                dense_layers = st.text_input("Dense Layers", "128,64", key="custom_cnn_dense")
                layers_config = {"cnn": cnn_filters, "dense": dense_layers}
            elif architecture_type in ["LSTM + Dense", "GRU + Dense"]:
                rnn_units = st.text_input("RNN Units (comma-separated)", "128,64", key="custom_rnn_units")
                dense_layers = st.text_input("Dense Layers", "64,32", key="custom_rnn_dense")
                layers_config = {"rnn": rnn_units, "dense": dense_layers}
            else:  # CNN + LSTM + Dense
                cnn_filters = st.text_input("CNN Filters", "64,32", key="custom_hybrid_cnn")
                lstm_units = st.text_input("LSTM Units", "64", key="custom_hybrid_lstm")
                dense_layers = st.text_input("Dense Layers", "32", key="custom_hybrid_dense")
                layers_config = {"cnn": cnn_filters, "lstm": lstm_units, "dense": dense_layers}

            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05, key="custom_dropout")
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key="custom_lr")
            dynamic_lr = st.checkbox("Dynamic Learning Rate", value=True, key="custom_dynamic_lr")

            selected['custom_architecture'] = {
                'architecture_type': architecture_type,
                'layers_config': layers_config,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'dynamic_lr': dynamic_lr,
                'epochs': 150,
                'batch_size': 64,
                'timesteps': 5 if 'CNN' in architecture_type or 'LSTM' in architecture_type or 'GRU' in architecture_type else 1
            }

    # Row 4: Advanced Hybrid Models
    st.markdown("---")
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("#####  CEEMDAN-VMD-CNN-BiLSTM")
        use_ceemdan_vmd = st.checkbox("CEEMDAN-VMD-CNN-BiLSTM", value=False, key="use_ceemdan_vmd")
        if use_ceemdan_vmd:
            timesteps = st.slider("Lookback Timesteps", 1, 12, 6, key="ceemdan_timesteps")
            num_filters = st.slider("CNN Filters", 16, 256, 64, 16, key="ceemdan_filters")
            kernel_size = st.slider("Kernel Size", 2, 7, 3, key="ceemdan_kernel")
            lstm_units = st.slider("BiLSTM Units", 16, 256, 64, 16, key="ceemdan_lstm_units")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05, key="ceemdan_dropout")
            decomp_window = st.slider("Decomposition Window", 3, 25, 9, 2, key="ceemdan_decomp_window")
            epochs = st.slider("Epochs", 50, 300, 120, 10, key="ceemdan_epochs")
            batch_size = st.select_slider("Batch Size", [32, 64, 128, 256], value=64, key="ceemdan_batch")

            selected['ceemdan_vmd_cnn_bilstm'] = {
                'timesteps': timesteps,
                'num_filters': num_filters,
                'kernel_size': kernel_size,
                'lstm_units': lstm_units,
                'dropout': dropout,
                'decomp_window': decomp_window,
                'learning_rate': 0.001,
                'epochs': epochs,
                'batch_size': batch_size,
                'dynamic_lr': True
            }

    with col8:
        st.markdown("#####  IVMD-FE-Ad-Informer")
        use_ivmd_informer = st.checkbox("IVMD-FE-Ad-Informer", value=False, key="use_ivmd_informer")
        if use_ivmd_informer:
            timesteps = st.slider("Lookback Timesteps", 1, 12, 6, key="ivmd_timesteps")
            d_model = st.slider("Model Dim (d_model)", 32, 256, 64, 32, key="ivmd_d_model")
            num_heads = st.slider("Attention Heads", 2, 8, 4, 1, key="ivmd_heads")
            ff_dim = st.slider("FFN Dim", 64, 512, 128, 32, key="ivmd_ff_dim")
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05, key="ivmd_dropout")
            decomp_window = st.slider("IVMD Window", 3, 25, 7, 2, key="ivmd_decomp_window")
            fuzzy_r = st.slider("Fuzzy Entropy r", 0.05, 0.5, 0.2, 0.05, key="ivmd_fuzzy_r")
            epochs = st.slider("Epochs", 50, 300, 120, 10, key="ivmd_epochs")
            batch_size = st.select_slider("Batch Size", [32, 64, 128, 256], value=64, key="ivmd_batch")

            selected['ivmd_fe_ad_informer'] = {
                'timesteps': timesteps,
                'd_model': d_model,
                'num_heads': num_heads,
                'ff_dim': ff_dim,
                'dropout': dropout,
                'decomp_window': decomp_window,
                'fuzzy_r': fuzzy_r,
                'learning_rate': 0.001,
                'epochs': epochs,
                'batch_size': batch_size,
                'dynamic_lr': True
            }

    return selected


def select_ensemble_models():
    """Interface for selecting ensemble models."""
    st.markdown("#### Hybrid & Ensemble Models")

    selected = {}

    # Row 1: Simple Averaging and Weighted Blending
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#####  Simple Averaging Ensemble")
        use_avg = st.checkbox("Average of Multiple Models", value=False, key="use_avg_ensemble")
        if use_avg:
            st.info("Will average predictions from all trained models")
            selected['averaging_ensemble'] = {
                'method': 'mean',
                'trim_outliers': st.checkbox("Trim Outliers (remove top/bottom 10%)", value=False, key="avg_trim")
            }

    with col2:
        st.markdown("#####  Weighted Blending")
        use_weighted = st.checkbox("Custom Weighted Blend", value=False, key="use_weighted_blend")
        if use_weighted:
            st.info("Configure weights after training individual models")

            # Predefined weight schemes
            weight_scheme = st.selectbox(
                "Weight Scheme",
                ["Equal Weights", "Performance-Based", "Custom Weights"],
                key="weight_scheme"
            )

            if weight_scheme == "Performance-Based":
                weight_method = st.selectbox(
                    "Weighting Method",
                    ["Inverse MAE", "Inverse RMSE", "R2 Score"],
                    key="weight_method"
                )
                selected['weighted_blending'] = {
                    'scheme': weight_scheme,
                    'method': weight_method
                }
            else:
                selected['weighted_blending'] = {
                    'scheme': weight_scheme
                }

    # Row 2: Stacking and Ridge-LightGBM
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#####  Stacking Ensemble")
        use_stacking = st.checkbox("Multi-Level Stacking", value=False, key="use_stacking")
        if use_stacking:
            st.markdown("**Level 0 (Base Models)**: Trained models will be base predictors")

            meta_model = st.selectbox(
                "Meta-Learner (Level 1)",
                ["Ridge Regression", "Lasso", "Random Forest", "XGBoost", "LightGBM", "Neural Network"],
                key="stacking_meta"
            )

            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, key="stacking_cv")
            use_original_features = st.checkbox(
                "Include Original Features in Meta-Learner",
                value=True,
                help="If checked, meta-learner will use both base predictions and original features",
                key="stacking_orig_features"
            )

            # Meta-model hyperparameters
            if meta_model == "Ridge Regression":
                meta_alpha = st.slider("Meta Ridge Alpha", 0.01, 10.0, 1.0, 0.1, key="stacking_ridge_alpha")
                meta_params = {'alpha': meta_alpha}
            elif meta_model == "Random Forest":
                meta_n_est = st.slider("Meta RF Trees", 50, 300, 100, 50, key="stacking_rf_trees")
                meta_params = {'n_estimators': meta_n_est, 'max_depth': 10}
            elif meta_model == "Neural Network":
                meta_layers = st.text_input("Meta NN Layers", "64,32", key="stacking_nn_layers")
                meta_params = {'layers': [int(x.strip()) for x in meta_layers.split(',')]}
            else:
                meta_params = {}

            selected['stacking_ensemble'] = {
                'meta_model': meta_model,
                'meta_params': meta_params,
                'cv_folds': cv_folds,
                'use_original_features': use_original_features
            }

    with col4:
        st.markdown("#####  Ridge + LightGBM Ensemble")
        if LGB_AVAILABLE:
            use_ridge_lgb = st.checkbox("Ridge-LightGBM Blend", value=False, key="use_ridge_lgb")
            if use_ridge_lgb:
                ridge_weight = st.slider(
                    "Ridge Weight",
                    0.0, 1.0, 0.4, 0.05,
                    help="Weight for Ridge (LightGBM gets 1 - Ridge weight)",
                    key="ensemble_ridge_weight"
                )

                # Advanced options
                with st.expander("Advanced Options"):
                    ridge_alpha = st.slider("Ridge Alpha", 0.1, 10.0, 1.0, 0.1, key="ridge_lgb_alpha")
                    lgb_n_est = st.slider("LightGBM Trees", 50, 500, 200, 50, key="ridge_lgb_n_est")
                    lgb_depth = st.slider("LightGBM Depth", 3, 15, 10, 1, key="ridge_lgb_depth")
                    lgb_lr = st.slider("LightGBM LR", 0.01, 0.3, 0.05, 0.01, key="ridge_lgb_lr")

                selected['ridge_lightgbm_ensemble'] = {
                    'ridge_weight': ridge_weight,
                    'ridge_alpha': ridge_alpha,
                    'lgb_n_estimators': lgb_n_est,
                    'lgb_max_depth': lgb_depth,
                    'lgb_learning_rate': lgb_lr
                }

                st.info(f"Ridge: {ridge_weight*100:.0f}% | LightGBM: {(1-ridge_weight)*100:.0f}%")
        else:
            st.warning(" LightGBM required for this ensemble")

    # Row 3: Hybrid Deep Learning Ensembles
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#####  Hybrid DL Ensemble")
        use_hybrid_dl = st.checkbox("Deep Learning Hybrid", value=False, key="use_hybrid_dl")
        if use_hybrid_dl:
            dl_models = st.multiselect(
                "Select DL Models to Ensemble",
                ["ANN", "LSTM", "GRU", "Temporal CNN"],
                default=["LSTM", "GRU"],
                key="hybrid_dl_models"
            )

            ensemble_method = st.selectbox(
                "Combination Method",
                ["Average", "Weighted Average", "Concatenate Features + Dense"],
                key="hybrid_dl_method"
            )

            if ensemble_method == "Concatenate Features + Dense":
                final_layers = st.text_input("Final Dense Layers", "64,32", key="hybrid_dl_dense")
                selected['hybrid_dl_ensemble'] = {
                    'models': dl_models,
                    'method': ensemble_method,
                    'final_layers': [int(x.strip()) for x in final_layers.split(',')],
                    'epochs': 100
                }
            else:
                selected['hybrid_dl_ensemble'] = {
                    'models': dl_models,
                    'method': ensemble_method
                }

    with col6:
        st.markdown("#####  ML + DL Hybrid")
        use_ml_dl = st.checkbox("Combine ML & Deep Learning", value=False, key="use_ml_dl_hybrid")
        if use_ml_dl:
            st.info("Combines traditional ML (RF, XGBoost) with Deep Learning (LSTM, GRU)")

            ml_weight = st.slider(
                "ML Weight (vs DL)",
                0.0, 1.0, 0.5, 0.05,
                help="Balance between ML and DL predictions",
                key="ml_dl_weight"
            )

            selected['ml_dl_hybrid'] = {
                'ml_weight': ml_weight,
                'dl_weight': 1 - ml_weight
            }

            st.info(f"ML: {ml_weight*100:.0f}% | DL: {(1-ml_weight)*100:.0f}%")

    return selected


def persist_loaded_model_bundle(bundle: dict, source_label: str) -> str:
    """Validate and store a loaded model bundle in the session state."""
    if not isinstance(bundle, dict) or 'model' not in bundle:
        raise ValueError("Model bundle must be a dictionary containing a 'model' key.")

    model_name = bundle.get('model_name') or bundle.get('name')
    if not model_name:
        try:
            model_name = Path(str(source_label)).stem
        except Exception:
            model_name = "Loaded Model"

    feature_columns = bundle.get('feature_columns') or st.session_state.get('feature_columns', [])
    imputer = bundle.get('imputer')
    scaler = bundle.get('scaler')
    model_config = bundle.get('config') or {}
    model_object = bundle['model']

    st.session_state.loaded_model_bundle = bundle
    st.session_state.loaded_model_name = model_name
    st.session_state.loaded_model_source = source_label
    st.session_state.loaded_model_test_metrics = None
    st.session_state.loaded_model_predictions = None

    # Align downstream workflow state with the loaded bundle
    st.session_state.feature_columns = feature_columns
    if imputer is not None:
        st.session_state.imputer = imputer
    if scaler is not None:
        st.session_state.scaler = scaler

    st.session_state.models_trained = {model_name: model_object}
    st.session_state.model_configs = {model_name: model_config}
    st.session_state.model_predictions = {}
    st.session_state.model_metrics = {}
    st.session_state.predictions_generated = False
    st.session_state.prediction_dfs = {}

    return model_name


def load_pretrained_model_section():
    """Allow users to load a previously trained model bundle for test-only evaluation."""
    st.markdown("####  Load Pre-trained Model")
    st.info("Upload a saved GKFS model bundle to skip training and run direct testing on the held-out split.")

    uploaded_bundle = st.file_uploader(
        "Upload bundle (.pkl/.joblib)",
        type=["pkl", "joblib"],
        key="load_model_uploader"
    )
    if uploaded_bundle is not None:
        try:
            bundle = pickle.load(uploaded_bundle)
            model_name = persist_loaded_model_bundle(bundle, "Uploaded File")
            st.success(f"Loaded **{model_name}** from uploaded bundle.")
        except Exception as exc:
            st.error(f"Failed to load uploaded bundle: {exc}")

    st.markdown("**Load from existing server path**")
    default_path = st.session_state.get('loaded_model_path', '')
    bundle_path = st.text_input(
        "Absolute path to bundle",
        value=default_path,
        placeholder="/home/.../model_bundle.pkl",
        key="load_model_path_field"
    )
    if st.button("Load from path", key="load_model_path_btn"):
        if not bundle_path:
            st.warning("Provide a valid path before loading.")
        else:
            path_obj = Path(bundle_path).expanduser()
            if not path_obj.exists():
                st.error("Provided path does not exist.")
            else:
                try:
                    with open(path_obj, 'rb') as infile:
                        bundle = pickle.load(infile)
                    st.session_state.loaded_model_path = str(path_obj)
                    model_name = persist_loaded_model_bundle(bundle, str(path_obj))
                    st.success(f"Loaded **{model_name}** from {path_obj}.")
                except Exception as exc:
                    st.error(f"Failed to load bundle from path: {exc}")

    bundle = st.session_state.get('loaded_model_bundle')
    if bundle:
        model_name = st.session_state.get('loaded_model_name', 'Loaded Model')
        feature_count = len(bundle.get('feature_columns') or [])
        has_scaler = 'Yes' if bundle.get('scaler') is not None else 'No'
        has_imputer = 'Yes' if bundle.get('imputer') is not None else 'No'
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", model_name)
        with col2:
            st.metric("Feature Count", feature_count)
        with col3:
            st.metric("Scaler / Imputer", f"{has_scaler} / {has_imputer}")
        st.caption(f"Source: {st.session_state.get('loaded_model_source', 'N/A')}. Use the Testing button in Step 2 to evaluate it.")


def evaluate_loaded_model_on_test():
    """Run the loaded model bundle on the held-out test dataset only."""
    bundle = st.session_state.get('loaded_model_bundle')
    if not bundle:
        st.warning("Load a model bundle first.")
        return

    test_df = st.session_state.get('test_df')
    if test_df is None or test_df.empty:
        st.warning("Test dataset is unavailable. Complete data preparation first.")
        return

    target_col = 'target_horizon'
    if target_col not in test_df.columns:
        st.error("Test dataframe does not contain 'target_horizon'.")
        return

    feature_cols = bundle.get('feature_columns') or st.session_state.get('feature_columns', [])
    if not feature_cols:
        st.error("Feature column metadata missing in the bundle.")
        return

    missing_cols = [col for col in feature_cols if col not in test_df.columns]
    if missing_cols:
        preview = ', '.join(missing_cols[:5])
        suffix = '...' if len(missing_cols) > 5 else ''
        st.error(f"Test data is missing required columns: {preview}{suffix}")
        return

    model = bundle.get('model')
    if model is None:
        st.error("Loaded bundle does not contain a model object.")
        return

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    imputer = bundle.get('imputer')
    scaler = bundle.get('scaler')

    try:
        if imputer is not None:
            X_test = imputer.transform(X_test)
        else:
            X_test = X_test.to_numpy()

        if scaler is not None:
            X_test = scaler.transform(X_test)
        else:
            X_test = np.asarray(X_test)
    except Exception as exc:
        st.error(f"Failed to prepare features for testing: {exc}")
        return

    try:
        with st.spinner("Evaluating loaded model on test split..."):
            predictions = model.predict(X_test)
    except Exception as exc:
        st.error(f"Model inference failed: {exc}")
        return

    predictions = np.maximum(np.array(predictions).flatten(), 0)
    metrics = calculate_metrics(y_test, predictions)

    st.session_state.loaded_model_test_metrics = metrics
    st.session_state.loaded_model_predictions = predictions.tolist()

    model_name = st.session_state.get('loaded_model_name', 'Loaded Model')

    # --- NEW: Create prediction DataFrame for visualization page ---
    # Import helper functions from predictions_viz
    from app.pages.predictions_viz import select_best_fsp_by_prediction, create_prediction_dataframe

    # Select FSPs and create proper output DataFrame
    selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(
        test_df, predictions
    )

    output_df = create_prediction_dataframe(
        test_df,
        predictions,
        selected_fsps,
        scheduled_power,
        confidence,
        model_name
    )

    # Store in prediction_dfs for visualization page
    st.session_state.prediction_dfs = {model_name: output_df}
    st.session_state.predictions_generated = True
    # --- END NEW ---

    existing_metrics = dict(st.session_state.get('model_metrics', {}))
    bundle_metrics = bundle.get('metrics') if isinstance(bundle.get('metrics'), dict) else {}
    validation_metrics = (
        bundle.get('validation_metrics')
        or bundle.get('val_metrics')
        or (bundle_metrics.get('validation') if isinstance(bundle_metrics.get('validation'), dict) else None)
        or (bundle_metrics.get('val') if isinstance(bundle_metrics.get('val'), dict) else None)
    )
    if not isinstance(validation_metrics, dict):
        validation_metrics = metrics
    existing_metrics[model_name] = {
        'val': validation_metrics,
        'test': metrics
    }
    st.session_state.model_metrics = existing_metrics

    st.success(f"Test-only evaluation complete for {st.session_state.get('loaded_model_name', 'Loaded Model')}.")


def render_loaded_model_test_results():
    """Display metrics for the loaded model's test-only evaluation."""
    metrics = st.session_state.get('loaded_model_test_metrics')
    if not metrics:
        return

    model_name = st.session_state.get('loaded_model_name', 'Loaded Model')
    st.markdown(f"### Loaded Model Test Report ({model_name})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test MAE", f"{metrics['MAE']:.3f} MW")
    with col2:
        st.metric("Test RMSE", f"{metrics['RMSE']:.3f} MW")
    with col3:
        st.metric("Test R2", f"{metrics['R2']:.3f}")
    with col4:
        st.metric("Test sMAPE", f"{metrics['sMAPE']:.2f}%")

    detail_df = pd.DataFrame([
        {
            'Metric': 'MAE',
            'Value': metrics['MAE'],
            'Unit': 'MW'
        },
        {
            'Metric': 'RMSE',
            'Value': metrics['RMSE'],
            'Unit': 'MW'
        },
        {
            'Metric': 'R2',
            'Value': metrics['R2'],
            'Unit': 'Score'
        },
        {
            'Metric': 'sMAPE',
            'Value': metrics['sMAPE'],
            'Unit': '%'
        }
    ])

    st.dataframe(detail_df, hide_index=True, use_container_width=True)


def display_model_selection_summary(selected_models, loaded_model_bundle=None):
    """Display summary of selected models."""
    st.markdown("#### Selected Models Summary")

    if selected_models:
        st.success(f" {len(selected_models)} models selected for training")
        summary_data = []
        for model_name, config in selected_models.items():
            model_type = categorize_model(model_name)
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Type': model_type,
                'Key Parameters': format_config(config)
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
    else:
        st.info("No models selected yet. Choose models from the tabs above.")

    if loaded_model_bundle:
        model_name = st.session_state.get('loaded_model_name', 'Loaded Model')
        feature_count = len(loaded_model_bundle.get('feature_columns') or [])
        st.success(f" Pre-trained model ready: **{model_name}** (features: {feature_count})")


def categorize_model(model_name):
    """Categorize model type."""
    if model_name in ['ridge', 'lasso', 'harmonic_regression']:
        return 'Linear'
    elif model_name in ['random_forest', 'xgboost', 'lightgbm']:
        return 'Tree-based'
    elif model_name in [
        'ann', 'fcnn', 'lstm', 'gru', 'temporal_cnn', 'custom_architecture',
        'ceemdan_vmd_cnn_bilstm', 'ivmd_fe_ad_informer'
    ]:
        return 'Deep Learning'
    elif 'ensemble' in model_name or 'hybrid' in model_name or 'blending' in model_name or 'stacking' in model_name or 'averaging' in model_name:
        return 'Ensemble'
    else:
        return 'Other'


def format_config(config):
    """Format configuration for display."""
    key_params = []
    for key, value in list(config.items())[:3]:  # Show top 3 params
        key_params.append(f"{key}={value}")
    return ', '.join(key_params)


def train_all_models(selected_models):
    """Train all selected models."""
    st.markdown("#### Training Progress")

    # Prepare data
    train_df = st.session_state.train_df
    val_df = st.session_state.val_df
    test_df = st.session_state.test_df
    feature_cols = st.session_state.feature_columns

    TARGET = 'target_horizon'

    # Get features and target
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET].copy()

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    st.session_state.scaler = scaler
    st.session_state.imputer = imputer

    input_dim = X_train_scaled.shape[1]

    # Progress tracking - create container for dynamic updates
    progress_container = st.container()
    with progress_container:
        overall_progress = st.progress(0)
        model_progress = st.progress(0)
        status_text = st.empty()

    trained_models = {}
    predictions = {}
    metrics = {}

    # First pass: Train individual models
    individual_models = {k: v for k, v in selected_models.items()
                        if not any(x in k for x in ['ensemble', 'blending', 'stacking', 'hybrid', 'averaging'])}

    total_models = len(selected_models)
    current_idx = 0

    for model_name, config in individual_models.items():
        current_idx += 1

        # Reset model-specific progress bar
        model_progress.progress(0)

        # Determine if this is a DL model
        is_dl_model = model_name in [
            'ann', 'fcnn', 'lstm', 'gru', 'temporal_cnn', 'custom_architecture',
            'ceemdan_vmd_cnn_bilstm', 'ivmd_fe_ad_informer'
        ]

        try:
            # ===== Classical ML Models =====
            if model_name == 'ridge':
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")
                model = Ridge(alpha=config['alpha'])
                model.fit(X_train_scaled, y_train)
                val_pred = np.maximum(model.predict(X_val_scaled), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test_scaled), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'lasso':
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")
                model = Lasso(alpha=config['alpha'])
                model.fit(X_train_scaled, y_train)
                val_pred = np.maximum(model.predict(X_val_scaled), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test_scaled), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'harmonic_regression':
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")
                alpha = config.get('alpha', 0.5)
                model = Ridge(alpha=alpha)

                X_train_h = build_harmonic_regression_features(X_train_scaled, train_df, config)
                X_val_h = build_harmonic_regression_features(X_val_scaled, val_df, config)
                X_test_h = build_harmonic_regression_features(X_test_scaled, test_df, config)

                model.fit(X_train_h, y_train)
                val_pred = np.maximum(model.predict(X_val_h), 0)
                test_pred = np.maximum(model.predict(X_test_h), 0)

                overall_progress.progress(current_idx / total_models)

            elif model_name == 'random_forest':
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")
                model = RandomForestRegressor(**config, random_state=42)
                model.fit(X_train, y_train)
                val_pred = np.maximum(model.predict(X_val), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test), 0)  # Clip to >= 0

            elif model_name == 'xgboost' and XGB_AVAILABLE:
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")
                model = xgb.XGBRegressor(**config, random_state=42)
                model.fit(X_train, y_train)
                val_pred = np.maximum(model.predict(X_val), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'lightgbm' and LGB_AVAILABLE:
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")
                model = lgb.LGBMRegressor(**config, random_state=42)
                model.fit(X_train, y_train)
                val_pred = np.maximum(model.predict(X_val), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            # ===== Deep Learning Models =====
            elif model_name == 'ann' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 100)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                # Import custom callback
                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, model_name.upper(), epochs)

                model = build_ann_model(input_dim, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=15, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = model.predict(X_val_scaled, verbose=0).flatten()
                test_pred = model.predict(X_test_scaled, verbose=0).flatten()
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'fcnn' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 150)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, model_name.upper(), epochs)

                model = build_fcnn_model(input_dim, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=15, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = model.predict(X_val_scaled, verbose=0).flatten()
                test_pred = model.predict(X_test_scaled, verbose=0).flatten()
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'lstm' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 150)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, model_name.upper(), epochs)

                # Reshape for LSTM
                timesteps = config.get('timesteps', 3)
                X_train_lstm = reshape_for_rnn(X_train_scaled, timesteps)
                X_val_lstm = reshape_for_rnn(X_val_scaled, timesteps)
                X_test_lstm = reshape_for_rnn(X_test_scaled, timesteps)

                model = build_lstm_model(input_dim, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_lstm, y_train,
                    validation_data=(X_val_lstm, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=20, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = np.maximum(model.predict(X_val_lstm, verbose=0).flatten(), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test_lstm, verbose=0).flatten(), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'gru' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 150)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, model_name.upper(), epochs)

                # Reshape for GRU
                timesteps = config.get('timesteps', 3)
                X_train_gru = reshape_for_rnn(X_train_scaled, timesteps)
                X_val_gru = reshape_for_rnn(X_val_scaled, timesteps)
                X_test_gru = reshape_for_rnn(X_test_scaled, timesteps)

                model = build_gru_model(input_dim, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_gru, y_train,
                    validation_data=(X_val_gru, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=20, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = np.maximum(model.predict(X_val_gru, verbose=0).flatten(), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test_gru, verbose=0).flatten(), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'temporal_cnn' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 100)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, model_name.upper(), epochs)

                # Reshape for CNN
                timesteps = config.get('timesteps', 5)
                X_train_cnn = reshape_for_rnn(X_train_scaled, timesteps)
                X_val_cnn = reshape_for_rnn(X_val_scaled, timesteps)
                X_test_cnn = reshape_for_rnn(X_test_scaled, timesteps)

                model = build_temporal_cnn_model(input_dim, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_cnn, y_train,
                    validation_data=(X_val_cnn, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=15, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = np.maximum(model.predict(X_val_cnn, verbose=0).flatten(), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test_cnn, verbose=0).flatten(), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'custom_architecture' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 150)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, 'CUSTOM', epochs)

                # Build custom architecture
                timesteps = config.get('timesteps', 1)
                if timesteps > 1:
                    X_train_custom = reshape_for_rnn(X_train_scaled, timesteps)
                    X_val_custom = reshape_for_rnn(X_val_scaled, timesteps)
                    X_test_custom = reshape_for_rnn(X_test_scaled, timesteps)
                else:
                    X_train_custom = X_train_scaled
                    X_val_custom = X_val_scaled
                    X_test_custom = X_test_scaled

                model = build_custom_architecture(input_dim, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_custom, y_train,
                    validation_data=(X_val_custom, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=15, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = np.maximum(model.predict(X_val_custom, verbose=0).flatten(), 0)  # Clip to >= 0
                test_pred = np.maximum(model.predict(X_test_custom, verbose=0).flatten(), 0)  # Clip to >= 0
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'ceemdan_vmd_cnn_bilstm' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 120)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, 'CEEMDAN-VMD-CNN-BiLSTM', epochs)

                X_train_aug = prepare_ceemdan_vmd_features(X_train_scaled, config)
                X_val_aug = prepare_ceemdan_vmd_features(X_val_scaled, config)
                X_test_aug = prepare_ceemdan_vmd_features(X_test_scaled, config)

                timesteps = config.get('timesteps', 6)
                X_train_seq = reshape_for_rnn(X_train_aug, timesteps)
                X_val_seq = reshape_for_rnn(X_val_aug, timesteps)
                X_test_seq = reshape_for_rnn(X_test_aug, timesteps)

                input_dim_aug = X_train_aug.shape[1]
                model = build_cnn_bilstm_model(input_dim_aug, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_seq, y_train,
                    validation_data=(X_val_seq, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=15, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = np.maximum(model.predict(X_val_seq, verbose=0).flatten(), 0)
                test_pred = np.maximum(model.predict(X_test_seq, verbose=0).flatten(), 0)
                overall_progress.progress(current_idx / total_models)

            elif model_name == 'ivmd_fe_ad_informer' and TF_AVAILABLE and MODEL_BUILDERS_AVAILABLE:
                epochs = config.get('epochs', 120)
                status_text.text(f"Training {model_name}... ({current_idx}/{total_models}) - Epoch 0/{epochs}")

                from app.utils.model_builders import StreamlitProgressCallback
                progress_cb = StreamlitProgressCallback(model_progress, status_text, 'IVMD-FE-Ad-Informer', epochs)

                X_train_aug = prepare_ivmd_fe_features(X_train_scaled, config)
                X_val_aug = prepare_ivmd_fe_features(X_val_scaled, config)
                X_test_aug = prepare_ivmd_fe_features(X_test_scaled, config)

                timesteps = config.get('timesteps', 6)
                X_train_seq = reshape_for_rnn(X_train_aug, timesteps)
                X_val_seq = reshape_for_rnn(X_val_aug, timesteps)
                X_test_seq = reshape_for_rnn(X_test_aug, timesteps)

                input_dim_aug = X_train_aug.shape[1]
                model = build_informer_model(input_dim_aug, config)
                use_dynamic_lr = config.get('dynamic_lr', True)
                history = model.fit(
                    X_train_seq, y_train,
                    validation_data=(X_val_seq, y_val),
                    epochs=epochs,
                    batch_size=config.get('batch_size', 64),
                    callbacks=get_callbacks(patience=15, use_dynamic_lr=use_dynamic_lr, progress_callback=progress_cb),
                    verbose=0
                )
                val_pred = np.maximum(model.predict(X_val_seq, verbose=0).flatten(), 0)
                test_pred = np.maximum(model.predict(X_test_seq, verbose=0).flatten(), 0)
                overall_progress.progress(current_idx / total_models)

            else:
                st.warning(f" Model {model_name} not available")
                continue

            # Store results
            trained_models[model_name] = model
            predictions[model_name] = {
                'val': val_pred,
                'test': test_pred
            }

            # Calculate metrics
            metrics[model_name] = {
                'val': calculate_metrics(y_val, val_pred),
                'test': calculate_metrics(y_test, test_pred)
            }

        except Exception as e:
            st.error(f" Error training {model_name}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Second pass: Train ensemble models
    ensemble_models = {k: v for k, v in selected_models.items()
                       if any(x in k for x in ['ensemble', 'blending', 'stacking', 'hybrid', 'averaging'])}

    for model_name, config in ensemble_models.items():
        current_idx += 1
        model_progress.progress(0)
        status_text.text(f"Training {model_name}... ({current_idx}/{total_models})")

        try:
            if model_name == 'ridge_lightgbm_ensemble' and LGB_AVAILABLE:
                # Train Ridge
                ridge = Ridge(alpha=config.get('ridge_alpha', 1.0))
                ridge.fit(X_train_scaled, y_train)

                # Train LightGBM
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=config.get('lgb_n_estimators', 200),
                    max_depth=config.get('lgb_max_depth', 10),
                    learning_rate=config.get('lgb_learning_rate', 0.05),
                    verbose=-1,
                    random_state=42
                )
                lgb_model.fit(X_train, y_train)

                # Create ensemble
                model = RidgeLightGBMEnsemble(
                    imputer=imputer,
                    ridge_model=ridge,
                    lightgbm_model=lgb_model,
                    ridge_weight=config['ridge_weight'],
                    scaler=scaler
                )

                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)

            elif model_name == 'averaging_ensemble':
                # Average all trained model predictions
                if not predictions:
                    st.warning(" No models trained yet for averaging")
                    continue

                val_preds = np.array([p['val'] for p in predictions.values()])
                test_preds = np.array([p['test'] for p in predictions.values()])

                if config.get('trim_outliers', False):
                    # Trim top and bottom 10% per timestamp
                    val_pred = np.mean(np.clip(val_preds,
                                              np.percentile(val_preds, 10, axis=0),
                                              np.percentile(val_preds, 90, axis=0)), axis=0)
                    test_pred = np.mean(np.clip(test_preds,
                                               np.percentile(test_preds, 10, axis=0),
                                               np.percentile(test_preds, 90, axis=0)), axis=0)
                else:
                    val_pred = np.mean(val_preds, axis=0)
                    test_pred = np.mean(test_preds, axis=0)

                model = {'type': 'averaging', 'config': config}

            elif model_name == 'weighted_blending':
                # Weighted blending of predictions
                if not predictions or not metrics:
                    st.warning(" No models trained yet for weighted blending")
                    continue

                scheme = config.get('scheme', 'Equal Weights')

                if scheme == 'Equal Weights':
                    weights = np.ones(len(predictions)) / len(predictions)
                elif scheme == 'Performance-Based':
                    method = config.get('method', 'Inverse MAE')
                    if method == 'Inverse MAE':
                        mae_vals = [metrics[m]['val']['MAE'] for m in predictions.keys()]
                        inv_mae = 1.0 / np.array(mae_vals)
                        weights = inv_mae / inv_mae.sum()
                    elif method == 'Inverse RMSE':
                        rmse_vals = [metrics[m]['val']['RMSE'] for m in predictions.keys()]
                        inv_rmse = 1.0 / np.array(rmse_vals)
                        weights = inv_rmse / inv_rmse.sum()
                    else:  # R2 Score
                        r2_vals = [metrics[m]['val']['R2'] for m in predictions.keys()]
                        weights = np.array(r2_vals) / np.array(r2_vals).sum()
                else:
                    weights = np.ones(len(predictions)) / len(predictions)

                val_preds = np.array([p['val'] for p in predictions.values()])
                test_preds = np.array([p['test'] for p in predictions.values()])

                val_pred = np.average(val_preds, axis=0, weights=weights)
                test_pred = np.average(test_preds, axis=0, weights=weights)

                model = {'type': 'weighted_blending', 'config': config, 'weights': weights.tolist()}

            elif model_name == 'stacking_ensemble':
                # Stacking with meta-learner
                if not predictions:
                    st.warning(" No base models trained yet for stacking")
                    continue

                # Filter out failed predictions (check if all have same shape)
                valid_predictions = {}
                expected_val_len = len(y_val)
                expected_test_len = len(y_test)

                for pred_name, pred_data in predictions.items():
                    if (len(pred_data['val']) == expected_val_len and
                        len(pred_data['test']) == expected_test_len):
                        valid_predictions[pred_name] = pred_data
                    else:
                        st.warning(f" Skipping {pred_name} in stacking (shape mismatch)")

                if len(valid_predictions) < 2:
                    st.warning(" Not enough valid models for stacking ensemble")
                    continue

                # Store base model names for later prediction
                base_model_names = list(valid_predictions.keys())

                # Create meta-features (base model predictions)
                meta_train = np.column_stack([p['val'] for p in valid_predictions.values()])
                meta_test = np.column_stack([p['test'] for p in valid_predictions.values()])

                # Optionally include original features
                if config.get('use_original_features', True):
                    meta_train = np.column_stack([meta_train, X_val_scaled])
                    meta_test = np.column_stack([meta_test, X_test_scaled])

                # Train meta-learner
                meta_model_name = config['meta_model']
                meta_params = config.get('meta_params', {})

                if meta_model_name == 'Ridge Regression':
                    meta_model = Ridge(**meta_params)
                elif meta_model_name == 'Lasso':
                    meta_model = Lasso(alpha=meta_params.get('alpha', 1.0))
                elif meta_model_name == 'Random Forest':
                    meta_model = RandomForestRegressor(**meta_params, random_state=42)
                elif meta_model_name == 'XGBoost' and XGB_AVAILABLE:
                    meta_model = xgb.XGBRegressor(**meta_params, random_state=42)
                elif meta_model_name == 'LightGBM' and LGB_AVAILABLE:
                    meta_model = lgb.LGBMRegressor(**meta_params, verbose=-1, random_state=42)
                else:
                    st.warning(f" Meta-model {meta_model_name} not available")
                    continue

                meta_model.fit(meta_train, y_val)
                val_pred = meta_model.predict(meta_train)
                test_pred = meta_model.predict(meta_test)

                model = {'type': 'stacking', 'meta_model': meta_model, 'config': config, 'base_models': base_model_names}

            else:
                st.warning(f" Ensemble {model_name} not implemented yet")
                continue

            # Store results
            trained_models[model_name] = model
            predictions[model_name] = {
                'val': val_pred,
                'test': test_pred
            }

            # Calculate metrics
            metrics[model_name] = {
                'val': calculate_metrics(y_val, val_pred),
                'test': calculate_metrics(y_test, test_pred)
            }

            overall_progress.progress(current_idx / total_models)

        except Exception as e:
            st.error(f" Error training ensemble {model_name}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    overall_progress.progress(1.0)
    model_progress.progress(1.0)
    status_text.text(" Training complete!")

    # Update session state
    st.session_state.models_trained = trained_models
    st.session_state.model_configs = selected_models  # Save configs for prediction
    st.session_state.model_predictions = predictions
    st.session_state.model_metrics = metrics

    st.success(f" Successfully trained {len(trained_models)} models!")

    # Force sidebar update
    st.rerun()


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe_denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'sMAPE': smape
    }


def save_models_section():
    """Section for saving trained models with user-defined names and versioning."""
    st.markdown("Select models to save with custom names and automatic versioning")

    trained_models = st.session_state.models_trained
    model_configs = st.session_state.model_configs
    metrics = st.session_state.model_metrics

    if not trained_models:
        st.warning("No models to save")
        return

    # Get plant name from session state
    plant_name = st.session_state.get('plant_selected', 'unknown_plant')

    # Create outputs directory structure
    base_output_dir = PROJECT_DIR / "outputs" / "saved_models"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Save options
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Select Models to Save:**")
        models_to_save = st.multiselect(
            "Choose models",
            options=list(trained_models.keys()),
            default=list(trained_models.keys())[:1] if trained_models else [],
            format_func=lambda x: x.replace('_', ' ').title(),
            key="models_to_save_select"
        )

    with col2:
        # Custom naming options
        st.markdown("**Naming Convention:**")
        use_custom_name = st.checkbox("Use custom name", value=False, key="use_custom_model_name")

        if use_custom_name:
            custom_prefix = st.text_input(
                "Custom prefix",
                value="",
                placeholder="e.g., production_model",
                key="custom_model_prefix"
            )
        else:
            custom_prefix = ""

    # Version management
    st.markdown("**Version Control:**")
    auto_version = st.checkbox("Auto-increment version", value=True, key="auto_version_models")

    if not auto_version:
        manual_version = st.text_input(
            "Manual version tag",
            value="v1.0",
            placeholder="e.g., v2.3 or beta",
            key="manual_version_tag"
        )

    # Additional metadata
    with st.expander(" Add Metadata (Optional)"):
        description = st.text_area(
            "Model description",
            placeholder="Brief description of this model version...",
            key="model_description"
        )
        tags = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., production, tuned, experimental",
            key="model_tags"
        )

    # Save button
    if st.button(" Save Selected Models", type="primary", key="save_models_btn"):
        if not models_to_save:
            st.warning("Please select at least one model to save")
            return

        with st.spinner("Saving models..."):
            saved_count = 0
            save_results = []

            for model_name in models_to_save:
                try:
                    # Get model and associated data
                    model = trained_models[model_name]
                    config = model_configs.get(model_name, {})
                    model_metrics = metrics.get(model_name, {})

                    # Determine version
                    if auto_version:
                        version = get_next_version(base_output_dir, plant_name, model_name, custom_prefix)
                    else:
                        version = manual_version

                    # Build filename
                    if custom_prefix:
                        filename_base = f"{custom_prefix}_{plant_name}_{model_name}_{version}"
                    else:
                        filename_base = f"{plant_name}_{model_name}_{version}"

                    # Create model-specific directory
                    model_dir = base_output_dir / plant_name / model_name / version
                    model_dir.mkdir(parents=True, exist_ok=True)

                    # Save model based on type
                    model_path = model_dir / f"{filename_base}.pkl"

                    # Handle different model types
                    if hasattr(model, 'save'):  # Keras/TF models
                        keras_path = model_dir / f"{filename_base}.h5"
                        model.save(str(keras_path))

                        # Save scaler and imputer separately for DL models
                        with open(model_dir / f"{filename_base}_scaler.pkl", 'wb') as f:
                            pickle.dump(st.session_state.scaler, f)
                        with open(model_dir / f"{filename_base}_imputer.pkl", 'wb') as f:
                            pickle.dump(st.session_state.imputer, f)

                        # ALSO save a unified bundle for easy loading
                        bundle_path = model_dir / f"{filename_base}_bundle.pkl"
                        with open(bundle_path, 'wb') as f:
                            pickle.dump({
                                'model': model,
                                'model_name': model_name,
                                'scaler': st.session_state.scaler,
                                'imputer': st.session_state.imputer,
                                'feature_columns': st.session_state.feature_columns,
                                'config': config,
                                'metrics': model_metrics
                            }, f)

                        save_info = {
                            'model_path': str(keras_path),
                            'bundle_path': str(bundle_path),
                            'scaler_path': str(model_dir / f"{filename_base}_scaler.pkl"),
                            'imputer_path': str(model_dir / f"{filename_base}_imputer.pkl")
                        }
                    elif hasattr(model, 'ridge_model') and hasattr(model, 'lightgbm_model'):
                        # Handle RidgeLightGBMEnsemble - save components separately to avoid pickle issues
                        # Save individual components
                        with open(model_dir / f"{filename_base}_ridge.pkl", 'wb') as f:
                            pickle.dump(model.ridge_model, f)
                        with open(model_dir / f"{filename_base}_lgbm.pkl", 'wb') as f:
                            pickle.dump(model.lightgbm_model, f)
                        with open(model_dir / f"{filename_base}_scaler.pkl", 'wb') as f:
                            pickle.dump(model.scaler if model.scaler else st.session_state.scaler, f)
                        with open(model_dir / f"{filename_base}_imputer.pkl", 'wb') as f:
                            pickle.dump(model.imputer if model.imputer else st.session_state.imputer, f)

                        # Save config for reconstruction
                        ensemble_config = {
                            'model_type': 'RidgeLightGBMEnsemble',
                            'model_name': model_name,
                            'ridge_weight': getattr(model, 'ridge_weight', 0.4),
                            'feature_columns': st.session_state.feature_columns,
                            'config': config,
                            'metrics': model_metrics
                        }
                        with open(model_dir / f"{filename_base}_config.json", 'w') as f:
                            json.dump(ensemble_config, f, indent=2, default=str)

                        save_info = {
                            'model_path': str(model_dir),
                            'components': ['ridge', 'lgbm', 'scaler', 'imputer', 'config']
                        }
                    else:  # Scikit-learn models or ensemble dicts
                        with open(model_path, 'wb') as f:
                            pickle.dump({
                                'model': model,
                                'model_name': model_name,
                                'scaler': st.session_state.scaler,
                                'imputer': st.session_state.imputer,
                                'feature_columns': st.session_state.feature_columns,
                                'config': config,
                                'metrics': model_metrics
                            }, f)
                        save_info = {'model_path': str(model_path)}

                    # Save metadata
                    metadata = {
                        'model_name': model_name,
                        'model_type': str(type(model).__name__),
                        'plant': plant_name,
                        'version': version,
                        'saved_date': datetime.now().isoformat(),
                        'config': config,
                        'metrics': {
                            'validation': model_metrics.get('val', {}),
                            'test': model_metrics.get('test', {})
                        },
                        'feature_columns': st.session_state.feature_columns,
                        'selected_fsps': st.session_state.selected_fsps,
                        'data_info': {
                            'train_samples': len(st.session_state.train_df),
                            'val_samples': len(st.session_state.val_df),
                            'test_samples': len(st.session_state.test_df),
                            'train_ratio': st.session_state.train_ratio,
                            'val_ratio': st.session_state.val_ratio,
                            'test_ratio': st.session_state.test_ratio
                        },
                        'description': description if description else f"Trained {model_name} model",
                        'tags': [t.strip() for t in tags.split(',')] if tags else []
                    }

                    metadata_path = model_dir / f"{filename_base}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4, default=str)

                    # Create a README
                    readme_path = model_dir / "README.md"
                    create_model_readme(readme_path, filename_base, metadata, save_info)

                    saved_count += 1
                    save_results.append({
                        'model': model_name,
                        'path': str(model_dir),
                        'version': version,
                        'status': ' Success'
                    })

                except Exception as e:
                    save_results.append({
                        'model': model_name,
                        'path': 'N/A',
                        'version': 'N/A',
                        'status': f' Error: {str(e)}'
                    })

            # Display results
            st.success(f" Successfully saved {saved_count} / {len(models_to_save)} models!")

            results_df = pd.DataFrame(save_results)
            st.dataframe(results_df)

            # Show saved directory
            if saved_count > 0:
                st.info(f" Models saved to: `{base_output_dir}`")


def get_next_version(base_dir: Path, plant: str, model_name: str, prefix: str = "") -> str:
    """Get the next version number for a model."""
    model_dir = base_dir / plant / model_name

    if not model_dir.exists():
        return "v1.0.0"

    # Find existing versions
    existing_versions = []
    for version_dir in model_dir.iterdir():
        if version_dir.is_dir():
            version_name = version_dir.name
            # Extract version number (e.g., v1.0.0 -> [1, 0, 0])
            if version_name.startswith('v'):
                try:
                    version_parts = version_name[1:].split('.')
                    # Ensure we have at least 3 parts by padding with zeros
                    while len(version_parts) < 3:
                        version_parts.append('0')
                    version_tuple = tuple(int(p) for p in version_parts[:3])  # Take only first 3
                    existing_versions.append(version_tuple)
                except ValueError:
                    continue

    if not existing_versions:
        return "v1.0.0"

    # Get the latest version and increment
    latest = max(existing_versions)
    new_version = (latest[0], latest[1], latest[2] + 1)
    return f"v{new_version[0]}.{new_version[1]}.{new_version[2]}"


def create_model_readme(path: Path, filename: str, metadata: dict, save_info: dict):
    """Create a README file for the saved model."""
    readme_content = f"""# Model: {metadata['model_name'].replace('_', ' ').title()}

## Overview
- **Version:** {metadata['version']}
- **Plant:** {metadata['plant']}
- **Model Type:** {metadata['model_type']}
- **Saved Date:** {metadata['saved_date']}

## Description
{metadata['description']}

## Performance Metrics

### Validation Set
- MAE: {metadata['metrics']['validation'].get('MAE', 'N/A')}
- RMSE: {metadata['metrics']['validation'].get('RMSE', 'N/A')}
- R2: {metadata['metrics']['validation'].get('R2', 'N/A')}
- sMAPE: {metadata['metrics']['validation'].get('sMAPE', 'N/A')}

### Test Set
- MAE: {metadata['metrics']['test'].get('MAE', 'N/A')}
- RMSE: {metadata['metrics']['test'].get('RMSE', 'N/A')}
- R2: {metadata['metrics']['test'].get('R2', 'N/A')}
- sMAPE: {metadata['metrics']['test'].get('sMAPE', 'N/A')}

## Configuration
```json
{json.dumps(metadata['config'], indent=2)}
```

## Training Data
- Train Samples: {metadata['data_info']['train_samples']}
- Validation Samples: {metadata['data_info']['val_samples']}
- Test Samples: {metadata['data_info']['test_samples']}
- Split Ratio: {metadata['data_info']['train_ratio']:.0%} / {metadata['data_info']['val_ratio']:.0%} / {metadata['data_info']['test_ratio']:.0%}

## Features
Selected FSPs: {', '.join(metadata['selected_fsps'])}

Number of Features: {len(metadata['feature_columns'])}

## Files
- Model: `{Path(save_info['model_path']).name}`
- Metadata: `{filename}_metadata.json`
{'- Scaler: `' + Path(save_info.get('scaler_path', '')).name + '`' if 'scaler_path' in save_info else ''}
{'- Imputer: `' + Path(save_info.get('imputer_path', '')).name + '`' if 'imputer_path' in save_info else ''}

## Tags
{', '.join(metadata['tags']) if metadata['tags'] else 'None'}

## Usage
```python
import pickle

# Load model
with open('{Path(save_info['model_path']).name}', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
imputer = model_data['imputer']
feature_columns = model_data['feature_columns']

# Make predictions
X_new = ... # Your new data
X_imputed = imputer.transform(X_new[feature_columns])
X_scaled = scaler.transform(X_imputed)
predictions = model.predict(X_scaled)
```

---
*Generated automatically by GKFS Auto Switch - FSP Forecasting Platform*
"""

    with open(path, 'w') as f:
        f.write(readme_content)


def display_training_results():
    """Display training results and comparisons."""
    metrics = st.session_state.model_metrics

    # Create comparison DataFrame
    comparison_data = []
    for model_name, model_metrics in metrics.items():
        val_metrics = model_metrics['val']
        test_metrics = model_metrics['test']

        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Val MAE': f"{val_metrics['MAE']:.3f}",
            'Val RMSE': f"{val_metrics['RMSE']:.3f}",
            'Val R2': f"{val_metrics['R2']:.3f}",
            'Test MAE': f"{test_metrics['MAE']:.3f}",
            'Test RMSE': f"{test_metrics['RMSE']:.3f}",
            'Test R2': f"{test_metrics['R2']:.3f}"
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.markdown("#### Performance Comparison")
    st.dataframe(comparison_df)

    # Visualization
    plot_model_comparison(metrics)

    # Best model
    best_model = min(metrics.items(), key=lambda x: x[1]['val']['MAE'])
    st.success(f" Best Model (Validation MAE): **{best_model[0].replace('_', ' ').title()}** ({best_model[1]['val']['MAE']:.3f} MW)")

    # Model Saving Section
    st.markdown("---")
    st.markdown("####  Save Trained Models")
    save_models_section()

    st.info(" Models trained! Proceed to **Predictions & Visualization** ")


def plot_model_comparison(metrics):
    """Plot model comparison charts."""
    # Prepare data
    models = []
    val_mae = []
    test_mae = []
    val_r2 = []
    test_r2 = []

    for model_name, model_metrics in metrics.items():
        models.append(model_name.replace('_', ' ').title())
        val_mae.append(model_metrics['val']['MAE'])
        test_mae.append(model_metrics['test']['MAE'])
        val_r2.append(model_metrics['val']['R2'])
        test_r2.append(model_metrics['test']['R2'])

    # MAE comparison
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Bar(name='Validation', x=models, y=val_mae, marker_color='#2E86AB'))
    fig_mae.add_trace(go.Bar(name='Test', x=models, y=test_mae, marker_color='#F18F01'))
    fig_mae.update_layout(
        title='MAE Comparison (Lower is Better)',
        xaxis_title='Model',
        yaxis_title='MAE (MW)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_mae, width='stretch')

    # R2 comparison
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(name='Validation', x=models, y=val_r2, marker_color='#2E86AB'))
    fig_r2.add_trace(go.Bar(name='Test', x=models, y=test_r2, marker_color='#F18F01'))
    fig_r2.update_layout(
        title='R2 Score Comparison (Higher is Better)',
        xaxis_title='Model',
        yaxis_title='R2 Score',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_r2, width='stretch')
