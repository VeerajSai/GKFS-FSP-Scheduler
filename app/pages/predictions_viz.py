"""
Predictions & Visualization Page
=================================

Handles:
- Generate predictions for all models
- Export predictions to CSV
- Interactive visualizations with Plotly
- Quantile forecasts
- Time series comparisons
- FSP selection visualization

Maintainer: Project Team
Date: January 2027
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import io
import zipfile
from scipy.stats import norm

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.preprocessing import FSP_PROVIDERS
from app.utils.model_builders import (
    reshape_for_rnn,
    build_harmonic_regression_features,
    prepare_ceemdan_vmd_features,
    prepare_ivmd_fe_features
)
from app.utils.ollama_insights import (
    OllamaInsightsGenerator,
    analyze_single_day_forecast,
    render_single_day_insights,
    setup_ollama_selector,
    get_available_models
)
from app.utils.page_summary import render_page_summary, get_page_context


def get_fsp_color_map():
    """Return a consistent, vibrant palette for all FSPs."""
    return {
        'FA_PROVIDER_A': '#d81b60',  # vivid red-magenta
        'FA_PROVIDER_B': '#7b1fa2',   # deep purple
        'FA_PROVIDER_C': '#5e60ce',     # bright violet
        'FA_TECHDEV': '#d4a017',      # rich gold
        'FA_KONA': '#b36b00',         # dark amber/yellow
        'FA_CUSTOM': '#1b5e20',       # dark green
        'DF_PERSIST': '#00838f',      # teal accent
    }


def render_fsp_color_legend(fsp_colors: dict):
    """Render a compact legend mapping FSP names to their colors."""
    items = []
    for name, color in fsp_colors.items():
        items.append(
            f"<div style='display:flex;align-items:center;margin-right:12px;margin-bottom:6px;'>"
            f"<span style='width:14px;height:14px;border-radius:3px;display:inline-block;background:{color};margin-right:6px;'></span>"
            f"<span style='font-size:0.9rem;'>{name}</span>"
            f"</div>"
        )

    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:center;'>" + "".join(items) + "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


def build_fsp_palette(fsp_names):
    """Return a color dict limited to the provided FSP names (no hardcoding extras)."""
    base_palette = get_fsp_color_map()
    fallback = px.colors.qualitative.Bold + px.colors.qualitative.Set3 + px.colors.qualitative.Dark24
    palette = {}
    for idx, fsp in enumerate(fsp_names):
        palette[fsp] = base_palette.get(fsp, fallback[idx % len(fallback)])
    return palette


def show():
    """Main function for predictions and visualization page."""
    st.header(" Predictions & Visualization")
    st.markdown("Generate predictions, export results, and explore interactive visualizations")

    # AI-generated page summary
    render_page_summary("Predictions & Visualization", get_page_context("Predictions & Visualization"))

    # Check if models are available
    if not st.session_state.models_trained:
        st.warning(" Please train or load a model in **Model Training** before continuing")
        return

    # Step 1: Generate Predictions
    st.markdown("### Step 1: Generate Predictions")

    if st.button(" Generate Predictions for All Models", type="primary"):
        with st.spinner("Generating predictions..."):
            generate_all_predictions()
            st.success(" Predictions generated successfully!")

    # If predictions are generated
    if st.session_state.predictions_generated:
        # Step 2: Export Options
        st.markdown("---")
        st.markdown("### Step 2: Export Predictions")

        export_predictions()

        # Step 3: Visualizations
        st.markdown("---")
        st.markdown("### Step 3: Interactive Visualizations")

        # Setup Ollama model selector for insights
        st.markdown("** AI Insights Configuration**")
        available_models = get_available_models()
        if available_models:
            st.success(f" Ollama connected | {len(available_models)} models available")
            with st.expander(" Available Models"):
                for model in available_models:
                    st.caption(f" {model}")
        else:
            st.warning(" Ollama server not accessible - using template insights")

        # Visualization tabs
        # NOTE: DSM Analysis tab removed  commented out below
        viz_tab1, viz_tab3, viz_tab4, viz_tab5, viz_tab6 = st.tabs([
            " Quantile Forecasts",
            # " DSM Analysis",
            " Time Series Comparison",
            " FSP Selection Analysis",
            " Error Analysis",
            " Test Set Aggregates"
        ])

        with viz_tab1:
            visualize_quantile_forecasts()

        # with viz_tab2:
        #     visualize_dsm_analysis()

        with viz_tab3:
            visualize_time_series()

        with viz_tab4:
            visualize_fsp_selection()

        with viz_tab5:
            visualize_error_analysis()

        with viz_tab6:
            visualize_test_set_aggregates()


def generate_all_predictions():
    """Generate predictions for all trained models."""
    models = st.session_state.models_trained
    test_df = st.session_state.test_df
    feature_cols = st.session_state.feature_columns
    imputer = st.session_state.imputer
    scaler = st.session_state.scaler
    model_configs = st.session_state.get('model_configs', {})

    TARGET = 'target_horizon'

    # Prepare test data
    X_test = test_df[feature_cols].copy()
    if imputer is not None:
        X_test_imputed = imputer.transform(X_test)
    else:
        # Handle missing imputer by using median imputation as fallback
        X_test_imputed = X_test.fillna(X_test.median()).to_numpy()

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test_imputed)
    else:
        X_test_scaled = np.asarray(X_test_imputed)

    # Convert back to DataFrame for tree-based models (preserves feature names)
    X_test_df = pd.DataFrame(X_test_imputed, columns=feature_cols, index=test_df.index)

    prediction_dfs = {}
    base_predictions = {}  # Store base model predictions for ensemble use

    # First pass: Generate predictions for all base models (non-ensemble)
    for model_name, model in models.items():
        # Skip ensemble models in first pass
        if isinstance(model, dict):
            continue

        # Get predictions based on model type
        if model_name in ['ridge', 'lasso']:
            # Linear models use scaled data
            predictions = np.maximum(model.predict(X_test_scaled), 0)  # Clip to >= 0
        elif model_name == 'harmonic_regression':
            config = model_configs.get(model_name, {})
            X_test_h = build_harmonic_regression_features(X_test_scaled, test_df, config)
            predictions = np.maximum(model.predict(X_test_h), 0)
        elif model_name in ['ann', 'fcnn']:
            # Feedforward neural networks use scaled 2D data
            predictions = model.predict(X_test_scaled, verbose=0)
            predictions = np.maximum(predictions.flatten(), 0)  # Clip to >= 0
        elif model_name in ['lstm', 'gru']:
            # LSTM/GRU models need 3D input with proper timesteps
            config = model_configs.get(model_name, {})
            timesteps = config.get('timesteps', 3)
            X_test_3d = reshape_for_rnn(X_test_scaled, timesteps)
            predictions = model.predict(X_test_3d, verbose=0)
            predictions = np.maximum(predictions.flatten(), 0)  # Clip to >= 0
        elif model_name == 'temporal_cnn':
            # Temporal CNN with different default timesteps
            config = model_configs.get(model_name, {})
            timesteps = config.get('timesteps', 5)
            X_test_3d = reshape_for_rnn(X_test_scaled, timesteps)
            predictions = model.predict(X_test_3d, verbose=0)
            predictions = np.maximum(predictions.flatten(), 0)  # Clip to >= 0
        elif model_name == 'custom_architecture':
            # Custom architecture with configurable timesteps
            config = model_configs.get(model_name, {})
            timesteps = config.get('timesteps', 1)
            if timesteps > 1:
                X_test_3d = reshape_for_rnn(X_test_scaled, timesteps)
                predictions = model.predict(X_test_3d, verbose=0)
            else:
                predictions = model.predict(X_test_scaled, verbose=0)
            predictions = np.maximum(predictions.flatten(), 0)  # Clip to >= 0
        elif model_name == 'ceemdan_vmd_cnn_bilstm':
            config = model_configs.get(model_name, {})
            timesteps = config.get('timesteps', 6)
            X_test_aug = prepare_ceemdan_vmd_features(X_test_scaled, config)
            X_test_3d = reshape_for_rnn(X_test_aug, timesteps)
            predictions = model.predict(X_test_3d, verbose=0)
            predictions = np.maximum(predictions.flatten(), 0)
        elif model_name == 'ivmd_fe_ad_informer':
            config = model_configs.get(model_name, {})
            timesteps = config.get('timesteps', 6)
            X_test_aug = prepare_ivmd_fe_features(X_test_scaled, config)
            X_test_3d = reshape_for_rnn(X_test_aug, timesteps)
            predictions = model.predict(X_test_3d, verbose=0)
            predictions = np.maximum(predictions.flatten(), 0)
        else:
            # Tree-based and other models use unscaled data with feature names
            predictions = np.maximum(model.predict(X_test_df), 0)  # Clip to >= 0

        # Store base predictions for ensemble models
        base_predictions[model_name] = predictions

        # Select FSPs and create output
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

        prediction_dfs[model_name] = output_df

    # Second pass: Handle ensemble models
    for model_name, model in models.items():
        # Check if model is a dict-based ensemble
        if isinstance(model, dict):
            # Handle dict-based ensemble models
            if model.get('type') == 'stacking':
                # Stacking ensemble: generate fresh predictions from base models
                base_model_names = model.get('base_models', [])

                # If base_models not saved (old trained model), try to infer from meta-model
                if not base_model_names and base_predictions:
                    # Get meta-model expected features
                    meta_model = model.get('meta_model')
                    use_original_features = model.get('config', {}).get('use_original_features', True)

                    if meta_model and hasattr(meta_model, 'n_features_in_'):
                        expected_features = meta_model.n_features_in_
                        n_base_models_needed = expected_features - len(feature_cols) if use_original_features else expected_features

                        # Use the first n base models (in order they were added to dict)
                        base_model_names = list(base_predictions.keys())[:n_base_models_needed]
                        st.info(f"i {model_name}: Inferred {n_base_models_needed} base models. For best results, retrain the stacking ensemble.")

                if base_model_names and base_predictions:
                    # Use only the base models that were used during training, in the same order
                    base_preds_list = []
                    for bm_name in base_model_names:
                        if bm_name in base_predictions:
                            base_preds_list.append(base_predictions[bm_name])

                    if base_preds_list:
                        # Check if all predictions have the same length
                        min_len = min(len(p) for p in base_preds_list)
                        base_preds_list = [p[:min_len] for p in base_preds_list]

                        meta_features = np.column_stack(base_preds_list)

                        # Optionally include original features
                        if model.get('config', {}).get('use_original_features', True):
                            meta_features = np.column_stack([meta_features, X_test_scaled[:min_len]])

                        predictions = np.maximum(model['meta_model'].predict(meta_features), 0)  # Clip to >= 0
                    else:
                        st.warning(f" Required base models not found for {model_name}")
                        continue
                else:
                    st.warning(f" No base models available for {model_name}")
                    continue

            elif model.get('type') == 'averaging':
                # Averaging ensemble: average base predictions
                if base_predictions:
                    base_preds_list = list(base_predictions.values())
                    min_len = min(len(p) for p in base_preds_list)
                    base_preds_array = np.array([p[:min_len] for p in base_preds_list])

                    if model.get('config', {}).get('trim_outliers', False):
                        # Trim top and bottom 10%
                        predictions = np.mean(np.percentile(base_preds_array, [10, 90], axis=0), axis=0)
                    else:
                        predictions = np.mean(base_preds_array, axis=0)

                    # Ensure non-negative predictions
                    predictions = np.maximum(predictions, 0)
                else:
                    st.warning(f" No base models available for {model_name}")
                    continue

            elif model.get('type') == 'weighted_blending':
                # Weighted blending: use weights to combine predictions
                if base_predictions:
                    weights = model.get('weights', [])
                    base_preds_list = list(base_predictions.values())

                    if len(weights) != len(base_preds_list):
                        st.warning(f" Weight mismatch for {model_name}")
                        continue

                    min_len = min(len(p) for p in base_preds_list)
                    base_preds_array = np.array([p[:min_len] for p in base_preds_list])

                    predictions = np.average(base_preds_array, axis=0, weights=weights)
                    # Ensure non-negative predictions
                    predictions = np.maximum(predictions, 0)
                else:
                    st.warning(f" No base models available for {model_name}")
                    continue
            else:
                st.warning(f" Unknown ensemble type for {model_name}")
                continue

            # Select FSPs and create output for ensemble
            # Adjust test_df if predictions are shorter due to RNN models
            test_df_subset = test_df.iloc[:len(predictions)]

            selected_fsps, scheduled_power, confidence = select_best_fsp_by_prediction(
                test_df_subset, predictions
            )

            output_df = create_prediction_dataframe(
                test_df_subset,
                predictions,
                selected_fsps,
                scheduled_power,
                confidence,
                model_name
            )

            prediction_dfs[model_name] = output_df

    st.session_state.prediction_dfs = prediction_dfs
    st.session_state.predictions_generated = True


def _enforce_min_block_constraint(fsps, powers, confs, fsp_forecast_lookup,
                                   predictions, min_blocks=6):
    """
    Enforce minimum consecutive-block constraint for FSP selection.

    Any FSP run shorter than *min_blocks* is merged into the longer
    adjacent run so that switching only happens after at least
    *min_blocks* of the same FSP.

    Parameters
    ----------
    fsps : list[str]           per-block FSP names
    powers : list[float]       per-block scheduled power
    confs : list[float]        per-block confidence scores
    fsp_forecast_lookup : dict  {FSP_NAME: np.array of forecast values}
    predictions : np.array     ML predicted actual power per block
    min_blocks : int           minimum consecutive blocks (default 6)

    Returns
    -------
    (list[str], list[float], list[float])
    """
    n = len(fsps)
    if n <= min_blocks:
        return fsps, powers, confs

    changed = True
    max_iters = 200
    itr = 0

    while changed and itr < max_iters:
        changed = False
        itr += 1

        # Build runs: [(start, end_exclusive, fsp_name), ...]
        runs = []
        i = 0
        while i < n:
            j = i
            while j < n and fsps[j] == fsps[i]:
                j += 1
            runs.append((i, j, fsps[i]))
            i = j

        # Find the shortest run that violates the constraint (skip UNKNOWN)
        shortest_idx = None
        shortest_len = min_blocks
        for r_idx, (start, end, fsp) in enumerate(runs):
            run_len = end - start
            if run_len < shortest_len and fsp != 'UNKNOWN':
                shortest_len = run_len
                shortest_idx = r_idx

        if shortest_idx is None:
            break  # all runs satisfy the constraint

        start, end, _short_fsp = runs[shortest_idx]

        # Determine which neighbour to merge into
        prev_fsp = runs[shortest_idx - 1][2] if shortest_idx > 0 else None
        next_fsp = runs[shortest_idx + 1][2] if shortest_idx < len(runs) - 1 else None
        prev_len = (runs[shortest_idx - 1][1] - runs[shortest_idx - 1][0]) if shortest_idx > 0 else 0
        next_len = (runs[shortest_idx + 1][1] - runs[shortest_idx + 1][0]) if shortest_idx < len(runs) - 1 else 0

        # Prefer same-name neighbours (bridges a gap), then longer neighbour
        if prev_fsp == next_fsp and prev_fsp is not None and prev_fsp != 'UNKNOWN':
            merge_fsp = prev_fsp
        elif prev_len >= next_len and prev_fsp is not None and prev_fsp != 'UNKNOWN':
            merge_fsp = prev_fsp
        elif next_fsp is not None and next_fsp != 'UNKNOWN':
            merge_fsp = next_fsp
        elif prev_fsp is not None and prev_fsp != 'UNKNOWN':
            merge_fsp = prev_fsp
        else:
            break  # cannot merge (all neighbours are UNKNOWN)

        # Replace the short run with merge_fsp
        for idx in range(start, end):
            fsps[idx] = merge_fsp
            # Update scheduled power from the forecast lookup
            if merge_fsp in fsp_forecast_lookup and idx < len(fsp_forecast_lookup[merge_fsp]):
                powers[idx] = float(fsp_forecast_lookup[merge_fsp][idx])
            # Recalculate confidence
            pred = predictions[idx] if idx < len(predictions) else np.nan
            if not np.isnan(pred):
                errors = {}
                for fsp_name, fcast in fsp_forecast_lookup.items():
                    if idx < len(fcast) and not np.isnan(fcast[idx]):
                        errors[fsp_name] = abs(float(fcast[idx]) - pred)
                if errors:
                    min_err = min(errors.values())
                    max_err = max(errors.values())
                    confs[idx] = (max_err - min_err) / (max_err + 1e-8) if max_err > 0 else 0.5

        changed = True

    return fsps, powers, confs


def select_best_fsp_by_prediction(df, predictions):
    """Select FSP whose forecast is closest to predicted actual power.

    Applies a minimum-6-block constraint **per day** so that FSP switches
    only occur after at least 6 consecutive blocks of the same FSP.
    """
    from src.data.preprocessing import get_fsp_forecast_columns

    MIN_BLOCKS = 6  # minimum consecutive blocks before switching

    # Get selected FSPs from session state
    selected_fsps_list = st.session_state.get('selected_fsps', [])

    fsp_cols = get_fsp_forecast_columns(df)

    # Build FSP forecast lookup (needed for min-block smoothing)
    fsp_forecast_lookup = {}
    for fsp_col in fsp_cols:
        fsp_name = fsp_col.replace('forecast_power_', '').upper()
        if selected_fsps_list and fsp_name not in selected_fsps_list:
            continue
        if fsp_col in df.columns:
            fsp_forecast_lookup[fsp_name] = df[fsp_col].values

    # --- Pass 1: per-block best-FSP selection (original logic) ---
    selected_fsps = []
    scheduled_power = []
    confidence = []

    n = len(df)
    # Pad predictions to match df length
    preds_full = np.full(n, np.nan)
    pred_len = min(len(predictions), n)
    preds_full[:pred_len] = predictions[:pred_len]

    for i in range(n):
        pred = preds_full[i]

        # Get FSP values for this row  ONLY from selected FSPs
        fsp_values = {}
        for fsp_col in fsp_cols:
            fsp_name = fsp_col.replace('forecast_power_', '').upper()
            if selected_fsps_list and fsp_name not in selected_fsps_list:
                continue
            val = df.iloc[i].get(fsp_col, np.nan)
            if not np.isnan(val):
                fsp_values[fsp_name] = val

        if fsp_values and not np.isnan(pred):
            errors = {fsp: abs(val - pred) for fsp, val in fsp_values.items()}
            best_fsp = min(errors, key=errors.get)
            selected_fsps.append(best_fsp)
            scheduled_power.append(fsp_values[best_fsp])
            min_err, max_err = min(errors.values()), max(errors.values())
            conf = (max_err - min_err) / (max_err + 1e-8) if max_err > 0 else 0.5
            confidence.append(conf)
        else:
            selected_fsps.append('UNKNOWN')
            scheduled_power.append(np.nan)
            confidence.append(0.0)

    # --- Pass 2: enforce min-block constraint PER DAY ---
    # Group row indices by date so each day is smoothed independently
    has_date = 'date' in df.columns
    if has_date:
        dates = df['date'].values
        day_groups = {}
        for i in range(n):
            d = str(dates[i])
            day_groups.setdefault(d, []).append(i)
    else:
        # No date column  treat entire dataset as one group
        day_groups = {'all': list(range(n))}

    for _date_key, indices in day_groups.items():
        # Extract per-day slices (as plain Python lists for safe mutation)
        day_fsps = [selected_fsps[i] for i in indices]
        day_powers = [scheduled_power[i] for i in indices]
        day_confs = [confidence[i] for i in indices]
        day_preds = np.array([preds_full[i] for i in indices])

        # Per-day forecast lookup
        day_fsp_lookup = {}
        for fsp_name, fcast in fsp_forecast_lookup.items():
            day_fsp_lookup[fsp_name] = np.array(
                [fcast[i] if i < len(fcast) else np.nan for i in indices]
            )

        # Apply the smoothing constraint
        day_fsps, day_powers, day_confs = _enforce_min_block_constraint(
            day_fsps, day_powers, day_confs,
            day_fsp_lookup, day_preds, min_blocks=MIN_BLOCKS
        )

        # Write smoothed values back
        for local_idx, global_idx in enumerate(indices):
            selected_fsps[global_idx] = day_fsps[local_idx]
            scheduled_power[global_idx] = day_powers[local_idx]
            confidence[global_idx] = day_confs[local_idx]

    return (np.array(selected_fsps),
            np.array(scheduled_power, dtype=float),
            np.array(confidence, dtype=float))


def create_prediction_dataframe(df, predictions, selected_fsps_array, scheduled_power, confidence, model_name):
    """Create structured prediction DataFrame - only includes selected FSP columns."""
    output = pd.DataFrame()

    # Time columns
    output['timestamp'] = df.get('timestamp', pd.NaT)
    output['date'] = df.get('date', '')
    output['block'] = df.get('block', 0)

    # Only include SELECTED FSP forecasts
    from src.data.preprocessing import get_fsp_forecast_columns
    fsp_cols = get_fsp_forecast_columns(df)

    # Get selected FSPs from session state
    selected_fsps_list = st.session_state.get('selected_fsps', [])

    for fsp_col in fsp_cols:
        fsp_name = fsp_col.replace('forecast_power_', '').upper()

        # Only include selected FSPs in output
        if selected_fsps_list and fsp_name not in selected_fsps_list:
            continue

        output[f'{fsp_name}_power'] = df.get(fsp_col, np.nan)

    # Actual and manual
    output['actual_power'] = df.get('target_horizon', np.nan)
    output['manual_scheduled_power'] = df.get('schedule_power', np.nan)

    # ML outputs
    output['ml_predicted_power'] = predictions
    output['ml_selected_fsp'] = selected_fsps_array
    output['ml_scheduled_power'] = scheduled_power
    output['selection_confidence'] = confidence
    output['model_name'] = model_name

    # Errors
    output['ml_predicted_error'] = np.abs(output['actual_power'] - output['ml_predicted_power'])
    output['ml_scheduled_error'] = np.abs(output['actual_power'] - output['ml_scheduled_power'])
    output['manual_error'] = np.abs(output['actual_power'] - output['manual_scheduled_power'])

    # Legacy compatibility
    output['ml_error'] = output['ml_scheduled_error']

    # Add quantile columns for uncertainty quantification
    # Calculate residuals for std estimation
    residuals = output['actual_power'] - output['ml_predicted_power']
    residual_std = float(np.nanstd(residuals))
    if np.isnan(residual_std) or residual_std == 0:
        residual_std = 1.0  # Default to 1.0 if no variation

    # Compute quantiles using normal distribution assumption
    output['q25'] = output['ml_predicted_power'] + norm.ppf(0.25) * residual_std
    output['q50'] = output['ml_predicted_power']  # Median = predicted
    output['q75'] = output['ml_predicted_power'] + norm.ppf(0.75) * residual_std
    output['q80'] = output['ml_predicted_power'] + norm.ppf(0.80) * residual_std
    output['q90'] = output['ml_predicted_power'] + norm.ppf(0.90) * residual_std

    # Ensure non-negative quantiles (power cannot be negative)
    for q_col in ['q25', 'q50', 'q75', 'q80', 'q90']:
        output[q_col] = np.maximum(output[q_col], 0)

    return output


def export_predictions():
    """Provide export options for predictions."""
    prediction_dfs = st.session_state.prediction_dfs
    plant_name = st.session_state.get('plant_selected', 'unknown_plant')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models", len(prediction_dfs))

    with col2:
        total_predictions = sum(len(df) for df in prediction_dfs.values())
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col3:
        st.metric("Export Formats", "CSV, ZIP")

    # Individual exports
    st.markdown("#### Individual Model Exports")

    for model_name, pred_df in prediction_dfs.items():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.text(f" {model_name.replace('_', ' ').title()}")

        with col2:
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label=" CSV",
                data=csv,
                file_name=f"predictions_{plant_name}_{model_name}.csv",
                mime="text/csv",
                key=f"download_{model_name}"
            )

        with col3:
            view_clicked = st.button(
                " View",
                key=f"view_{model_name}"
            )

        if view_clicked:
            st.session_state[f"show_preview_{model_name}"] = not st.session_state.get(f"show_preview_{model_name}", False)

        if st.session_state.get(f"show_preview_{model_name}", False):
            with st.expander(f" Preview: {model_name.replace('_', ' ').title()}", expanded=True):
                view_mode = st.radio(
                    "Select rows to display",
                    ["First 50", "Last 50", "Random 50"],
                    horizontal=True,
                    key=f"view_mode_{model_name}"
                )

                if view_mode == "First 50":
                    display_df = pred_df.head(50)
                elif view_mode == "Last 50":
                    display_df = pred_df.tail(50)
                else:
                    sample_size = min(50, len(pred_df))
                    display_df = pred_df.sample(n=sample_size, random_state=42)

                st.caption(f"Showing {len(display_df)} of {len(pred_df)} rows")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Combined export
    st.markdown("#### Combined Export (All Models)")

    if st.button(" Prepare ZIP Download"):
        zip_buffer = create_zip_export(prediction_dfs, plant_name)

        st.download_button(
            label=" Download All Predictions (ZIP)",
            data=zip_buffer,
            file_name=f"all_predictions_{plant_name}.zip",
            mime="application/zip"
        )


def create_zip_export(prediction_dfs, plant_name='unknown_plant'):
    """Create ZIP file containing all predictions."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for model_name, pred_df in prediction_dfs.items():
            csv_data = pred_df.to_csv(index=False)
            zip_file.writestr(f"predictions_{plant_name}_{model_name}.csv", csv_data)

    zip_buffer.seek(0)
    return zip_buffer


def visualize_time_series():
    """Interactive time series comparison visualization."""
    st.markdown("#### Time Series Comparison")
    st.info("Compare actual power, manual schedule, ML predictions, and FSP forecasts")

    prediction_dfs = st.session_state.prediction_dfs

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(prediction_dfs.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )

    pred_df = prediction_dfs[model_name]

    # Plot range selection (date-based when timestamp is available)
    has_timestamp = 'timestamp' in pred_df.columns and not pred_df['timestamp'].isna().all()
    plot_mode = None
    start_idx = None
    sample_size = None

    if has_timestamp:
        ts_series = pd.to_datetime(pred_df['timestamp'], errors='coerce')
        has_valid_ts = ts_series.notna().any()
    else:
        ts_series = None
        has_valid_ts = False

    if has_valid_ts:
        plot_mode = st.radio(
            "Plot Range",
            ["Full Test Set", "Date Range", "Single Day", "Single Month"],
            horizontal=True
        )
    else:
        st.info("Timestamp not available. Using sample-based plotting.")
        plot_mode = "Sample Range"

    if plot_mode == "Sample Range":
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.slider(
                "Start Sample",
                0,
                len(pred_df) - 100,
                0,
                step=100
            )
        with col2:
            sample_size = st.slider(
                "Number of Samples to Display",
                100,
                min(1000, len(pred_df)),
                500,
                step=100
            )

    # Ensure error columns exist on full test set (backward compatibility for old predictions)
    if 'ml_predicted_error' not in pred_df.columns:
        pred_df['ml_predicted_error'] = np.abs(pred_df['actual_power'] - pred_df['ml_predicted_power'])
    if 'ml_scheduled_error' not in pred_df.columns:
        pred_df['ml_scheduled_error'] = np.abs(pred_df['actual_power'] - pred_df['ml_scheduled_power'])
    if 'manual_error' not in pred_df.columns:
        pred_df['manual_error'] = np.abs(pred_df['actual_power'] - pred_df['manual_scheduled_power'])

    # Store full test set for metrics calculation (before filtering)
    full_test_df = pred_df.copy()

    # Filter data for plotting only
    if plot_mode == "Sample Range":
        plot_df = pred_df.iloc[start_idx:start_idx + sample_size].copy()
    elif plot_mode == "Full Test Set":
        plot_df = pred_df.copy()
    else:
        pred_df = pred_df.copy()
        pred_df['_timestamp'] = ts_series
        pred_df = pred_df[pred_df['_timestamp'].notna()]

        if plot_mode == "Date Range":
            min_date = pred_df['_timestamp'].min().date()
            max_date = pred_df['_timestamp'].max().date()
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
            mask = (pred_df['_timestamp'].dt.date >= start_date) & (pred_df['_timestamp'].dt.date <= end_date)
            plot_df = pred_df[mask].copy()
        elif plot_mode == "Single Day":
            min_date = pred_df['_timestamp'].min().date()
            max_date = pred_df['_timestamp'].max().date()
            selected_day = st.date_input(
                "Select Day",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            mask = pred_df['_timestamp'].dt.date == selected_day
            plot_df = pred_df[mask].copy()
        else:  # Single Month
            month_options = (
                pred_df['_timestamp']
                .dt.to_period('M')
                .sort_values()
                .unique()
            )
            month_labels = [str(m) for m in month_options]
            selected_month = st.selectbox("Select Month (YYYY-MM)", month_labels)
            month_period = pd.Period(selected_month, freq='M')
            mask = pred_df['_timestamp'].dt.to_period('M') == month_period
            plot_df = pred_df[mask].copy()

        if plot_df.empty:
            st.warning("No data found for the selected range. Showing full test set instead.")
            plot_df = pred_df.copy()

    # Create x-axis: use timestamp if available, otherwise use block numbers
    if '_timestamp' in plot_df.columns and not plot_df['_timestamp'].isna().all():
        x_axis = plot_df['_timestamp']
        x_label = 'Timestamp'
    elif 'timestamp' in plot_df.columns and not plot_df['timestamp'].isna().all():
        x_axis = plot_df['timestamp']
        x_label = 'Timestamp'
    elif 'block' in plot_df.columns:
        x_axis = plot_df['block']
        x_label = 'Block Number'
    else:
        x_axis = plot_df.index
        x_label = 'Index'

    # Create plot
    fig = go.Figure()

    # Actual power
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=plot_df['actual_power'],
        mode='lines',
        name='Actual Power',
        line=dict(color='black', width=2)
    ))

    # Manual schedule
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=plot_df['manual_scheduled_power'],
        mode='lines',
        name='Manual Schedule',
        line=dict(color='red', width=1, dash='dash')
    ))

    # ML predicted
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=plot_df['ml_predicted_power'],
        mode='lines',
        name='ML Predicted',
        line=dict(color='blue', width=1.5)
    ))

    # ML scheduled
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=plot_df['ml_scheduled_power'],
        mode='lines',
        name='ML Scheduled (Selected FSP)',
        line=dict(color='green', width=1.5)
    ))

    # Add FSP forecasts - only show selected FSPs
    show_fsps = st.checkbox("Show Individual FSP Forecasts (Selected FSPs Only)", value=False)

    if show_fsps:
        from src.data.preprocessing import get_fsp_forecast_columns
        test_df = st.session_state.test_df
        fsp_cols = get_fsp_forecast_columns(test_df)

        # Get selected FSPs to filter
        selected_fsps = st.session_state.get('selected_fsps', [])

        # Build palette only for available FSPs
        available_fsp_names = [col.replace('forecast_power_', '').upper() for col in fsp_cols]
        fsp_colors = build_fsp_palette(available_fsp_names)
        render_fsp_color_legend(fsp_colors)

        for idx, fsp_col in enumerate(fsp_cols):
            fsp_name = fsp_col.replace('forecast_power_', '').upper()

            # Skip FSPs not in selected list
            if selected_fsps and fsp_name not in selected_fsps:
                continue

            col_name = f'{fsp_name}_power'

            if col_name in plot_df.columns:
                # Get color for this FSP
                color = fsp_colors.get(fsp_name, px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)])

                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=plot_df[col_name],
                    mode='lines',
                    name=f'FSP: {fsp_name}',
                    line=dict(color=color, width=1.5),  # Solid line, not dotted
                    visible='legendonly'  # Hidden by default
                ))

    fig.update_layout(
        title=f'Time Series Comparison - {model_name.replace("_", " ").title()}',
        xaxis_title=x_label,
        yaxis_title='Power (MW)',
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    st.plotly_chart(fig, width='stretch')

    # Statistics - Show all MAE metrics for entire test set (not just filtered plot range)
    st.markdown("####  Performance Metrics")
    st.caption(" These metrics reflect the **entire test set**, not just the filtered plot range above.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ml_predicted_mae = full_test_df['ml_predicted_error'].mean()
        st.metric("ML Predicted MAE", f"{ml_predicted_mae:.3f} MW",
                  help="MAE between Actual Power and ML Predicted Power (Entire Test Set)")

    with col2:
        ml_scheduled_mae = full_test_df['ml_scheduled_error'].mean()
        st.metric("ML Scheduled MAE", f"{ml_scheduled_mae:.3f} MW",
                  help="MAE between Actual Power and ML Scheduled Power (selected FSP) (Entire Test Set)")

    with col3:
        manual_mae = full_test_df['manual_error'].mean()
        st.metric("Manual MAE", f"{manual_mae:.3f} MW",
                  help="MAE between Actual Power and Manual Scheduled Power (Entire Test Set)")

    with col4:
        # Calculate improvement: Manual vs ML Scheduled
        if manual_mae > 0:
            improvement = ((manual_mae - ml_scheduled_mae) / manual_mae) * 100
        else:
            improvement = 0
        st.metric("Total Improvement", f"{improvement:.1f}%",
                  help="Improvement of ML Scheduled over Manual Schedule (Entire Test Set)")


def visualize_fsp_selection():
    """Visualize FSP selection patterns - filtered to selected FSPs only."""
    st.markdown("#### FSP Selection Analysis")

    prediction_dfs = st.session_state.prediction_dfs
    selected_fsps_list = st.session_state.get('selected_fsps', [])

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(prediction_dfs.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="fsp_model_select"
    )

    pred_df = prediction_dfs[model_name]

    # Filter to only selected FSPs
    if selected_fsps_list:
        pred_df_filtered = pred_df[pred_df['ml_selected_fsp'].isin(selected_fsps_list)].copy()
        st.info(f" Analyzing {len(selected_fsps_list)} selected FSPs: {', '.join(selected_fsps_list)}")
    else:
        pred_df_filtered = pred_df.copy()
        st.info(" Showing all FSPs (no selection filter applied)")

    # FSP selection frequency
    st.markdown("##### FSP Selection Frequency")

    fsp_counts = pred_df_filtered['ml_selected_fsp'].value_counts()

    # Palette limited to FSPs that actually appear in the data
    fsp_colors = build_fsp_palette(fsp_counts.index.tolist())

    colors_list = [fsp_colors.get(fsp, '#7f7f7f') for fsp in fsp_counts.index]

    fig = px.pie(
        values=fsp_counts.values,
        names=fsp_counts.index,
        title='FSP Selection Distribution (Selected FSPs Only)',
        hole=0.4,
        color_discrete_sequence=colors_list
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, width='stretch')

    # Selection confidence
    st.markdown("##### Selection Confidence Distribution")

    fig = px.histogram(
        pred_df_filtered,
        x='selection_confidence',
        nbins=50,
        title='Distribution of Selection Confidence',
        labels={'selection_confidence': 'Confidence Score'}
    )
    fig.add_vline(
        x=pred_df_filtered['selection_confidence'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {pred_df_filtered['selection_confidence'].mean():.3f}"
    )
    st.plotly_chart(fig, width='stretch')

    # FSP performance comparison
    st.markdown("##### FSP Performance After Selection")

    fsp_performance = []
    for fsp in pred_df_filtered['ml_selected_fsp'].unique():
        if fsp != 'UNKNOWN':
            fsp_data = pred_df_filtered[pred_df_filtered['ml_selected_fsp'] == fsp]
            avg_error = fsp_data['ml_error'].mean()
            count = len(fsp_data)
            fsp_performance.append({
                'FSP': fsp,
                'Selections': count,
                'Avg Error (MW)': avg_error
            })

    perf_df = pd.DataFrame(fsp_performance).sort_values('Avg Error (MW)')

    fig = px.bar(
        perf_df,
        x='Avg Error (MW)',
        y='FSP',
        orientation='h',
        title='Average Error by Selected FSP (Selected FSPs Only)',
        color='Selections',
        text='Avg Error (MW)'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig, width='stretch')


def visualize_error_analysis():
    """Error analysis and distribution."""
    st.markdown("#### Error Analysis")

    prediction_dfs = st.session_state.prediction_dfs

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(prediction_dfs.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="error_model_select"
    )

    pred_df = prediction_dfs[model_name]

    # Error distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ML Error Distribution")
        fig = px.histogram(
            pred_df,
            x='ml_error',
            nbins=50,
            title='ML Prediction Error Distribution'
        )
        fig.add_vline(
            x=pred_df['ml_error'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {pred_df['ml_error'].mean():.3f}"
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("##### Manual Error Distribution")
        fig = px.histogram(
            pred_df,
            x='manual_error',
            nbins=50,
            title='Manual Schedule Error Distribution'
        )
        fig.add_vline(
            x=pred_df['manual_error'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {pred_df['manual_error'].mean():.3f}"
        )
        st.plotly_chart(fig, width='stretch')

    # Error comparison scatter
    st.markdown("##### ML vs Manual Error Comparison")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df['manual_error'],
        y=pred_df['ml_error'],
        mode='markers',
        marker=dict(
            size=5,
            color=pred_df['selection_confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=pred_df['ml_selected_fsp'],
        hovertemplate='Manual Error: %{x:.2f}<br>ML Error: %{y:.2f}<br>FSP: %{text}'
    ))

    # Add diagonal line (equal error)
    max_error = max(pred_df['manual_error'].max(), pred_df['ml_error'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_error],
        y=[0, max_error],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Equal Error Line'
    ))

    fig.update_layout(
        title='ML Error vs Manual Error',
        xaxis_title='Manual Error (MW)',
        yaxis_title='ML Error (MW)',
        height=500
    )

    st.plotly_chart(fig, width='stretch')

    # Statistics
    better_count = (pred_df['ml_error'] < pred_df['manual_error']).sum()
    better_pct = (better_count / len(pred_df)) * 100

    st.info(f" ML performed better than manual in **{better_count:,}** cases ({better_pct:.1f}%)")


def render_fsp_waterfall_chart(plot_df, block_axis):
    """
    Create a dynamic waterfall chart showing FSP selection for all 96 blocks.

    Args:
        plot_df: DataFrame containing prediction data for a single day
        block_axis: Array of block numbers (1-96)
    """
    # Get the FSP color palette
    fsp_color_map = get_fsp_color_map()

    # Get selected FSPs from session state
    selected_fsps_list = st.session_state.get('selected_fsps', [])

    # Extract FSP columns dynamically from the dataframe
    fsp_columns = [col for col in plot_df.columns if col.endswith('_power') and col not in
                   ['actual_power', 'manual_scheduled_power', 'ml_predicted_power', 'ml_scheduled_power']]

    fsp_info = []
    for col in fsp_columns:
        fsp_name = col.replace('_power', '').replace('_POWER', '').upper()
        fsp_info.append((fsp_name, col))

    if not fsp_info:
        st.warning("No FSP forecast columns available for the waterfall chart.")
        return

    if selected_fsps_list:
        filtered = [info for info in fsp_info if info[0] in selected_fsps_list]
        if filtered:
            fsp_info = filtered

    available_fsps = [name for name, _ in fsp_info]

    # Map each FSP to its forecast series for hover display
    fsp_forecasts = {}
    for fsp_name, col in fsp_info:
        if col in plot_df.columns:
            fsp_forecasts[fsp_name] = plot_df[col].to_numpy()
        else:
            fsp_forecasts[fsp_name] = np.zeros(len(plot_df))

    # Get the ML selected FSP for each block
    ml_selected_fsp = plot_df['ml_selected_fsp'].values

    # Create figure
    fig = go.Figure()

    # Build FSP palette for available FSPs
    fsp_palette = {}
    for fsp in available_fsps:
        fsp_palette[fsp] = fsp_color_map.get(fsp, '#888888')

    # Create a base trace for each FSP
    # We'll show stacked bars where each FSP contributes when selected
    fsp_data = {}
    for fsp in available_fsps:
        fsp_data[fsp] = []

    # For each block, determine which FSP was selected
    for idx, block in enumerate(block_axis):
        selected = ml_selected_fsp[idx] if idx < len(ml_selected_fsp) else 'UNKNOWN'

        # Get the scheduled power for this block
        scheduled_power = plot_df['ml_scheduled_power'].values[idx] if idx < len(plot_df) else 0

        # Assign power to the selected FSP for this block
        for fsp in available_fsps:
            if fsp == selected:
                fsp_data[fsp].append(scheduled_power)
            else:
                fsp_data[fsp].append(0)

    # Add traces for each FSP
    for fsp in available_fsps:
        forecast_values = fsp_forecasts.get(fsp, np.zeros(len(block_axis)))
        if len(forecast_values) != len(block_axis):
            aligned = np.zeros(len(block_axis))
            limit = min(len(block_axis), len(forecast_values))
            if limit > 0:
                aligned[:limit] = forecast_values[:limit]
            forecast_values = aligned

        fig.add_trace(go.Bar(
            x=block_axis,
            y=fsp_data[fsp],
            name=fsp,
            marker_color=fsp_palette.get(fsp, '#888888'),
            customdata=forecast_values,
            hovertemplate=f"{fsp}: " + "%{customdata:.2f} MW<extra></extra>",
            legendgroup=fsp,
        ))

    # Add actual power as a line for reference
    fig.add_trace(go.Scatter(
        x=block_axis,
        y=plot_df['actual_power'].values,
        name='Actual Power',
        mode='lines',
        line=dict(color='black', width=2, dash='dot'),
        hovertemplate="Actual: %{y:.2f} MW<extra></extra>",
        legendgroup='actual'
    ))

    # Add ML predicted power as a line
    fig.add_trace(go.Scatter(
        x=block_axis,
        y=plot_df['ml_predicted_power'].values,
        name='ML Predicted',
        mode='lines',
        line=dict(color='#FF6B35', width=2),
        hovertemplate="ML Predicted: %{y:.2f} MW<extra></extra>",
        legendgroup='ml_pred'
    ))

    # Update layout
    fig.update_layout(
        title=f"FSP Selection Waterfall - {plot_df['date_key'].iloc[0]}",
        xaxis_title="Block Number (15-min intervals)",
        yaxis_title="Power (MW)",
        barmode='stack',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=4,  # Show every 4th block
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)'
        ),
        plot_bgcolor='rgba(240,240,240,0.3)',
        showlegend=True
    )

    # st.plotly_chart(fig, use_container_width=True) -> use width="stretch"
    st.plotly_chart(fig, key="dsm_analysis_plot", width="stretch")

    # Add summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        unique_fsps_used = len([fsp for fsp in available_fsps if sum(fsp_data[fsp]) > 0])
        st.metric("FSPs Used", unique_fsps_used)

    with col2:
        total_scheduled = plot_df['ml_scheduled_power'].sum()
        st.metric("Total Scheduled", f"{total_scheduled:,.2f} MW")

    with col3:
        total_actual = plot_df['actual_power'].sum()
        accuracy = (1 - abs(total_scheduled - total_actual) / total_actual) * 100 if total_actual > 0 else 0
        st.metric("Daily Accuracy", f"{accuracy:.2f}%")

    with col4:
        avg_confidence = plot_df['selection_confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # FSP Usage Distribution
    st.markdown("##### FSP Usage Distribution")
    fsp_usage_counts = {}
    for fsp in available_fsps:
        count = sum(1 for val in fsp_data[fsp] if val > 0)
        if count > 0:
            fsp_usage_counts[fsp] = count

    if fsp_usage_counts:
        usage_df = pd.DataFrame([
            {"FSP": fsp, "Blocks Used": count, "Percentage": f"{(count/96)*100:.1f}%"}
            for fsp, count in sorted(fsp_usage_counts.items(), key=lambda x: x[1], reverse=True)
        ])

        st.dataframe(
            usage_df,
            use_container_width=True,
            hide_index=True
        )


def render_daily_forecast_heatmap(plot_df, block_axis, fsp_cols, fsp_colors):
    """Render a 96-block heatmap of forecasted power for every selected FSP."""
    st.markdown("##### Forecast Heatmap (96 Blocks)")
    st.caption("Cell shading encodes forecasted MW; colored squares mark the scheduled FSP for each 15-minute block.")

    if plot_df is None or plot_df.empty:
        st.info("No forecast data available for the selected day.")
        return

    if not fsp_cols:
        st.info("FSP forecast columns unavailable for the selected timeframe.")
        return

    if block_axis is None:
        block_axis = np.arange(1, len(plot_df) + 1)

    fsp_pairs = []
    for col in fsp_cols:
        if col in plot_df.columns:
            name = col.replace('forecast_power_', '').upper()
            fsp_pairs.append((name, col))

    if not fsp_pairs:
        st.info("Unable to locate forecast columns for the selected FSPs.")
        return

    selected_order = [fsp.upper() for fsp in st.session_state.get('selected_fsps', [])]
    ordered_names = []
    for name in selected_order:
        if any(pair[0] == name for pair in fsp_pairs) and name not in ordered_names:
            ordered_names.append(name)
    for name, _ in fsp_pairs:
        if name not in ordered_names:
            ordered_names.append(name)

    name_to_col = {name: col for name, col in fsp_pairs}
    block_axis = np.array(block_axis[:len(plot_df)], dtype=int)
    x_labels = [f"B{int(val):02d}" for val in block_axis]

    heatmap_rows = []
    value_matrix = []
    for name in ordered_names:
        col = name_to_col.get(name)
        if not col:
            continue
        series = plot_df[col].astype(float).to_numpy()[:len(x_labels)]
        value_matrix.append(series)
        heatmap_rows.append(name)

    if not value_matrix:
        st.info("Forecast values are missing for the selected configuration.")
        return

    values = np.array(value_matrix, dtype=float)
    valid_values = values[~np.isnan(values)]
    if valid_values.size:
        zmin, zmax = float(valid_values.min()), float(valid_values.max())
    else:
        zmin, zmax = 0.0, 1.0

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_labels,
        y=heatmap_rows,
        z=values,
        colorscale=[
            [0.0, "#003049"],
            [0.35, "#468faf"],
            [0.65, "#f9c74f"],
            [1.0, "#f9844a"]
        ],
        colorbar=dict(title="Forecast (MW)"),
        hovertemplate="FSP %{y}<br>Block %{x}<br>Forecast %{z:.2f} MW<extra></extra>",
        zmin=zmin,
        zmax=zmax,
        showscale=True,
        xgap=1,
        ygap=1
    ))

    scheduled_series = plot_df.get('ml_selected_fsp')
    highlight_x, highlight_y = [], []
    highlight_vals, highlight_colors = [], []
    scheduled_marker_color = '#d81b60'  # single color to indicate scheduled FSP blocks

    if scheduled_series is not None:
        scheduled_upper = scheduled_series.astype(str).str.upper().tolist()

        for idx, fsp in enumerate(scheduled_upper):
            if idx >= len(x_labels):
                break
            if fsp not in heatmap_rows:
                continue

            highlight_x.append(x_labels[idx])
            highlight_y.append(fsp)

            col = name_to_col.get(fsp)
            if col and col in plot_df.columns:
                cell_value = plot_df.iloc[idx][col]
            else:
                cell_value = np.nan
            highlight_vals.append(cell_value)
            highlight_colors.append(scheduled_marker_color)

        if highlight_x:
            fig.add_trace(
                go.Scatter(
                    x=highlight_x,
                    y=highlight_y,
                    mode='markers',
                    name='Scheduled FSP',
                    marker=dict(
                        symbol='square',
                        size=24,
                        color=highlight_colors,
                        line=dict(color='rgba(255,255,255,0.9)', width=1.2)
                    ),
                    customdata=highlight_vals,
                    hovertemplate=(
                        "Block %{x}<br>"
                        "Scheduled %{y}<br>"
                        "Forecast %{customdata:.2f} MW"
                        "<extra></extra>"
                    ),
                    showlegend=True
                )
            )

    text_x, text_y, text_vals = [], [], []
    for row_name, row_values in zip(heatmap_rows, values):
        for block_label, cell_val in zip(x_labels, row_values):
            text_x.append(block_label)
            text_y.append(row_name)
            text_vals.append("-" if np.isnan(cell_val) else f"{cell_val:.1f}")

    fig.add_trace(go.Scatter(
        x=text_x,
        y=text_y,
        mode='text',
        text=text_vals,
        textfont=dict(color='#ffffff', size=11, family='Montserrat'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        height=max(480, 140 + 28 * len(heatmap_rows)),
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(title='FSP', autorange='reversed', tickfont=dict(size=11), showgrid=True, gridcolor='rgba(255,255,255,0.25)'),
        xaxis=dict(title='Block (15-min)', tickangle=-45, showgrid=True, gridcolor='rgba(255,255,255,0.25)', dtick=1),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )

    if len(block_axis) != 96:
        st.caption(" Fewer than 96 records found for this day; heatmap reflects available blocks only.")

    # st.plotly_chart(fig, use_container_width=True) -> use width="stretch"
    st.plotly_chart(fig, key="dsm_heatmap_plot", width="stretch")


def visualize_quantile_forecasts():
    """Visualize residual-based quantile ribbons with daily drill-down, block axis, FSP overlays, and percentile-only hovers."""
    st.markdown("#### Quantile Forecasts")
    st.info("Residual-based quantile ribbons (F10-F90) with percentiles for each point, daily block drill-down, and optional FSP overlays.")

    prediction_dfs = st.session_state.prediction_dfs
    test_df = st.session_state.test_df

    model_name = st.selectbox(
        "Select Model",
        options=list(prediction_dfs.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="quantile_model_select"
    )

    pred_df = prediction_dfs[model_name].copy()

    time_col = None
    for cand in ["timestamp", "date"]:
        if cand in pred_df.columns:
            time_col = cand
            break

    if time_col:
        pred_df[time_col] = pd.to_datetime(pred_df[time_col])
        pred_df = pred_df.sort_values(time_col)
    else:
        pred_df = pred_df.reset_index(drop=True)

    residuals = pred_df['actual_power'] - pred_df['ml_predicted_power']
    residual_std = float(np.nanstd(residuals))
    if np.isnan(residual_std) or residual_std == 0:
        residual_std = 1e-3  # Avoid zero-width bands

    quantile_levels = {
        "F10": 0.10,
        "F25": 0.25,
        "F50": 0.50,
        "F75": 0.75,
        "F90": 0.90
    }
    z_scores = {label: norm.ppf(q) for label, q in quantile_levels.items()}

    for label, z_val in z_scores.items():
        pred_df[label] = pred_df['ml_predicted_power'] + z_val * residual_std

    pred_df['actual_percentile'] = np.clip(
        norm.cdf((pred_df['actual_power'] - pred_df['ml_predicted_power']) / residual_std) * 100,
        0,
        100
    )
    pred_df['ml_sched_percentile'] = np.clip(
        norm.cdf((pred_df['ml_scheduled_power'] - pred_df['ml_predicted_power']) / residual_std) * 100,
        0,
        100
    )

    if time_col and time_col in test_df.columns:
        from src.data.preprocessing import get_fsp_forecast_columns
        fsp_cols = get_fsp_forecast_columns(test_df)
        fsp_merge = test_df[[time_col] + fsp_cols].copy()
        fsp_merge[time_col] = pd.to_datetime(fsp_merge[time_col])
        pred_df = pred_df.merge(fsp_merge, on=time_col, how='left')
    elif not time_col:
        from src.data.preprocessing import get_fsp_forecast_columns
        fsp_cols = get_fsp_forecast_columns(test_df)
        test_df = test_df.reset_index(drop=True)
        pred_df['row_idx'] = np.arange(len(pred_df))
        test_df = test_df.reset_index(drop=True)
        test_df['row_idx'] = np.arange(len(test_df))
        pred_df = pred_df.merge(test_df[['row_idx'] + fsp_cols], on='row_idx', how='left')
    else:
        fsp_cols = []

    if time_col:
        pred_df['date_key'] = pred_df[time_col].dt.date
    else:
        pred_df['date_key'] = pd.Series(["All"] * len(pred_df))

    has_time = bool(time_col)
    if has_time:
        plot_mode = st.radio(
            "Plot Range",
            ["Single Day", "Date Range", "Single Month"],
            horizontal=True,
            key="quantile_plot_range"
        )
    else:
        st.info("Timestamp not available. Showing all data.")
        plot_mode = "Single Day"

    plot_df = None
    block_axis = None

    if plot_mode == "Single Day":
        available_dates = sorted(pred_df['date_key'].dropna().unique())
        # Use calendar-style date_input for single date selection
        selected_date = st.date_input(
            " Select Date for Daily Ribbon",
            value=available_dates[0] if available_dates else None,
            min_value=min(available_dates) if available_dates else None,
            max_value=max(available_dates) if available_dates else None,
            key="quantile_single_date_calendar"
        )
        plot_df = pred_df[pred_df['date_key'] == selected_date].copy()
        block_axis = np.arange(1, len(plot_df) + 1)
    elif plot_mode == "Date Range":
        min_date = pred_df['date_key'].min()
        max_date = pred_df['date_key'].max()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="quantile_date_range"
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
        mask = (pred_df['date_key'] >= start_date) & (pred_df['date_key'] <= end_date)
        plot_df = pred_df[mask].copy()
    elif plot_mode == "Single Day":
        available_dates = sorted(pred_df['date_key'].dropna().unique())
        selected_date = st.selectbox(
            "Select Date for Daily Ribbon",
            options=available_dates,
            format_func=lambda d: d if isinstance(d, str) else d.isoformat(),
            key="quantile_date_select"
        )
        plot_df = pred_df[pred_df['date_key'] == selected_date].copy()
        block_axis = np.arange(1, len(plot_df) + 1)
    else:  # Single Month
        month_options = (
            pred_df[time_col]
            .dt.to_period('M')
            .sort_values()
            .unique()
        )
        month_labels = [str(m) for m in month_options]
        selected_month = st.selectbox("Select Month (YYYY-MM)", month_labels, key="quantile_month_select")
        month_period = pd.Period(selected_month, freq='M')
        mask = pred_df[time_col].dt.to_period('M') == month_period
        plot_df = pred_df[mask].copy()

    if plot_df is None or plot_df.empty:
        st.warning("No data available for the selected range. Showing full test set instead.")
        plot_df = pred_df.copy()

    show_all_fsps = st.checkbox("Show all FSP forecast lines", value=False, key="show_all_fsps_quantile")

    # Palette limited to FSPs available in the merged test data
    available_fsp_names = [col.replace('forecast_power_', '').upper() for col in fsp_cols]
    fsp_colors = build_fsp_palette(available_fsp_names)
    render_fsp_color_legend(fsp_colors)

    def compute_fsp_percentiles(df: pd.DataFrame):
        if not fsp_cols:
            return {}
        pct_cols = {}
        for col in fsp_cols:
            fsp_name = col.replace('forecast_power_', '').upper()
            pct_series = np.clip(
                norm.cdf((df[col] - df['ml_predicted_power']) / residual_std) * 100,
                0,
                100
            )
            pct_cols[fsp_name] = pct_series
        return pct_cols

    fsp_percentiles = compute_fsp_percentiles(pred_df)

    def build_quantile_figure(df: pd.DataFrame, title: str, x_override=None, include_fsps=False):
        x_axis = x_override if x_override is not None else (df[time_col] if time_col else df.index)
        x_label = 'Block (15-min)' if x_override is not None else ('Timestamp' if time_col else 'Index')
        customdata_actual = np.column_stack([
            df['actual_percentile']
        ])
        customdata_sched = np.column_stack([
            df['ml_sched_percentile'],
            df['ml_selected_fsp']
        ])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['F90'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['F10'],
            mode='lines',
            name='80% CI (F10-F90)',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(33, 158, 188, 0.2)'
        ))

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['F75'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['F25'],
            mode='lines',
            name='50% CI (F25-F75)',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(33, 158, 188, 0.35)'
        ))

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['F50'],
            mode='lines',
            name='F50 (Predicted)',
            line=dict(color='#1f77b4', width=2, dash='dash')
        ))

        if include_fsps and fsp_cols:
            for col in fsp_cols:
                fsp_name = col.replace('forecast_power_', '').upper()
                pct_series = fsp_percentiles.get(fsp_name)
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=df[col],
                    mode='lines',
                    name=f"FSP {fsp_name}",
                    line=dict(color=fsp_colors.get(fsp_name, '#7f7f7f'), width=1),
                    customdata=np.column_stack([pct_series]) if pct_series is not None else None,
                    hovertemplate=(
                        "FSP percentile vs band: %{customdata[0]:.1f}th<extra></extra>"
                        if pct_series is not None else "FSP percentile unavailable<extra></extra>"
                    ),
                    opacity=0.6
                ))

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['actual_power'],
            mode='lines',
            name='Actual',
            line=dict(color='#111111', width=2),
            customdata=customdata_actual,
            hovertemplate=(
                "Point percentile vs band: %{customdata[0]:.1f}th<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df['ml_scheduled_power'],
            mode='markers',
            name='ML Scheduled (FSP)',
            marker=dict(
                size=7,
                color=[fsp_colors.get(fsp, '#7f7f7f') for fsp in df['ml_selected_fsp']],
                line=dict(width=1, color='DarkSlateGray')
            ),
            customdata=customdata_sched,
            hovertemplate=(
                "ML scheduled percentile: %{customdata[0]:.1f}th<br>FSP: %{customdata[1]}<extra></extra>"
            )
        ))

        # Add manually scheduled power if available
        if 'manual_scheduled_power' in df.columns and df['manual_scheduled_power'].notna().any():
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=df['manual_scheduled_power'],
                mode='lines',
                name='Manual Scheduled',
                line=dict(
                    color='#006400',  # Dark green
                    width=2
                ),
                hovertemplate="Manual Scheduled: %{y:.2f} MW<extra></extra>"
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title='Power (MW)',
            height=550,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig

    if plot_mode == "Single Day":
        st.markdown("##### Daily Quantile Ribbon (Blocks 1-96)")
        title = f"Daily Quantile Bands ({plot_df['date_key'].iloc[0]})"
        st.plotly_chart(
            build_quantile_figure(plot_df, title, x_override=block_axis, include_fsps=show_all_fsps),
            use_container_width=True
        )
        render_daily_forecast_heatmap(plot_df, block_axis, fsp_cols, fsp_colors)

        # Add FSP Selection Waterfall Chart below quantile plot
        st.markdown("---")
        st.markdown("##### FSP Selection Waterfall (All 96 Blocks)")
        st.info("Dynamic visualization showing which FSP was selected by the ML model for each 15-minute block")
        render_fsp_waterfall_chart(plot_df, block_axis)

        # Generate and display AI-powered insights
        with st.spinner(" Generating AI insights..."):
            try:
                # Initialize Ollama generator
                generator = OllamaInsightsGenerator()

                # Analyze the single day forecast
                analysis = analyze_single_day_forecast(
                    prediction_df=plot_df,
                    test_df=test_df,
                    date=plot_df['date_key'].iloc[0],
                    residual_std=residual_std,
                    model_name=model_name,
                    generator=generator
                )

                # Render insights below the plot
                render_single_day_insights(analysis)

            except Exception as e:
                st.warning(f"Could not generate insights: {str(e)}")
                st.info("Insights generation requires an Ollama server configured with OLLAMA_BASE_URL.")
    elif plot_mode == "Date Range":
        st.markdown("##### Date-Range Quantile Ribbon")
        st.plotly_chart(
            build_quantile_figure(plot_df, "Date-Range Quantile Bands", include_fsps=False),
            use_container_width=True
        )
    elif plot_mode == "Single Month":
        st.markdown("##### Monthly Quantile Ribbon")
        st.plotly_chart(
            build_quantile_figure(plot_df, "Monthly Quantile Bands", include_fsps=False),
            use_container_width=True
        )


def visualize_test_set_aggregates():
    """
    Display comprehensive test set aggregates table showing total power values
    and percentages for actual, ML-predicted, ML-scheduled, and manual-scheduled power.
    """
    st.markdown("#### Test Set Aggregates Summary")
    st.info("Comprehensive comparison of total power values across actual, ML predictions, ML scheduling, and manual scheduling")

    prediction_dfs = st.session_state.prediction_dfs

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(prediction_dfs.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="aggregates_model_select"
    )

    pred_df = prediction_dfs[model_name]

    # Calculate aggregates
    total_actual = pred_df['actual_power'].sum()
    total_ml_predicted = pred_df['ml_predicted_power'].sum()
    total_ml_scheduled = pred_df['ml_scheduled_power'].sum()
    total_manual_scheduled = pred_df['manual_scheduled_power'].sum()

    # Calculate percentages (relative to actual power as baseline)
    pct_ml_predicted = (total_ml_predicted / total_actual * 100) if total_actual > 0 else 0
    pct_ml_scheduled = (total_ml_scheduled / total_actual * 100) if total_actual > 0 else 0
    pct_manual_scheduled = (total_manual_scheduled / total_actual * 100) if total_actual > 0 else 0

    # Calculate differences from actual
    diff_ml_predicted = total_ml_predicted - total_actual
    diff_ml_scheduled = total_ml_scheduled - total_actual
    diff_manual_scheduled = total_manual_scheduled - total_actual

    # Create aggregates table
    aggregates_data = {
        'Metric': [
            'Total Actual Power',
            'Total ML-Predicted Power',
            'Total ML-Scheduled Power',
            'Total Manual-Scheduled Power'
        ],
        'Value (MW)': [
            f"{total_actual:,.2f}",
            f"{total_ml_predicted:,.2f}",
            f"{total_ml_scheduled:,.2f}",
            f"{total_manual_scheduled:,.2f}"
        ],
        'Percentage (%)': [
            "100.00",
            f"{pct_ml_predicted:.2f}",
            f"{pct_ml_scheduled:.2f}",
            f"{pct_manual_scheduled:.2f}"
        ],
        'Difference from Actual (MW)': [
            "-",
            f"{diff_ml_predicted:+,.2f}",
            f"{diff_ml_scheduled:+,.2f}",
            f"{diff_manual_scheduled:+,.2f}"
        ]
    }

    aggregates_df = pd.DataFrame(aggregates_data)

    # Display the table with nice formatting
    st.markdown("##### Aggregate Power Summary")
    st.dataframe(
        aggregates_df,
        use_container_width=True,
        hide_index=True
    )

    # Additional metrics in columns
    st.markdown("---")
    st.markdown("##### Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mae_ml = pred_df['ml_scheduled_error'].mean()
        st.metric(
            "ML MAE",
            f"{mae_ml:.3f} MW",
            help="Mean Absolute Error for ML-scheduled power"
        )

    with col2:
        mae_manual = pred_df['manual_error'].mean()
        st.metric(
            "Manual MAE",
            f"{mae_manual:.3f} MW",
            help="Mean Absolute Error for manual-scheduled power"
        )

    with col3:
        improvement = ((mae_manual - mae_ml) / mae_manual * 100) if mae_manual > 0 else 0
        st.metric(
            "ML Improvement",
            f"{improvement:.1f}%",
            delta=f"{improvement:.1f}%",
            delta_color="normal",
            help="Percentage improvement of ML over manual scheduling"
        )

    with col4:
        better_count = (pred_df['ml_scheduled_error'] < pred_df['manual_error']).sum()
        better_pct = (better_count / len(pred_df) * 100) if len(pred_df) > 0 else 0
        st.metric(
            "ML Better Cases",
            f"{better_pct:.1f}%",
            help="Percentage of cases where ML performed better than manual"
        )

    # Visualize the comparison
    st.markdown("---")
    st.markdown("##### Power Comparison Chart")

    # Create bar chart comparing totals
    fig = go.Figure()

    categories = ['Actual', 'ML Predicted', 'ML Scheduled', 'Manual Scheduled']
    values = [total_actual, total_ml_predicted, total_ml_scheduled, total_manual_scheduled]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f'{v:,.0f} MW' for v in values],
        textposition='outside',
        marker_color=colors,
        hovertemplate='%{x}<br>%{y:,.2f} MW<extra></extra>'
    ))

    fig.update_layout(
        title='Total Power Comparison (MW)',
        yaxis_title='Total Power (MW)',
        height=500,
        showlegend=False
    )

    # st.plotly_chart(fig, use_container_width=True) -> use width="stretch"
    st.plotly_chart(fig, key="fsp_selection_pie_plot", width="stretch")

    # Error distribution comparison
    st.markdown("---")
    st.markdown("##### Error Distribution Comparison")

    fig2 = go.Figure()

    fig2.add_trace(go.Box(
        y=pred_df['ml_scheduled_error'],
        name='ML Scheduled Error',
        marker_color='#2ca02c'
    ))

    fig2.add_trace(go.Box(
        y=pred_df['manual_error'],
        name='Manual Error',
        marker_color='#d62728'
    ))

    fig2.update_layout(
        title='Error Distribution: ML vs Manual Scheduling',
        yaxis_title='Absolute Error (MW)',
        height=400
    )

    st.plotly_chart(fig2, key="fsp_selection_comparison_plot", width="stretch")

    # Download aggregates as CSV
    st.markdown("---")
    st.markdown("##### Export Aggregates")

    plant_name = st.session_state.get('plant_selected', 'unknown_plant')
    csv_data = aggregates_df.to_csv(index=False)
    st.download_button(
        label=" Download Aggregates Table (CSV)",
        data=csv_data,
        file_name=f"test_set_aggregates_{plant_name}_{model_name}.csv",
        mime="text/csv"
    )


# def visualize_dsm_analysis():
# """
# DSM (Deviation Settlement Mechanism) Analysis and Visualization.
#
# Implements comprehensive DSM calculations based on grid frequency bands,
# deviation patterns, and financial settlement mechanisms.
# """
# st.markdown("#### DSM Analysis - Deviation Settlement Mechanism")
# st.info("""
# **DSM Analysis:** Financial settlement for deviations between scheduled and actual generation
# based on grid frequency conditions. This analysis follows CERC DSM regulations for General Seller category.
# """)
#
# prediction_dfs = st.session_state.prediction_dfs
# test_df = st.session_state.test_df
#
# if not prediction_dfs:
# st.warning("No predictions available. Please generate predictions first.")
# return
#
# # Model selection
# col1, col2 = st.columns([2, 1])
#
# with col1:
# model_name = st.selectbox(
# "Select Model for DSM Analysis",
# options=list(prediction_dfs.keys()),
# format_func=lambda x: x.replace('_', ' ').title(),
# key="dsm_model_select"
# )
#
# with col2:
# # DSM Configuration
# st.markdown("**DSM Parameters:**")
# reference_rate = st.number_input(
# "Reference Rate (Rs/Unit)",
# min_value=0.0,
# value=4.7555,
# step=0.01,
# format="%.4f",
# key="dsm_reference_rate",
# help="Base reference rate from GRID India (update regularly)"
# )
#
# ppa_tariff = st.number_input(
# "PPA Tariff (Rs/MWh)",
# min_value=0.0,
# value=3500.0,
# step=10.0,
# key="dsm_ppa_tariff",
# help="Power Purchase Agreement tariff"
# )
#
# # Get prediction data
# pred_df = prediction_dfs[model_name].copy()
#
# # Check for required columns
# required_cols = ['ml_predicted_power', 'actual_power', 'ml_scheduled_power']
# missing_cols = [col for col in required_cols if col not in pred_df.columns]
#
# if missing_cols:
# st.error(f"Missing required columns: {', '.join(missing_cols)}")
# return
#
# # Check for frequency column (if available in test_df)
# has_frequency = False
# freq_col = None
#
# for possible_freq_col in ['frequency', 'grid_frequency', 'freq']:
# if possible_freq_col in test_df.columns:
# freq_col = possible_freq_col
# has_frequency = True
# break
#
# if not has_frequency:
# st.warning("""
# **Grid frequency data not found in dataset.**
# Using simulated frequency based on time patterns for demonstration.
# For production use, ensure grid frequency column is available in the data.
# """)
# # Generate simulated frequency for demonstration
# np.random.seed(42)
# pred_df['frequency'] = 50.0 + np.random.normal(0, 0.03, len(pred_df))
# pred_df['frequency'] = pred_df['frequency'].clip(49.85, 50.10)
# freq_col = 'frequency'
# else:
# # Merge frequency from test_df
# time_col = None
# for cand in ["timestamp", "date", "datetime"]:
# if cand in pred_df.columns and cand in test_df.columns:
# time_col = cand
# break
#
# if time_col:
# pred_df[time_col] = pd.to_datetime(pred_df[time_col])
# test_freq = test_df[[time_col, freq_col]].copy()
# test_freq[time_col] = pd.to_datetime(test_freq[time_col])
# pred_df = pred_df.merge(test_freq, on=time_col, how='left')
# pred_df.rename(columns={freq_col: 'frequency'}, inplace=True)
# else:
# # Fallback: use index-based merge
# test_freq = test_df[[freq_col]].copy().reset_index(drop=True)
# pred_df_reset = pred_df.reset_index(drop=True)
# pred_df = pd.concat([pred_df_reset, test_freq], axis=1)
# pred_df.rename(columns={freq_col: 'frequency'}, inplace=True)
#
# # Perform DSM calculations
# with st.spinner("Calculating DSM metrics..."):
# dsm_df = calculate_dsm_metrics(pred_df, reference_rate, ppa_tariff)
#
# # Display time range selection (similar to quantile forecasts)
# time_col = None
# for cand in ["timestamp", "date", "datetime"]:
# if cand in dsm_df.columns:
# time_col = cand
# break
#
# if time_col:
# dsm_df[time_col] = pd.to_datetime(dsm_df[time_col])
# dsm_df['date_key'] = dsm_df[time_col].dt.date
#
# st.markdown("---")
# st.markdown("### Visualization Controls")
#
# plot_mode = st.radio(
# "Select Plot Range",
# ["Single Day", "Date Range", "Single Month"],
# horizontal=True,
# key="dsm_plot_range"
# )
#
# plot_df = None
# block_axis = None
#
# if plot_mode == "Single Day":
# available_dates = sorted(dsm_df['date_key'].dropna().unique())
# selected_date = st.date_input(
# " Select Date for DSM Analysis",
# value=available_dates[0] if available_dates else None,
# min_value=min(available_dates) if available_dates else None,
# max_value=max(available_dates) if available_dates else None,
# key="dsm_single_date"
# )
# plot_df = dsm_df[dsm_df['date_key'] == selected_date].copy()
# block_axis = np.arange(1, len(plot_df) + 1)
#
# elif plot_mode == "Date Range":
# min_date = dsm_df['date_key'].min()
# max_date = dsm_df['date_key'].max()
# date_range = st.date_input(
# "Select Date Range",
# value=(min_date, max_date),
# min_value=min_date,
# max_value=max_date,
# key="dsm_date_range"
# )
# if isinstance(date_range, tuple) and len(date_range) == 2:
# start_date, end_date = date_range
# else:
# start_date, end_date = min_date, max_date
# mask = (dsm_df['date_key'] >= start_date) & (dsm_df['date_key'] <= end_date)
# plot_df = dsm_df[mask].copy()
#
# else: # Single Month
# month_options = (
# dsm_df[time_col]
# .dt.to_period('M')
# .sort_values()
# .unique()
# )
# month_labels = [str(m) for m in month_options]
# selected_month = st.selectbox(
# "Select Month (YYYY-MM)",
# month_labels,
# key="dsm_month_select"
# )
# month_period = pd.Period(selected_month, freq='M')
# mask = dsm_df[time_col].dt.to_period('M') == month_period
# plot_df = dsm_df[mask].copy()
# else:
# st.info("Timestamp not available. Showing all data.")
# plot_df = dsm_df.copy()
# block_axis = np.arange(1, len(plot_df) + 1)
#
# if plot_df is None or plot_df.empty:
# st.warning("No data available for the selected range.")
# return
#
# # Display DSM Plots
# st.markdown("---")
# st.markdown("### DSM Visualizations")
#
# # Plot 1: Deviation and Frequency Bands
# plot_deviation_frequency(plot_df, block_axis, time_col)
#
# # Plot 2: Band-wise Energy Distribution
# plot_band_distribution(plot_df, block_axis, time_col)
#
# # Plot 3: Financial Impact
# plot_financial_impact(plot_df, block_axis, time_col)
#
# # DSM Calculation Tables for Full Test Set
# st.markdown("---")
# st.markdown("### DSM Calculation Tables (Full Test Set)")
#
# display_dsm_tables(dsm_df, reference_rate, ppa_tariff, model_name)
#
#
# def calculate_dsm_metrics(df: pd.DataFrame, reference_rate: float, ppa_tariff: float) -> pd.DataFrame:
# """
# Calculate comprehensive DSM metrics based on CERC regulations.
#
# Parameters:
# -----------
# df : DataFrame with actual_power, ml_scheduled_power, and frequency columns
# reference_rate : Base reference rate (Rs/Unit)
# ppa_tariff : PPA tariff (Rs/MWh)
#
# Returns:
# --------
# DataFrame with DSM calculations
# """
# dsm_df = df.copy()
#
# # Basic calculations
# dsm_df['deviation_mw'] = dsm_df['actual_power'] - dsm_df['ml_scheduled_power']
# dsm_df['deviation_kwh'] = dsm_df['deviation_mw'] * 0.25 # 15-min block = 0.25 hours
#
# # Error percentage
# dsm_df['error_pct'] = np.where(
# dsm_df['ml_scheduled_power'] != 0,
# (dsm_df['deviation_mw'] / dsm_df['ml_scheduled_power']) * 100,
# 0
# )
# dsm_df['abs_error_pct'] = np.abs(dsm_df['error_pct'])
#
# # Classify injection type
# dsm_df['injection_type'] = np.where(
# dsm_df['deviation_mw'] >= 0,
# 'Over Injection',
# 'Under Injection'
# )
#
# # Classify frequency bands
# dsm_df['freq_band'] = pd.cut(
# dsm_df['frequency'],
# bins=[0, 49.90, 49.95, 50.03, 50.05, 100],
# labels=['Band 1 (49.90)', 'Band 2 (49.90-49.95)', 'Band 3 (49.95-50.03)',
# 'Band 4 (50.03-50.05)', 'Band 5 (50.05)'],
# include_lowest=True
# )
#
# # Initialize DSM units and charges
# dsm_df['dsm_units_kwh'] = 0.0
# dsm_df['dsm_rate'] = 0.0
# dsm_df['dsm_charge'] = 0.0
# dsm_df['dsm_type'] = ''
#
# # Calculate DSM for each frequency band and injection type
# # Band 1: Frequency 49.90 Hz
# mask_b1_over = (dsm_df['frequency'] <= 49.90) & (dsm_df['deviation_mw'] >= 0)
# dsm_df.loc[mask_b1_over, 'dsm_units_kwh'] = dsm_df.loc[mask_b1_over, 'deviation_kwh']
# dsm_df.loc[mask_b1_over, 'dsm_rate'] = reference_rate * 1.50 # 150% of reference
# dsm_df.loc[mask_b1_over, 'dsm_type'] = 'Receivable'
#
# mask_b1_under = (dsm_df['frequency'] <= 49.90) & (dsm_df['deviation_mw'] < 0)
# dsm_df.loc[mask_b1_under, 'dsm_units_kwh'] = dsm_df.loc[mask_b1_under, 'deviation_kwh'].abs()
# dsm_df.loc[mask_b1_under, 'dsm_rate'] = reference_rate * 2.00 # 200% of reference (penalty)
# dsm_df.loc[mask_b1_under, 'dsm_type'] = 'Payable'
#
# # Band 2: 49.90 < Frequency < 49.95 Hz
# mask_b2_over = (dsm_df['frequency'] > 49.90) & (dsm_df['frequency'] < 49.95) & (dsm_df['deviation_mw'] >= 0)
# dsm_df.loc[mask_b2_over, 'dsm_units_kwh'] = dsm_df.loc[mask_b2_over, 'deviation_kwh']
# dsm_df.loc[mask_b2_over, 'dsm_rate'] = reference_rate * 1.20 # 120% of reference
# dsm_df.loc[mask_b2_over, 'dsm_type'] = 'Receivable'
#
# mask_b2_under = (dsm_df['frequency'] > 49.90) & (dsm_df['frequency'] < 49.95) & (dsm_df['deviation_mw'] < 0)
# dsm_df.loc[mask_b2_under, 'dsm_units_kwh'] = dsm_df.loc[mask_b2_under, 'deviation_kwh'].abs()
# dsm_df.loc[mask_b2_under, 'dsm_rate'] = reference_rate * 1.50 # 150% of reference
# dsm_df.loc[mask_b2_under, 'dsm_type'] = 'Payable'
#
# # Band 3: 49.95 Frequency 50.03 Hz (Normal Range)
# mask_b3_over = (dsm_df['frequency'] >= 49.95) & (dsm_df['frequency'] <= 50.03) & (dsm_df['deviation_mw'] >= 0)
# # Cap over-injection at 10% in normal range
# dsm_df.loc[mask_b3_over, 'dsm_units_kwh'] = np.where(
# dsm_df.loc[mask_b3_over, 'abs_error_pct'] > 10,
# dsm_df.loc[mask_b3_over, 'ml_scheduled_power'] * 0.10 * 0.25,
# dsm_df.loc[mask_b3_over, 'deviation_kwh']
# )
# dsm_df.loc[mask_b3_over, 'dsm_rate'] = reference_rate * 1.00 # 100% of reference
# dsm_df.loc[mask_b3_over, 'dsm_type'] = 'Receivable'
#
# mask_b3_under = (dsm_df['frequency'] >= 49.95) & (dsm_df['frequency'] <= 50.03) & (dsm_df['deviation_mw'] < 0)
# # Progressive penalty for under-injection
# dsm_df.loc[mask_b3_under, 'dsm_units_kwh'] = np.where(
# dsm_df.loc[mask_b3_under, 'abs_error_pct'] > 10,
# dsm_df.loc[mask_b3_under, 'ml_scheduled_power'] * 0.10 * 0.25,
# dsm_df.loc[mask_b3_under, 'deviation_kwh'].abs()
# )
# dsm_df.loc[mask_b3_under, 'dsm_rate'] = np.where(
# dsm_df.loc[mask_b3_under, 'abs_error_pct'] > 15,
# reference_rate * 1.50, # 150% for > 15% error
# np.where(
# dsm_df.loc[mask_b3_under, 'abs_error_pct'] > 10,
# reference_rate * 1.20, # 120% for 10-15% error
# reference_rate * 1.00 # 100% for < 10% error
# )
# )
# dsm_df.loc[mask_b3_under, 'dsm_type'] = 'Payable'
#
# # Band 4: 50.03 < Frequency < 50.05 Hz
# mask_b4_over = (dsm_df['frequency'] > 50.03) & (dsm_df['frequency'] < 50.05) & (dsm_df['deviation_mw'] >= 0)
# dsm_df.loc[mask_b4_over, 'dsm_units_kwh'] = dsm_df.loc[mask_b4_over, 'deviation_kwh']
# dsm_df.loc[mask_b4_over, 'dsm_rate'] = reference_rate * 0.50 # 50% of reference
# dsm_df.loc[mask_b4_over, 'dsm_type'] = 'Receivable'
#
# mask_b4_under = (dsm_df['frequency'] > 50.03) & (dsm_df['frequency'] < 50.05) & (dsm_df['deviation_mw'] < 0)
# dsm_df.loc[mask_b4_under, 'dsm_units_kwh'] = dsm_df.loc[mask_b4_under, 'deviation_kwh'].abs()
# dsm_df.loc[mask_b4_under, 'dsm_rate'] = reference_rate * 0.75 # 75% of reference
# dsm_df.loc[mask_b4_under, 'dsm_type'] = 'Payable'
#
# # Band 5: Frequency 50.05 Hz
# mask_b5_over = (dsm_df['frequency'] >= 50.05) & (dsm_df['deviation_mw'] >= 0)
# dsm_df.loc[mask_b5_over, 'dsm_units_kwh'] = dsm_df.loc[mask_b5_over, 'deviation_kwh']
# dsm_df.loc[mask_b5_over, 'dsm_rate'] = 0.0 # No payment for over-injection at high frequency
# dsm_df.loc[mask_b5_over, 'dsm_type'] = 'Receivable'
#
# mask_b5_under = (dsm_df['frequency'] >= 50.05) & (dsm_df['deviation_mw'] < 0)
# dsm_df.loc[mask_b5_under, 'dsm_units_kwh'] = dsm_df.loc[mask_b5_under, 'deviation_kwh'].abs()
# dsm_df.loc[mask_b5_under, 'dsm_rate'] = reference_rate * 0.50 # 50% of reference
# dsm_df.loc[mask_b5_under, 'dsm_type'] = 'Payable'
#
# # Calculate DSM charges (Rs)
# # Convert units from kWh to MWh and apply rate
# dsm_df['dsm_charge'] = (dsm_df['dsm_units_kwh'] / 1000) * dsm_df['dsm_rate']
#
# # Sign convention: Receivable is positive, Payable is negative
# dsm_df.loc[dsm_df['dsm_type'] == 'Payable', 'dsm_charge'] = -dsm_df.loc[dsm_df['dsm_type'] == 'Payable', 'dsm_charge']
#
# # Revenue calculations
# dsm_df['gross_revenue_actual'] = (dsm_df['actual_power'] * 0.25) * (ppa_tariff / 1000) # MWh * Rs/MWh
# dsm_df['gross_revenue_scheduled'] = (dsm_df['ml_scheduled_power'] * 0.25) * (ppa_tariff / 1000)
# dsm_df['net_revenue_realized'] = dsm_df['gross_revenue_scheduled'] + dsm_df['dsm_charge']
# dsm_df['revenue_loss'] = dsm_df['gross_revenue_actual'] - dsm_df['net_revenue_realized']
#
# return dsm_df
#
#
# def plot_deviation_frequency(df: pd.DataFrame, block_axis, time_col):
# """Plot deviation vs frequency with color-coded bands."""
# st.markdown("#### 1 Deviation and Frequency Bands")
#
# x_axis = block_axis if block_axis is not None else (df[time_col] if time_col else df.index)
# x_label = 'Block (15-min)' if block_axis is not None else ('Timestamp' if time_col else 'Index')
#
# # Create figure with secondary y-axis
# from plotly.subplots import make_subplots
#
# fig = make_subplots(
# rows=2, cols=1,
# shared_xaxes=True,
# subplot_titles=('Grid Frequency with Bands', 'Power Deviation (Actual - Scheduled)'),
# vertical_spacing=0.12,
# row_heights=[0.4, 0.6]
# )
#
# # Plot 1: Frequency with band regions
# fig.add_trace(
# go.Scatter(
# x=x_axis,
# y=df['frequency'],
# mode='lines',
# name='Grid Frequency',
# line=dict(color='#1f77b4', width=2),
# hovertemplate='%{y:.3f} Hz<extra></extra>'
# ),
# row=1, col=1
# )
#
# # Add frequency band reference lines
# band_lines = [
# (49.90, 'Band 1/2', '#d62728'),
# (49.95, 'Band 2/3', '#ff7f0e'),
# (50.03, 'Band 3/4', '#2ca02c'),
# (50.05, 'Band 4/5', '#9467bd')
# ]
#
# for freq_val, band_name, color in band_lines:
# fig.add_hline(
# y=freq_val,
# line_dash="dash",
# line_color=color,
# opacity=0.5,
# annotation_text=band_name,
# annotation_position="right",
# row=1, col=1
# )
#
# # Plot 2: Deviation with color-coding by injection type
# over_mask = df['deviation_mw'] >= 0
# under_mask = df['deviation_mw'] < 0
#
# fig.add_trace(
# go.Scatter(
# x=x_axis[over_mask],
# y=df.loc[over_mask, 'deviation_mw'],
# mode='lines',
# name='Over Injection',
# line=dict(color='#2ca02c', width=1.5),
# fill='tozeroy',
# fillcolor='rgba(44, 160, 44, 0.2)',
# hovertemplate='Over: %{y:.2f} MW<extra></extra>'
# ),
# row=2, col=1
# )
#
# fig.add_trace(
# go.Scatter(
# x=x_axis[under_mask],
# y=df.loc[under_mask, 'deviation_mw'],
# mode='lines',
# name='Under Injection',
# line=dict(color='#d62728', width=1.5),
# fill='tozeroy',
# fillcolor='rgba(214, 39, 40, 0.2)',
# hovertemplate='Under: %{y:.2f} MW<extra></extra>'
# ),
# row=2, col=1
# )
#
# fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=2, col=1)
#
# fig.update_xaxes(title_text=x_label, row=2, col=1)
# fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
# fig.update_yaxes(title_text="Deviation (MW)", row=2, col=1)
#
# fig.update_layout(
# height=700,
# showlegend=True,
# hovermode='x unified'
# )
#
# # st.plotly_chart(fig, use_container_width=True) -> use width="stretch"
# st.plotly_chart(fig, key="error_analysis_dist_plot", width="stretch")
#
#
# def plot_band_distribution(df: pd.DataFrame, block_axis, time_col):
# """Plot band-wise energy distribution and injection patterns."""
# st.markdown("#### 2 Frequency Band Distribution")
#
# # Band-wise statistics
# band_stats = df.groupby(['freq_band', 'injection_type'], observed=True).agg({
# 'deviation_kwh': 'sum',
# 'dsm_units_kwh': 'sum',
# 'dsm_charge': 'sum'
# }).reset_index()
#
# band_stats['deviation_mwh'] = band_stats['deviation_kwh'] / 1000
# band_stats['dsm_units_mwh'] = band_stats['dsm_units_kwh'] / 1000
#
# # Create subplots for band analysis
# from plotly.subplots import make_subplots
#
# fig = make_subplots(
# rows=1, cols=2,
# subplot_titles=('Energy Distribution by Band', 'DSM Charges by Band'),
# specs=[[{"type": "bar"}, {"type": "bar"}]]
# )
#
# # Plot 1: Energy distribution
# for inj_type in band_stats['injection_type'].unique():
# data = band_stats[band_stats['injection_type'] == inj_type]
# color = '#2ca02c' if inj_type == 'Over Injection' else '#d62728'
#
# fig.add_trace(
# go.Bar(
# x=data['freq_band'],
# y=data['deviation_mwh'],
# name=f'{inj_type} (MWh)',
# marker_color=color,
# hovertemplate='%{x}<br>%{y:.2f} MWh<extra></extra>'
# ),
# row=1, col=1
# )
#
# # Plot 2: DSM charges
# for inj_type in band_stats['injection_type'].unique():
# data = band_stats[band_stats['injection_type'] == inj_type]
# color = '#2ca02c' if inj_type == 'Over Injection' else '#d62728'
#
# fig.add_trace(
# go.Bar(
# x=data['freq_band'],
# y=data['dsm_charge'],
# name=f'{inj_type} (Rs)',
# marker_color=color,
# showlegend=False,
# hovertemplate='%{x}<br>%{y:,.0f}<extra></extra>'
# ),
# row=1, col=2
# )
#
# fig.update_xaxes(title_text="Frequency Band", row=1, col=1)
# fig.update_xaxes(title_text="Frequency Band", row=1, col=2)
# fig.update_yaxes(title_text="Energy (MWh)", row=1, col=1)
# fig.update_yaxes(title_text="DSM Charge (Rs)", row=1, col=2)
#
# fig.update_layout(
# height=500,
# barmode='group',
# showlegend=True
# )
#
# # st.plotly_chart(fig, use_container_width=True) -> use width="stretch"
# st.plotly_chart(fig, key="test_agg_band_dist_plot", width="stretch")
#
# # Display band statistics table
# st.markdown("##### Band-wise Statistics Table")
#
# display_band_stats = band_stats.copy()
# display_band_stats['DSM Charge (Rs)'] = display_band_stats['dsm_charge'].apply(lambda x: f"{x:,.2f}")
# display_band_stats['Energy (MWh)'] = display_band_stats['deviation_mwh'].apply(lambda x: f"{x:.2f}")
# display_band_stats = display_band_stats[['freq_band', 'injection_type', 'Energy (MWh)', 'DSM Charge (Rs)']]
# display_band_stats.columns = ['Frequency Band', 'Injection Type', 'Energy (MWh)', 'DSM Charge (Rs)']
#
# st.dataframe(display_band_stats, use_container_width=True, hide_index=True)
#
#
# def plot_financial_impact(df: pd.DataFrame, block_axis, time_col):
# """Plot financial impact and revenue analysis."""
# st.markdown("#### 3 Financial Impact Analysis")
#
# x_axis = block_axis if block_axis is not None else (df[time_col] if time_col else df.index)
# x_label = 'Block (15-min)' if block_axis is not None else ('Timestamp' if time_col else 'Index')
#
# # Create figure
# fig = go.Figure()
#
# # Plot DSM charges (cumulative)
# dsm_cumulative = df['dsm_charge'].cumsum()
#
# fig.add_trace(
# go.Scatter(
# x=x_axis,
# y=dsm_cumulative,
# mode='lines',
# name='Cumulative DSM',
# line=dict(color='#1f77b4', width=2),
# fill='tozeroy',
# fillcolor='rgba(31, 119, 180, 0.2)',
# hovertemplate='Cumulative: %{y:,.2f}<extra></extra>'
# )
# )
#
# # Add individual charge markers for significant values
# significant_mask = df['dsm_charge'].abs() > df['dsm_charge'].abs().quantile(0.90)
#
# fig.add_trace(
# go.Scatter(
# x=x_axis[significant_mask],
# y=dsm_cumulative[significant_mask],
# mode='markers',
# name='Significant Charges',
# marker=dict(
# color=np.where(df.loc[significant_mask, 'dsm_charge'] > 0, '#2ca02c', '#d62728'),
# size=8,
# symbol='circle'
# ),
# hovertemplate='%{y:,.2f}<extra></extra>'
# )
# )
#
# fig.update_layout(
# title='Cumulative DSM Charges Over Time',
# xaxis_title=x_label,
# yaxis_title='Cumulative DSM (Rs)',
# height=500,
# hovermode='x unified'
# )
#
# # st.plotly_chart(fig, use_container_width=True) -> use width="stretch"
# st.plotly_chart(fig, key="test_agg_financial_plot", width="stretch")
#
# # Financial metrics
# col1, col2, col3, col4 = st.columns(4)
#
# total_dsm = df['dsm_charge'].sum()
# total_receivable = df[df['dsm_type'] == 'Receivable']['dsm_charge'].sum()
# total_payable = df[df['dsm_type'] == 'Payable']['dsm_charge'].sum()
# net_dsm = total_receivable + total_payable # payable is already negative
#
# with col1:
# st.metric(
# "Total Receivable",
# f"{total_receivable:,.0f}",
# delta="Revenue" if total_receivable > 0 else None,
# delta_color="normal"
# )
#
# with col2:
# st.metric(
# "Total Payable",
# f"{abs(total_payable):,.0f}",
# delta="Penalty" if total_payable < 0 else None,
# delta_color="inverse"
# )
#
# with col3:
# st.metric(
# "Net DSM",
# f"{net_dsm:,.0f}",
# delta="Positive" if net_dsm > 0 else "Negative",
# delta_color="normal" if net_dsm > 0 else "inverse"
# )
#
# with col4:
# avg_dsm_per_block = total_dsm / len(df) if len(df) > 0 else 0
# st.metric(
# "Avg DSM/Block",
# f"{avg_dsm_per_block:,.2f}",
# help="Average DSM charge per 15-min block"
# )
#
#
# def display_dsm_tables(df: pd.DataFrame, reference_rate: float, ppa_tariff: float, model_name: str):
# """Display comprehensive DSM calculation tables for the entire test set."""
# st.markdown("""
# Comprehensive DSM calculations for the entire test set. These tables provide
# band-wise breakdowns, financial summaries, and performance metrics.
# """)
#
# # Table 1: Overall Summary
# st.markdown("#### Overall DSM Summary")
#
# total_actual_gen = df['actual_power'].sum() * 0.25 / 1000 # MWh
# total_scheduled_gen = df['ml_scheduled_power'].sum() * 0.25 / 1000 # MWh
# total_deviation = total_actual_gen - total_scheduled_gen
#
# total_receivable = df[df['dsm_type'] == 'Receivable']['dsm_charge'].sum()
# total_payable = df[df['dsm_type'] == 'Payable']['dsm_charge'].sum()
# net_dsm = total_receivable + total_payable
#
# gross_revenue_actual = df['gross_revenue_actual'].sum()
# gross_revenue_scheduled = df['gross_revenue_scheduled'].sum()
# net_revenue_realized = df['net_revenue_realized'].sum()
# total_revenue_loss = df['revenue_loss'].sum()
#
# revenue_loss_pct = (total_revenue_loss / gross_revenue_actual * 100) if gross_revenue_actual != 0 else 0
#
# summary_data = {
# 'Metric': [
# 'Total Actual Generation (MWh)',
# 'Total Scheduled Generation (MWh)',
# 'Total Deviation (MWh)',
# 'Deviation %',
# '',
# 'Gross Revenue (Actual) (Rs)',
# 'Gross Revenue (Scheduled) (Rs)',
# 'DSM Receivable (Rs)',
# 'DSM Payable (Rs)',
# 'Net DSM Settlement (Rs)',
# 'Net Revenue Realized (Rs)',
# 'Revenue Loss (Rs)',
# 'Revenue Loss %'
# ],
# 'Value': [
# f"{total_actual_gen:,.2f}",
# f"{total_scheduled_gen:,.2f}",
# f"{total_deviation:+,.2f}",
# f"{(total_deviation/total_scheduled_gen*100):+.2f}%" if total_scheduled_gen != 0 else "N/A",
# '',
# f"{gross_revenue_actual:,.2f}",
# f"{gross_revenue_scheduled:,.2f}",
# f"{total_receivable:,.2f}",
# f"{abs(total_payable):,.2f}",
# f"{net_dsm:+,.2f}",
# f"{net_revenue_realized:,.2f}",
# f"{total_revenue_loss:,.2f}",
# f"{revenue_loss_pct:.2f}%"
# ]
# }
#
# summary_df = pd.DataFrame(summary_data)
# st.dataframe(summary_df, use_container_width=True, hide_index=True)
#
# # Table 2: Band-wise Breakdown
# st.markdown("---")
# st.markdown("#### Band-wise Breakdown")
#
# band_breakdown = df.groupby(['freq_band', 'injection_type'], observed=True).agg({
# 'deviation_kwh': 'sum',
# 'dsm_units_kwh': 'sum',
# 'dsm_charge': 'sum'
# }).reset_index()
#
# band_breakdown['Energy (MWh)'] = band_breakdown['deviation_kwh'] / 1000
# band_breakdown['DSM Units (MWh)'] = band_breakdown['dsm_units_kwh'] / 1000
# band_breakdown['DSM Charge (Rs)'] = band_breakdown['dsm_charge']
# band_breakdown['Count'] = df.groupby(['freq_band', 'injection_type'], observed=True).size().values
#
# band_display = band_breakdown[[
# 'freq_band', 'injection_type', 'Count', 'Energy (MWh)',
# 'DSM Units (MWh)', 'DSM Charge (Rs)'
# ]].copy()
#
# band_display.columns = [
# 'Frequency Band', 'Injection Type', 'Block Count',
# 'Energy (MWh)', 'DSM Units (MWh)', 'DSM Charge (Rs)'
# ]
#
# band_display['Energy (MWh)'] = band_display['Energy (MWh)'].apply(lambda x: f"{x:,.2f}")
# band_display['DSM Units (MWh)'] = band_display['DSM Units (MWh)'].apply(lambda x: f"{x:,.2f}")
# band_display['DSM Charge (Rs)'] = band_display['DSM Charge (Rs)'].apply(lambda x: f"{x:,.2f}")
#
# st.dataframe(band_display, use_container_width=True, hide_index=True)
#
# # Table 3: Error Band Analysis
# st.markdown("---")
# st.markdown("#### Error Band Analysis")
#
# df['error_band'] = pd.cut(
# df['abs_error_pct'],
# bins=[0, 5, 10, 15, 20, 100],
# labels=['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
# )
#
# error_breakdown = df.groupby(['error_band', 'injection_type'], observed=True).agg({
# 'deviation_kwh': 'sum',
# 'dsm_charge': 'sum'
# }).reset_index()
#
# error_breakdown['Count'] = df.groupby(['error_band', 'injection_type'], observed=True).size().values
# error_breakdown['Energy (MWh)'] = error_breakdown['deviation_kwh'].abs() / 1000
# error_breakdown['DSM Charge (Rs)'] = error_breakdown['dsm_charge']
#
# error_display = error_breakdown[[
# 'error_band', 'injection_type', 'Count', 'Energy (MWh)', 'DSM Charge (Rs)'
# ]].copy()
#
# error_display.columns = [
# 'Error Band', 'Injection Type', 'Block Count', 'Energy (MWh)', 'DSM Charge (Rs)'
# ]
#
# error_display['Energy (MWh)'] = error_display['Energy (MWh)'].apply(lambda x: f"{x:,.2f}")
# error_display['DSM Charge (Rs)'] = error_display['DSM Charge (Rs)'].apply(lambda x: f"{x:,.2f}")
#
# st.dataframe(error_display, use_container_width=True, hide_index=True)
#
# # Table 4: Daily Aggregates
# time_col = None
# for cand in ["timestamp", "date", "datetime"]:
# if cand in df.columns:
# time_col = cand
# break
#
# if time_col:
# st.markdown("---")
# st.markdown("#### Daily DSM Aggregates")
#
# df[time_col] = pd.to_datetime(df[time_col])
# df['date'] = df[time_col].dt.date
#
# daily_agg = df.groupby('date').agg({
# 'actual_power': lambda x: (x.sum() * 0.25 / 1000), # MWh
# 'ml_scheduled_power': lambda x: (x.sum() * 0.25 / 1000), # MWh
# 'dsm_charge': 'sum',
# 'gross_revenue_actual': 'sum',
# 'net_revenue_realized': 'sum',
# 'revenue_loss': 'sum'
# }).reset_index()
#
# daily_agg['deviation'] = daily_agg['actual_power'] - daily_agg['ml_scheduled_power']
#
# daily_agg.columns = [
# 'Date', 'Actual Gen (MWh)', 'Scheduled Gen (MWh)', 'Net DSM (Rs)',
# 'Gross Revenue (Rs)', 'Net Revenue (Rs)', 'Revenue Loss (Rs)', 'Deviation (MWh)'
# ]
#
# # Format for display
# daily_display = daily_agg.copy()
# daily_display['Actual Gen (MWh)'] = daily_display['Actual Gen (MWh)'].apply(lambda x: f"{x:,.2f}")
# daily_display['Scheduled Gen (MWh)'] = daily_display['Scheduled Gen (MWh)'].apply(lambda x: f"{x:,.2f}")
# daily_display['Deviation (MWh)'] = daily_display['Deviation (MWh)'].apply(lambda x: f"{x:+,.2f}")
# daily_display['Net DSM (Rs)'] = daily_display['Net DSM (Rs)'].apply(lambda x: f"{x:+,.2f}")
# daily_display['Gross Revenue (Rs)'] = daily_display['Gross Revenue (Rs)'].apply(lambda x: f"{x:,.2f}")
# daily_display['Net Revenue (Rs)'] = daily_display['Net Revenue (Rs)'].apply(lambda x: f"{x:,.2f}")
# daily_display['Revenue Loss (Rs)'] = daily_display['Revenue Loss (Rs)'].apply(lambda x: f"{x:,.2f}")
#
# st.dataframe(daily_display, use_container_width=True, hide_index=True, height=400)
#
# # Export options
# st.markdown("---")
# st.markdown("#### Export DSM Tables")
#
# col1, col2 = st.columns(2)
#
# with col1:
# # Export summary
# csv_summary = summary_df.to_csv(index=False)
# plant_name = st.session_state.get('plant_selected', 'unknown_plant')
# st.download_button(
# label=" Download Summary Table",
# data=csv_summary,
# file_name=f"dsm_summary_{plant_name}_{model_name}.csv",
# mime="text/csv"
# )
#
# with col2:
# # Export band breakdown
# csv_band = band_display.to_csv(index=False)
# st.download_button(
# label=" Download Band Breakdown",
# data=csv_band,
# file_name=f"dsm_band_breakdown_{plant_name}_{model_name}.csv",
# mime="text/csv"
# )
#
# if time_col:
# # Export daily aggregates
# csv_daily = daily_display.to_csv(index=False)
# st.download_button(
# label=" Download Daily Aggregates",
# data=csv_daily,
# file_name=f"dsm_daily_aggregates_{plant_name}_{model_name}.csv",
# mime="text/csv"
# )
#
#
# st.caption(
# "Hover shows percentiles only (no MW values) for the F10F90 bands."
# )
