"""
Model Comparison Page
=====================

Handles:
- Side-by-side model comparison
- Performance metrics dashboard
- Feature importance visualization
- Custom ensemble creation
- Model export and versioning

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

from app.utils.page_summary import render_page_summary, get_page_context


def show():
    """Main function for model comparison page."""
    st.header(" Model Comparison & Analysis")
    st.markdown("Compare models, analyze performance, and create custom ensembles")

    # AI-generated page summary
    render_page_summary("Model Comparison", get_page_context("Model Comparison"))

    # Check if models are available and metrics exist
    if not st.session_state.models_trained:
        st.warning(" Please train or load models in **Model Training** page")
        return

    if not st.session_state.model_metrics:
        st.info("i Run training or execute the test-only evaluation for a loaded model to populate metrics.")
        return

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        " Performance Dashboard",
        " Detailed Comparison",
        " Feature Importance",
        " Custom Ensemble"
    ])

    with tab1:
        show_performance_dashboard()

    with tab2:
        show_detailed_comparison()

    with tab3:
        show_feature_importance()

    with tab4:
        show_custom_ensemble()


def show_performance_dashboard():
    """Performance metrics dashboard."""
    st.markdown("### Performance Metrics Dashboard")

    metrics = st.session_state.model_metrics

    # Metric selection
    metric_type = st.radio(
        "Select Metric",
        ["MAE", "RMSE", "R2", "sMAPE"],
        horizontal=True
    )

    # Prepare data
    models = []
    train_vals = []
    val_vals = []
    test_vals = []

    for model_name, model_metrics in metrics.items():
        models.append(model_name.replace('_', ' ').title())
        # Note: We don't have train metrics, using val as proxy
        val_vals.append(model_metrics['val'][metric_type])
        test_vals.append(model_metrics['test'][metric_type])

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Validation',
        x=models,
        y=val_vals,
        marker_color='#2E86AB',
        text=[f'{v:.3f}' for v in val_vals],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='Test',
        x=models,
        y=test_vals,
        marker_color='#F18F01',
        text=[f'{v:.3f}' for v in test_vals],
        textposition='outside'
    ))

    title_suffix = "(Lower is Better)" if metric_type in ["MAE", "RMSE", "sMAPE"] else "(Higher is Better)"

    fig.update_layout(
        title=f'{metric_type} Comparison {title_suffix}',
        xaxis_title='Model',
        yaxis_title=metric_type,
        barmode='group',
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, width='stretch')

    # Leaderboard
    st.markdown("####  Model Leaderboard")

    # Create leaderboard DataFrame
    leaderboard_data = []
    for model_name, model_metrics in metrics.items():
        leaderboard_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Val MAE': model_metrics['val']['MAE'],
            'Val RMSE': model_metrics['val']['RMSE'],
            'Val R2': model_metrics['val']['R2'],
            'Test MAE': model_metrics['test']['MAE'],
            'Test RMSE': model_metrics['test']['RMSE'],
            'Test R2': model_metrics['test']['R2']
        })

    leaderboard_df = pd.DataFrame(leaderboard_data)

    # Sort by validation MAE
    leaderboard_df = leaderboard_df.sort_values('Val MAE')
    leaderboard_df.insert(0, 'Rank', range(1, len(leaderboard_df) + 1))

    # Format numbers
    for col in leaderboard_df.columns:
        if col not in ['Rank', 'Model']:
            leaderboard_df[col] = leaderboard_df[col].apply(lambda x: f'{x:.4f}')

    st.dataframe(leaderboard_df)

    # Best model highlight
    best_model = leaderboard_df.iloc[0]
    st.success(f" **Best Model:** {best_model['Model']} (Val MAE: {best_model['Val MAE']})")


def show_detailed_comparison():
    """Detailed model comparison with statistical tests."""
    st.markdown("### Detailed Model Comparison")

    prediction_dfs = st.session_state.prediction_dfs

    if not prediction_dfs:
        st.info("Generate predictions first to see detailed comparison")
        return

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        model1 = st.selectbox(
            "Model 1",
            options=list(prediction_dfs.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col2:
        model2 = st.selectbox(
            "Model 2",
            options=[m for m in prediction_dfs.keys() if m != model1],
            format_func=lambda x: x.replace('_', ' ').title() if x else ""
        )

    # Check if we have two models to compare
    if not model2 or model1 == model2:
        st.warning(" Please select at least two different models to compare.")
        return

    # Get predictions
    pred1 = prediction_dfs[model1]
    pred2 = prediction_dfs[model2]

    # Error comparison scatter
    st.markdown("#### Error Comparison Scatter Plot")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pred1['ml_error'],
        y=pred2['ml_error'],
        mode='markers',
        marker=dict(
            size=5,
            color=pred1['selection_confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=pred1['ml_selected_fsp'],
        hovertemplate=f'{model1}: %{{x:.2f}}<br>{model2}: %{{y:.2f}}<br>FSP: %{{text}}'
    ))

    # Add diagonal
    max_error = max(pred1['ml_error'].max(), pred2['ml_error'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_error],
        y=[0, max_error],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Equal Error'
    ))

    fig.update_layout(
        title=f'{model1.title()} vs {model2.title()} Error Comparison',
        xaxis_title=f'{model1.title()} Error (MW)',
        yaxis_title=f'{model2.title()} Error (MW)',
        height=500
    )

    st.plotly_chart(fig, width='stretch')

    # Win/Loss analysis
    st.markdown("#### Win/Loss Analysis")

    col1, col2, col3 = st.columns(3)

    model1_wins = (pred1['ml_error'] < pred2['ml_error']).sum()
    model2_wins = (pred2['ml_error'] < pred1['ml_error']).sum()
    ties = (pred1['ml_error'] == pred2['ml_error']).sum()

    with col1:
        st.metric(f"{model1.title()} Wins", f"{model1_wins:,}",
                  f"{(model1_wins/len(pred1)*100):.1f}%")

    with col2:
        st.metric(f"{model2.title()} Wins", f"{model2_wins:,}",
                  f"{(model2_wins/len(pred2)*100):.1f}%")

    with col3:
        st.metric("Ties", f"{ties:,}", f"{(ties/len(pred1)*100):.1f}%")

    # Error difference distribution
    st.markdown("#### Error Difference Distribution")

    error_diff = pred1['ml_error'] - pred2['ml_error']

    fig = px.histogram(
        error_diff,
        nbins=50,
        title=f'Error Difference ({model1.title()} - {model2.title()})',
        labels={'value': 'Error Difference (MW)'}
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Equal Performance"
    )

    fig.add_vline(
        x=error_diff.mean(),
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {error_diff.mean():.3f}"
    )

    st.plotly_chart(fig, width='stretch')

    if error_diff.mean() < 0:
        st.success(f" {model1.title()} performs better on average by {abs(error_diff.mean()):.3f} MW")
    else:
        st.success(f" {model2.title()} performs better on average by {error_diff.mean():.3f} MW")


def show_feature_importance():
    """Feature importance visualization."""
    st.markdown("### Feature Importance Analysis")

    models = st.session_state.models_trained

    # Filter models with feature importance
    tree_models = {k: v for k, v in models.items()
                   if hasattr(v, 'feature_importances_') or hasattr(v, 'get_feature_importance')}

    if not tree_models:
        st.info("No tree-based models available. Feature importance only available for tree-based models (RF, XGB, LGB)")
        return

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=list(tree_models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )

    model = tree_models[model_name]
    feature_cols = st.session_state.feature_columns

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # Handle case where model dropped features
        if len(importances) != len(feature_cols):
            feature_cols = feature_cols[:len(importances)]

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Top N selection
        top_n = st.slider("Number of top features to display", 10, 50, 20)

        importance_df_top = importance_df.head(top_n)

        # Plot
        fig = px.bar(
            importance_df_top,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances - {model_name.replace("_", " ").title()}',
            color='Importance',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(height=max(400, top_n * 20))
        st.plotly_chart(fig, width='stretch')

        # Download option
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label=" Download Feature Importances (CSV)",
            data=csv,
            file_name=f"feature_importance_{model_name}.csv",
            mime="text/csv"
        )

        # Feature categories
        st.markdown("#### Feature Category Analysis")

        # Categorize features
        categories = {
            'Rolling': [f for f in feature_cols if 'rolling' in f],
            'Time': [f for f in feature_cols if any(x in f for x in ['hour', 'dow', 'month', 'weekend'])],
            'FSP Forecasts': [f for f in feature_cols if 'forecast_power' in f],
            'Weather': [f for f in feature_cols if any(x in f for x in ['windspeed', 'ghirr', 'flowrate'])],
            'Other': []
        }

        # Assign uncategorized features
        all_categorized = sum(categories.values(), [])
        categories['Other'] = [f for f in feature_cols if f not in all_categorized]

        # Calculate importance by category
        category_importance = {}
        for cat, features in categories.items():
            cat_importance = importance_df[importance_df['Feature'].isin(features)]['Importance'].sum()
            category_importance[cat] = cat_importance

        # Plot
        fig = px.pie(
            values=list(category_importance.values()),
            names=list(category_importance.keys()),
            title='Feature Importance by Category',
            hole=0.4
        )
        st.plotly_chart(fig, width='stretch')

    elif model_name == 'ensemble' and hasattr(model, 'get_feature_importance'):
        st.info("Ensemble model - showing LightGBM component importance")
        importance_info = model.get_feature_importance()

        if 'feature_importances' in importance_info:
            importances = importance_info['feature_importances']

            importance_df = pd.DataFrame({
                'Feature': feature_cols[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            top_n = st.slider("Number of top features to display", 10, 50, 20)
            importance_df_top = importance_df.head(top_n)

            fig = px.bar(
                importance_df_top,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top {top_n} Feature Importances - Ensemble (LightGBM Component)',
                color='Importance',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(height=max(400, top_n * 20))
            st.plotly_chart(fig, width='stretch')


def show_custom_ensemble():
    """Create custom ensemble from trained models."""
    st.markdown("### Custom Ensemble Creation")
    st.info("Create a weighted ensemble from your trained models")

    models = st.session_state.models_trained
    predictions = st.session_state.model_predictions

    if len(models) < 2:
        st.warning(" Train at least 2 models to create an ensemble")
        return

    st.markdown("#### Select Models and Weights")

    weights = {}
    selected_models = []

    for model_name in models.keys():
        col1, col2 = st.columns([3, 1])

        with col1:
            include = st.checkbox(
                f"Include {model_name.replace('_', ' ').title()}",
                value=True,
                key=f"ensemble_{model_name}"
            )

        with col2:
            if include:
                weight = st.number_input(
                    "Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(models),
                    step=0.05,
                    key=f"weight_{model_name}"
                )
                weights[model_name] = weight
                selected_models.append(model_name)

    # Normalize weights
    if weights:
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        st.markdown("#### Normalized Weights")
        weight_df = pd.DataFrame({
            'Model': [k.replace('_', ' ').title() for k in normalized_weights.keys()],
            'Weight': [f'{v:.3f}' for v in normalized_weights.values()],
            'Percentage': [f'{v*100:.1f}%' for v in normalized_weights.values()]
        })
        st.dataframe(weight_df)

        # Create ensemble predictions
        if st.button(" Create Ensemble", type="primary"):
            with st.spinner("Creating ensemble predictions..."):
                # Combine predictions
                val_preds = []
                test_preds = []

                for model_name, weight in normalized_weights.items():
                    val_preds.append(predictions[model_name]['val'] * weight)
                    test_preds.append(predictions[model_name]['test'] * weight)

                ensemble_val_pred = np.sum(val_preds, axis=0)
                ensemble_test_pred = np.sum(test_preds, axis=0)

                # Calculate metrics
                val_df = st.session_state.val_df
                test_df = st.session_state.test_df

                from app.pages.model_training import calculate_metrics

                val_metrics = calculate_metrics(val_df['target_horizon'], ensemble_val_pred)
                test_metrics = calculate_metrics(test_df['target_horizon'], ensemble_test_pred)

                # Display results
                st.success(" Ensemble created successfully!")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Val MAE", f"{val_metrics['MAE']:.3f}")

                with col2:
                    st.metric("Val R2", f"{val_metrics['R2']:.3f}")

                with col3:
                    st.metric("Test MAE", f"{test_metrics['MAE']:.3f}")

                with col4:
                    st.metric("Test R2", f"{test_metrics['R2']:.3f}")

                # Compare with individual models
                st.markdown("#### Ensemble vs Individual Models")

                comparison_data = []
                for model_name in selected_models:
                    model_metrics = st.session_state.model_metrics[model_name]
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Val MAE': model_metrics['val']['MAE'],
                        'Test MAE': model_metrics['test']['MAE']
                    })

                comparison_data.append({
                    'Model': 'Custom Ensemble',
                    'Val MAE': val_metrics['MAE'],
                    'Test MAE': test_metrics['MAE']
                })

                comp_df = pd.DataFrame(comparison_data)

                fig = px.bar(
                    comp_df,
                    x='Model',
                    y=['Val MAE', 'Test MAE'],
                    barmode='group',
                    title='Ensemble Performance Comparison'
                )

                st.plotly_chart(fig, width='stretch')

                # Check if ensemble is best
                best_val_mae = min([m['val']['MAE'] for m in st.session_state.model_metrics.values()])

                if val_metrics['MAE'] < best_val_mae:
                    st.success(f" Ensemble outperforms all individual models! Improvement: {(best_val_mae - val_metrics['MAE']):.3f} MW")
                else:
                    st.info(f"Ensemble MAE: {val_metrics['MAE']:.3f} MW (Best individual: {best_val_mae:.3f} MW)")
