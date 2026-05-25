"""
AI-Powered Page Summary Generator
==================================

Generates concise, bullet-point summaries for each page using Ollama.

Maintainer: Project Team
Date: February 2026
"""

import streamlit as st
from typing import Optional
from app.utils.ollama_insights import OllamaInsightsGenerator, DEFAULT_MODEL


def render_page_summary(page_name: str, page_context: str, model: Optional[str] = None):
    """
    Render an AI-generated summary for a page in a collapsible expander with strict line-by-line formatting.

    Args:
        page_name: Name of the page (e.g., "Data Selection")
        page_context: Context about what the page contains and does
        model: Optional Ollama model to use (uses default if not specified)
    """
    try:
        generator = OllamaInsightsGenerator()

        # Use provided model or default
        model_to_use = model or generator.selected_model

        # Create ULTRA strict prompt with explicit line break instruction
        prompt = f"""Generate bullet points for "{page_name}" page.

Context: {page_context}

OUTPUT EXACTLY IN THIS FORMAT - NO EXCEPTIONS:
- First point here
- Second point here
- Third point here
- Fourth point here
- Fifth point here

RULES:
1. Return ONLY the 5 bullet points above
2. Each bullet is ONE line starting with "- "
3. NO text before, after, or between bullets
4. NO paragraphs
5. Each point is 8-15 words max
6. No periods at end

Generate now:"""

        # Generate summary
        summary = generator.generate_insights(
            prompt,
            model=model_to_use,
            temperature=0.05,  # Ultra low for strict format
            max_tokens=150
        )

        # Aggressive post-processing
        lines = summary.strip().split('\n')

        # Extract only lines that are bullets
        bullet_lines = []
        for line in lines:
            line = line.strip()
            # Handle various bullet formats
            if line.startswith('-'):
                bullet_lines.append('- ' + line.lstrip('-').strip())
            elif line and not any(skip in line.lower() for skip in ['format', 'rules:', 'output', 'generate']):
                # If it looks like content, add bullet
                if len(line) > 5 and len(line) < 150:
                    bullet_lines.append('- ' + line)

        # Take only first 5-6 bullets
        bullet_lines = bullet_lines[:6]

        # Create final formatted output with explicit line breaks
        if bullet_lines:
            formatted_output = '\n\n'.join(bullet_lines)
        else:
            formatted_output = "Page summary could not be generated"

        # Display in a collapsible expander using code block for strict formatting
        with st.expander("What can you do on this page? (AI Summary)", expanded=False):
            # Use st.text instead of markdown to preserve exact formatting
            st.text(formatted_output)

    except Exception as e:
        # Ollama not available - continue without AI summary
        pass  # Silently skip - AI summary is optional


def get_page_context(page_name: str) -> str:
    """
    Get context description for each page.

    Args:
        page_name: Name of the page

    Returns:
        Context string describing the page's purpose
    """
    contexts = {
        "Data Selection": (
            "This page allows users to select a power plant from available options, "
            "load historical power generation data, configure the data window (months), "
            "visualize data gaps and completeness, and pivot FSP forecast data into columns."
        ),
        "Feature Engineering": (
            "This page handles temporal data splitting into train/validation/test sets, "
            "creates rolling window features (mean, std, min, max), generates time-based features "
            "(hour, day, month, season), encodes categorical features, and displays feature statistics."
        ),
        "FSP Selection": (
            "This page calculates Mean Absolute Error (MAE) for each Forecast Service Provider (FSP), "
            "ranks FSPs by accuracy, allows users to select which FSPs to include in model training, "
            "and visualizes FSP forecast accuracy comparisons."
        ),
        "Model Training": (
            "This page provides model selection from classical ML (Ridge, Lasso, Random Forest, XGBoost, LightGBM), "
            "deep learning (ANN, LSTM, GRU, CNN), and hybrid/ensemble models. It allows hyperparameter tuning, "
            "trains selected models with progress tracking, evaluates performance with MAE/RMSE/R2 metrics, "
            "and saves trained models for inference."
        ),
        "Predictions & Visualization": (
            "This page generates predictions for all trained models on the test set, "
            "exports predictions to CSV files, provides quantile-based forecast ribbons with confidence intervals, "
            "enables time series comparison between actual power, ML predictions, and FSP forecasts, "
            "shows FSP selection analysis, and displays error analysis between ML and manual scheduling."
        ),
        "Model Comparison": (
            "This page provides side-by-side comparison of all trained models, "
            "displays performance metrics (MAE, RMSE, MAPE, R2) in sortable tables, "
            "visualizes metric distributions with box plots and bar charts, "
            "and helps identify the best performing model for deployment."
        )
    }

    return contexts.get(page_name, "Machine learning page for power forecasting")
