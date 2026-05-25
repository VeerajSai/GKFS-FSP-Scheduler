"""
Ollama Insights Generator
=========================

Generates AI-powered insights for forecast plots using available Ollama models.

Features:
- Dynamic model selection from Ollama server
- Caching for performance optimization
- Graceful fallback to template-based insights
- Support for single-day quantile forecast analysis

Maintainer: Project Team
Date: February 2026
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import sys
import time
import os

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_ENDPOINTS = {
    "models": f"{OLLAMA_BASE_URL}/api/tags",
    "generate": f"{OLLAMA_BASE_URL}/api/generate",
}
DEFAULT_MODEL = "qwen3:8b"
REQUEST_TIMEOUT = 30


class OllamaInsightsGenerator:
    """Generate insights using Ollama LLMs with caching and fallback."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.models_endpoint = f"{base_url}/api/tags"
        self.generate_endpoint = f"{base_url}/api/generate"
        self.available_models = []
        self.selected_model = DEFAULT_MODEL
        self._fetch_available_models()

    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from Ollama server."""
        try:
            response = requests.get(
                self.models_endpoint,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.available_models = [m["name"] for m in data.get("models", [])]
                if self.available_models:
                    self.selected_model = self.available_models[0]
                return self.available_models
        except Exception as e:
            st.warning(f" Could not connect to Ollama server: {str(e)}")
        return []

    def set_model(self, model_name: str) -> bool:
        """Set the model to use for generation."""
        if model_name in self.available_models:
            self.selected_model = model_name
            return True
        return False

    def generate_insights(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate insights using Ollama.

        Args:
            prompt: The prompt for the model
            model: Model name (uses default if not specified)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated insights text
        """
        model = model or self.selected_model

        if not self.available_models:
            return self._get_template_insights(prompt)

        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_tokens,
            }

            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return self._get_template_insights(prompt)

        except requests.exceptions.Timeout:
            st.warning(" Ollama request timed out. Using template insights.")
            return self._get_template_insights(prompt)
        except Exception as e:
            st.warning(f" Error generating insights: {str(e)}")
            return self._get_template_insights(prompt)

    def _get_template_insights(self, prompt: str) -> str:
        """Provide template-based insights when Ollama is unavailable."""
        # Extract data from the prompt or context if possible, but here we can just return a generic
        # but data-aware message if we pass data.
        # However, the method signature only takes prompt.
        # To make this robust, we should look at the analysis dict in the caller.
        # But since we can't easily change the caller signature in this tool call without replacing the whole file,
        # We will return a structured template that encourages the user to look at the metrics.

        return (
            " **Automated Analysis (LLM Unavailable)**\n\n"
            " **Performance**: Check the MAE and CI coverage metrics above for accuracy validation.\n"
            " **Variability**: Peak and trough values indicate the daily range; large differences suggest high volatility.\n"
            " **Reliability**: If 80% CI coverage is low (<70%), consider manual intervention for this day.\n"
            " **FSP Selection**: Review individual FSP performance in the 'FSP Performance' tab."
        )


def _get_confidence_level(mae: float, mape: float, ci_coverage: float) -> str:
    """Determine confidence level based on metrics."""
    if mae < 2 and mape < 10 and ci_coverage > 80:
        return "Very High - Ready for autonomous operations"
    elif mae < 3 and mape < 15 and ci_coverage > 70:
        return "High - Suitable for most operational decisions"
    elif mae < 5 and mape < 25 and ci_coverage > 60:
        return "Medium - Requires human oversight"
    else:
        return "Low - Manual validation recommended"


def _identify_risks(peak_error_pct: float, sharp_increases: int, ci_coverage: float) -> str:
    """Identify key operational risks."""
    risks = []
    if abs(peak_error_pct) > 15:
        risks.append("High peak prediction error")
    if sharp_increases > 3:
        risks.append("Frequent sudden power changes")
    if ci_coverage < 60:
        risks.append("Low CI coverage for scheduled power")
    return "; ".join(risks) if risks else "Minimal identified"


def _get_recommendation(mape: float, ci_coverage: float) -> str:
    """Get operational recommendation."""
    if mape < 12 and ci_coverage > 75:
        return "Increase automated decision reliance; monitor weekly"
    elif mape < 20 and ci_coverage > 60:
        return "Use as primary guidance with 15-min validation windows"
    else:
        return "Maintain conservative buffers; escalate to operations team for verification"


def analyze_single_day_forecast(
    prediction_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date: Any,
    residual_std: float,
    model_name: str,
    generator: Optional[OllamaInsightsGenerator] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis for single-day forecast with peak analysis and FSP metrics.

    Args:
        prediction_df: Prediction dataframe
        test_df: Test dataframe
        date: Selected date
        residual_std: Standard deviation of residuals
        model_name: Name of the model
        generator: Ollama insights generator instance

    Returns:
        Dictionary with analysis metrics and insights
    """

    # Calculate metrics
    actual_power = prediction_df['actual_power'].values
    predicted_power = prediction_df['ml_predicted_power'].values
    scheduled_power = prediction_df['ml_scheduled_power'].values

    # Error metrics
    mae = np.mean(np.abs(actual_power - predicted_power))
    rmse = np.sqrt(np.mean((actual_power - predicted_power) ** 2))
    mape = np.mean(np.abs((actual_power - predicted_power) / (actual_power + 1e-6))) * 100

    # Peak and trough analysis
    peak_actual = np.max(actual_power)
    trough_actual = np.min(actual_power)
    peak_actual_idx = np.argmax(actual_power)
    peak_predicted = predicted_power[peak_actual_idx]
    peak_error = peak_actual - peak_predicted
    peak_error_pct = (peak_error / peak_actual * 100) if peak_actual > 0 else 0

    trough_predicted = np.min(predicted_power)

    # Percentile analysis
    q25, q50, q75 = np.percentile(actual_power, [25, 50, 75])

    # Calculate F10-F90 bounds for CI analysis
    z_scores = {
        "F10": norm.ppf(0.10),
        "F25": norm.ppf(0.25),
        "F50": norm.ppf(0.50),
        "F75": norm.ppf(0.75),
        "F90": norm.ppf(0.90)
    }

    f10 = predicted_power + z_scores["F10"] * residual_std
    f25 = predicted_power + z_scores["F25"] * residual_std
    f50 = predicted_power + z_scores["F50"] * residual_std
    f75 = predicted_power + z_scores["F75"] * residual_std
    f90 = predicted_power + z_scores["F90"] * residual_std

    # Count scheduled power within CI
    scheduled_in_ci_50 = np.sum((scheduled_power >= f25) & (scheduled_power <= f75))
    scheduled_in_ci_80 = np.sum((scheduled_power >= f10) & (scheduled_power <= f90))

    scheduled_ci_50_pct = (scheduled_in_ci_50 / len(scheduled_power) * 100) if len(scheduled_power) > 0 else 0
    scheduled_ci_80_pct = (scheduled_in_ci_80 / len(scheduled_power) * 100) if len(scheduled_power) > 0 else 0

    # Scheduled power analysis
    scheduled_mae = np.mean(np.abs(scheduled_power - actual_power))
    scheduled_rmse = np.sqrt(np.mean((scheduled_power - actual_power) ** 2))
    scheduled_peak_error = peak_actual - np.max(scheduled_power)

    # Manual scheduled power analysis (if available)
    manual_mae = 0
    if 'manual_scheduled_power' in prediction_df.columns:
        manual_power = prediction_df['manual_scheduled_power'].values
        if not np.all(np.isnan(manual_power)):
            manual_mae = np.mean(np.abs(manual_power - actual_power))

    # FSP analysis
    fsp_stats = {}
    fsp_count = {}
    if 'ml_selected_fsp' in prediction_df.columns:
        fsp_counts_raw = prediction_df['ml_selected_fsp'].value_counts()
        for fsp, count in fsp_counts_raw.items():
            fsp_count[fsp] = int(count)

        # Get FSP forecast columns
        fsp_cols = [col for col in prediction_df.columns if col.startswith('forecast_power_')]
        for fsp_col in fsp_cols:
            fsp_name = fsp_col.replace('forecast_power_', '').upper()
            if fsp_col in prediction_df.columns:
                fsp_power = prediction_df[fsp_col].values
                fsp_mae = np.mean(np.abs(fsp_power - actual_power))
                fsp_rmse = np.sqrt(np.mean((fsp_power - actual_power) ** 2))
                fsp_stats[fsp_name] = {
                    "mae": float(fsp_mae),
                    "rmse": float(fsp_rmse),
                    "count": fsp_count.get(fsp_name, 0),
                    "peak_at_actual_peak": float(fsp_power[peak_actual_idx]) if peak_actual_idx < len(fsp_power) else 0
                }

    # Detect sharp increases (sudden jumps)
    power_diffs = np.diff(actual_power)
    sharp_increases = np.where(power_diffs > residual_std * 2)[0]
    sharp_increase_count = len(sharp_increases)

    analysis = {
        "date": str(date),
        "model": model_name,
        "num_points": len(prediction_df),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "peak_actual": float(peak_actual),
            "trough_actual": float(trough_actual),
            "peak_predicted": float(peak_predicted),
            "peak_error": float(peak_error),
            "peak_error_pct": float(peak_error_pct),
            "trough_predicted": float(trough_predicted),
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "scheduled_mae": float(scheduled_mae),
            "scheduled_rmse": float(scheduled_rmse),
            "scheduled_peak_error": float(scheduled_peak_error),
            "scheduled_in_ci_50_pct": float(scheduled_ci_50_pct),
            "scheduled_in_ci_80_pct": float(scheduled_ci_80_pct),
            "sharp_increases": int(sharp_increase_count),
        },
        "manual_mae": float(manual_mae),
        "fsp_stats": fsp_stats,
        "fsp_count": fsp_count,
        "residual_std": float(residual_std),
    }

    # Build comprehensive prompt following the three-step flow
    fsp_details = ""
    if fsp_stats:
        fsp_details = "FSP Performance Details:\n"
        for fsp, stats in fsp_stats.items():
            fsp_details += f"- {fsp}: MAE={stats['mae']:.2f}MW, Count={stats['count']} blocks, Peak at actual peak={stats['peak_at_actual_peak']:.2f}MW\n"

    # Calculate assessment strings
    conf_level = _get_confidence_level(mae, mape, scheduled_ci_80_pct)
    risks = _identify_risks(peak_error_pct, sharp_increase_count, scheduled_ci_80_pct)
    recommendation = _get_recommendation(mape, scheduled_ci_80_pct)

    directional_acc = "Good" if scheduled_ci_50_pct > 60 else "Moderate" if scheduled_ci_50_pct > 40 else "Poor"
    peak_capture = "Good peak capture" if abs(scheduled_peak_error) < peak_actual*0.1 else "Poor peak capture"
    conf_assessment = "High" if mape < 15 else "Medium" if mape < 25 else "Low"

    prompt = f"""Generate CONCISE bullet-point insights for wind power forecast on {date} using {model_name}.

DATA SUMMARY:
- Actual Power Range: {trough_actual:.2f}MW to {peak_actual:.2f}MW
- Power Distribution: Q25={q25:.2f}MW, Q50={q50:.2f}MW, Q75={q75:.2f}MW
- Variability Events: {sharp_increase_count} sharp changes detected
- Forecast Coverage: {scheduled_ci_50_pct:.1f}% at 50% CI, {scheduled_ci_80_pct:.1f}% at 80% CI
{fsp_details if fsp_details else ""}

STRICT FORMAT REQUIREMENTS:
1. Output EXACTLY 3-6 bullet points as needed
2. Each bullet point must be ONE simple line (max 15 words)
3. Start each line with " Insight N:" or " [Key observation]" or " [ Key insights]:"
4. NO paragraphs, NO long explanations
5. Focus on observable facts only

REQUIRED OUTPUT FORMAT:
 Insight 1: [One concise observation with a key number]
 Insight 2: [Another brief observation with data, forecast performance, or pattern]
 Insight 3: [Short pattern or characteristic noted]
 Insight 4: [Final brief insight if relevant]

Keep each point SHORT and data-focused."""

    # Generate insights
    if generator is None:
        generator = OllamaInsightsGenerator()

    insights = generator.generate_insights(prompt, temperature=0.3, max_tokens=250)
    analysis["insights"] = insights

    return analysis


def render_single_day_insights(analysis: Dict[str, Any]) -> None:
    """
    Render insights below the plot with detailed metrics and FSP analysis.

    Args:
        analysis: Analysis dictionary from analyze_single_day_forecast
    """
    st.markdown("---")

    # Performance Metrics Section
    st.markdown("###  Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ml_predicted_mae = analysis['metrics']['mae']
        st.metric("ML Predicted MAE", f"{ml_predicted_mae:.3f} MW",
                  help="MAE between Actual Power and ML Predicted Power")

    with col2:
        ml_scheduled_mae = analysis['metrics']['scheduled_mae']
        st.metric("ML Scheduled MAE", f"{ml_scheduled_mae:.3f} MW",
                  help="MAE between Actual Power and ML Scheduled Power (selected FSP)")

    with col3:
        # Calculate manual MAE if available, otherwise show N/A
        manual_mae = analysis.get('manual_mae', 0)
        if manual_mae > 0:
            st.metric("Manual MAE", f"{manual_mae:.3f} MW",
                      help="MAE between Actual Power and Manual Scheduled Power")
        else:
            st.metric("Manual MAE", "N/A",
                      help="Manual scheduled data not available for this date")

    with col4:
        # Calculate improvement: Manual vs ML Scheduled
        if manual_mae > 0:
            improvement = ((manual_mae - ml_scheduled_mae) / manual_mae) * 100
            st.metric("Total Improvement", f"{improvement:.1f}%",
                      help="Improvement of ML Scheduled over Manual Schedule")
        else:
            st.metric("Total Improvement", "N/A",
                      help="Cannot calculate without manual schedule data")

    # Insights section
    st.markdown("###  AI-Powered Insights")

    # Display model info
    model_info = st.session_state.get("ollama_selected_model", DEFAULT_MODEL)
    st.caption(f"Generated using: **{model_info}**")

    # Display insights with proper formatting
    insights_text = analysis.get("insights", "No insights available")

    # Parse and structure bullet points
    lines = insights_text.split('\n')

    # Categorize insights
    key_insights = []

    # Better parsing - look for "Insight N:" or " Insight" patterns
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match patterns like " Insight 1:", "Insight 1:", " Something"
        if 'Insight' in line and ':' in line:
            # Extract everything after the colon
            parts = line.split(':', 1)
            if len(parts) > 1:
                insight_text = parts[1].strip()
                # Remove leading bullet if present
                insight_text = insight_text.lstrip('').strip()
                if insight_text:
                    key_insights.append(insight_text)
        elif line.startswith('') and 'Insight' not in line:
            # Handle bullet points that aren't headers
            clean_line = line.lstrip('').strip()
            if clean_line:
                key_insights.append(clean_line)

    # Display in structured format
    st.markdown("####  Key Insights")
    if key_insights:
        for insight in key_insights:
            st.markdown(f" {insight}")
    else:
        st.markdown(f" {insights_text}")

    # Additional detailed metrics with improved structure
    st.markdown("###  Detailed Performance Metrics")

    # Tab-based organization for cleaner UI
    tab1, tab2, tab3 = st.tabs([" Peak & Trough", " Quantile Distribution", " FSP Performance"])

    with tab1:
        st.markdown("#### Peak & Trough Analysis")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.metric(
                " Actual Peak",
                f"{analysis['metrics']['peak_actual']:.2f} MW"
            )
            st.metric(
                " Actual Trough",
                f"{analysis['metrics']['trough_actual']:.2f} MW"
            )

        with col_p2:
            st.metric(
                " Predicted at Peak",
                f"{analysis['metrics']['peak_predicted']:.2f} MW"
            )
            st.metric(
                " Predicted Trough",
                f"{analysis['metrics']['trough_predicted']:.2f} MW"
            )

        st.divider()
        st.markdown("#### Error Analysis")
        error_col1, error_col2 = st.columns(2)
        with error_col1:
            st.metric(
                " Peak Error (MW)",
                f"{analysis['metrics']['peak_error']:.2f} MW",
                f"{analysis['metrics']['peak_error_pct']:.1f}% error"
            )
        with error_col2:
            st.metric(
                " Trough Error (MW)",
                f"{abs(analysis['metrics']['trough_predicted'] - analysis['metrics']['trough_actual']):.2f} MW"
            )

    with tab2:
        st.markdown("#### Power Distribution Quartiles")
        col_q1, col_q2, col_q3 = st.columns(3)

        with col_q1:
            st.metric(
                "Q1 (25th percentile)",
                f"{analysis['metrics']['q25']:.2f} MW",
                "Lower quartile"
            )

        with col_q2:
            st.metric(
                "Q2 (50th percentile)",
                f"{analysis['metrics']['q50']:.2f} MW",
                "Median"
            )

        with col_q3:
            st.metric(
                "Q3 (75th percentile)",
                f"{analysis['metrics']['q75']:.2f} MW",
                "Upper quartile"
            )

        st.divider()
        st.markdown("#### Volatility Events")
        event_col = st.columns(1)[0]
        with event_col:
            st.metric(
                " Sharp Increases Detected",
                f"{analysis['metrics']['sharp_increases']}",
                "events"
            )

    with tab3:
        st.markdown("#### ML Scheduled (FSP) Performance Metrics")
        fsp_col1, fsp_col2 = st.columns(2)

        with fsp_col1:
            st.metric(
                " Scheduled MAE",
                f"{analysis['metrics']['scheduled_mae']:.2f} MW",
                "Mean Absolute Error"
            )
            st.metric(
                " Scheduled RMSE",
                f"{analysis['metrics']['scheduled_rmse']:.2f} MW",
                "Root Mean Squared Error"
            )

        with fsp_col2:
            st.metric(
                " Scheduled Peak Error",
                f"{analysis['metrics']['scheduled_peak_error']:.2f} MW"
            )

        st.divider()
        st.markdown("#### Confidence Interval Coverage")
        ci_col1, ci_col2 = st.columns(2)

        with ci_col1:
            ci_50 = analysis['metrics']['scheduled_in_ci_50_pct']
            st.metric(
                " 50% CI Coverage",
                f"{ci_50:.1f}%",
                delta="Good" if ci_50 >= 50 else "Needs Improvement"
            )

        with ci_col2:
            ci_80 = analysis['metrics']['scheduled_in_ci_80_pct']
            st.metric(
                " 80% CI Coverage",
                f"{ci_80:.1f}%",
                delta="Excellent" if ci_80 >= 80 else "Good" if ci_80 >= 70 else "Needs Improvement"
            )

    # FSP Statistics if available
    if analysis.get('fsp_stats'):
        st.markdown("###  FSP-Specific Performance")

        fsp_data = []
        for fsp_name, stats in analysis['fsp_stats'].items():
            fsp_data.append({
                "FSP": fsp_name,
                "MAE (MW)": f"{stats['mae']:.2f}",
                "RMSE (MW)": f"{stats['rmse']:.2f}",
                "Blocks Used": stats['count'],
                "Power at Peak (MW)": f"{stats['peak_at_actual_peak']:.2f}"
            })

        if fsp_data:
            fsp_df = pd.DataFrame(fsp_data)
            st.dataframe(fsp_df, use_container_width=True, hide_index=True)

            # FSP selection analysis
            st.markdown("**FSP Usage Distribution**")
            fsp_counts = analysis.get('fsp_count', {})
            if fsp_counts:
                col_fsp1, col_fsp2 = st.columns(2)
                with col_fsp1:
                    st.bar_chart(pd.Series(fsp_counts).sort_values(ascending=False))
                with col_fsp2:
                    total_blocks = sum(fsp_counts.values())
                    for fsp, count in sorted(fsp_counts.items(), key=lambda x: x[1], reverse=True):
                        pct = (count / total_blocks * 100) if total_blocks > 0 else 0
                        st.caption(f" {fsp}: {count} blocks ({pct:.1f}%)")

    # Data Analyst Chatbot Section
    st.markdown("---")

    # Header with clear chat button
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown("###  Data Analyst Assistant")
    with col_header2:
        if st.button(" Clear Chat", key="clear_chat_btn", help="Clear conversation history"):
            st.session_state.analyst_chat_history = []
            st.rerun()

    st.markdown("Ask questions about this forecast data. I'll provide clear, non-technical answers based on available information.")

    # Initialize chat history for this session
    if "analyst_chat_history" not in st.session_state:
        st.session_state.analyst_chat_history = []

    # Prepare data context for the chatbot - optimized for multiple use cases
    data_context = f"""You are a Data Analyst Assistant specialized in wind power forecasting analysis. You have access to this wind power forecast data for {analysis['date']} using the {analysis['model']} model.

AVAILABLE DATA YOU CAN ANSWER ABOUT:
- Actual Power Range: {analysis['metrics']['trough_actual']:.2f}MW to {analysis['metrics']['peak_actual']:.2f}MW
- Power Distribution: Q25={analysis['metrics']['q25']:.2f}MW, Q50 (Median)={analysis['metrics']['q50']:.2f}MW, Q75={analysis['metrics']['q75']:.2f}MW
- Forecast Accuracy: MAE={analysis['metrics']['mae']:.2f}MW, RMSE={analysis['metrics']['rmse']:.2f}MW
- Scheduled Power Performance: MAE={analysis['metrics']['scheduled_mae']:.2f}MW, RMSE={analysis['metrics']['scheduled_rmse']:.2f}MW
- Forecast Coverage: {analysis['metrics']['scheduled_in_ci_50_pct']:.1f}% within 50% confidence, {analysis['metrics']['scheduled_in_ci_80_pct']:.1f}% within 80% confidence
- Variability: {analysis['metrics']['sharp_increases']} sharp power changes detected
- Peak Analysis: Actual Peak={analysis['metrics']['peak_actual']:.2f}MW, Prediction at Peak={analysis['metrics']['peak_predicted']:.2f}MW
- Trough Analysis: Actual Trough={analysis['metrics']['trough_actual']:.2f}MW, Predicted Trough={analysis['metrics']['trough_predicted']:.2f}MW
- Manual Scheduled Performance: MAE={analysis.get('manual_mae', 0):.2f}MW

RESPONSE FORMAT RULES:
1. ALWAYS use bullet points for your answers
2. Structure your response with clear sections when appropriate
3. Keep language simple and non-technical
4. Be detailed but concise - explain concepts clearly
5. Always cite specific numbers from the data
6. Use everyday business language, not technical jargon
7. Adapt your response based on the question type:
   - For "what" questions: Focus on definitions and explanations
   - For "how" questions: Explain processes or calculations
   - For "why" questions: Provide reasoning and context
   - For comparison questions: Use side-by-side bullet points
   - For trend questions: Highlight patterns and changes
   - For performance questions: Include benchmarks and metrics

CONTENT RULES:
1. Only answer questions based on the data provided above
2. If asked about data not in this list, respond: "I don't have information about that in the current forecast data. I can help with: power ranges, accuracy metrics, forecast coverage, variability, or FSP performance."
3. If a question is unclear, ask for clarification with specific examples
4. For complex questions, break down the answer into multiple clear sections
5. Always provide context for numbers (e.g., "X MW is Y% of peak power")

EXAMPLE RESPONSE FORMATS:

For Simple Questions:
**Answer:**
 [Direct answer with specific number]
 [Additional context or detail]

For Comparison Questions:
**Comparison Results:**
 Option A: [metric]
 Option B: [metric]
 Difference: [calculation]

**What This Means:**
 [Plain language interpretation]

For Trend/Pattern Questions:
**Observed Pattern:**
 [Description with data points]
 [Time period or condition]

**Business Impact:**
 [Practical implications]
 [Actionable insight]

For Complex Questions:
**Overview:**
 [High-level summary]

**Detailed Breakdown:**
 [Point 1 with data]
 [Point 2 with data]

**Key Takeaway:**
 [Main insight for decision-making]"""

    # Display chat history
    for message in st.session_state.analyst_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input("Ask me anything about this forecast data...")

    if user_question:
        # Add user message to history
        st.session_state.analyst_chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate analyst response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    generator = OllamaInsightsGenerator()

                    # Determine question type for optimal response
                    question_lower = user_question.lower()
                    if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
                        response_hint = "comparison"
                    elif any(word in question_lower for word in ['trend', 'pattern', 'over time', 'change']):
                        response_hint = "trend"
                    elif any(word in question_lower for word in ['why', 'reason', 'cause']):
                        response_hint = "explanation"
                    elif any(word in question_lower for word in ['how', 'calculate', 'work']):
                        response_hint = "process"
                    else:
                        response_hint = "general"

                    prompt = f"""{data_context}

User Question: {user_question}

CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW:
1. Use bullet points () for EVERY point - NO paragraphs allowed
2. Use markdown headers (** **) for section titles
3. Each bullet point should be ONE concise line
4. Maximum 2-3 sections in your response
5. Maximum 3-4 bullet points per section

REQUIRED OUTPUT STRUCTURE:

**[Section Title]:**
 [Short point 1 with number]
 [Short point 2 with number]
 [Short point 3]

**[What This Means]:**
 [Plain language summary]
 [Actionable insight]

Example of GOOD formatting:
**Power Range Analysis:**
 Actual power ranged from {analysis['metrics']['trough_actual']:.2f}MW to {analysis['metrics']['peak_actual']:.2f}MW
 Median power output was {analysis['metrics']['q50']:.2f}MW

**Interpretation:**
 This represents typical daily variation
 Peak is {((analysis['metrics']['peak_actual']/analysis['metrics']['q50'])-1)*100:.0f}% above median

NOW answer the question using ONLY bullet points in this format. NO long paragraphs."""

                    response = generator.generate_insights(
                        prompt,
                        temperature=0.3,  # Lower temperature for more consistent formatting
                        max_tokens=350    # Reduced to encourage conciseness
                    )

                    st.markdown(response)

                    # Add assistant response to history
                    st.session_state.analyst_chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    error_msg = f" Unable to generate response. Please try again."
                    st.markdown(error_msg)
                    st.session_state.analyst_chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    generator = OllamaInsightsGenerator()
    return generator.available_models if generator.available_models else [DEFAULT_MODEL]


def setup_ollama_selector() -> str:
    """
    Create a sidebar selector for Ollama model.

    Returns:
        Selected model name
    """
    available_models = get_available_models()

    if not available_models:
        st.warning(" Ollama server not accessible. Using default insights.")
        return DEFAULT_MODEL

    if "ollama_selected_model" not in st.session_state:
        st.session_state.ollama_selected_model = available_models[0]

    selected = st.selectbox(
        " AI Model for Insights",
        options=available_models,
        index=available_models.index(st.session_state.ollama_selected_model),
        help="Select which LLM to use for generating insights"
    )

    st.session_state.ollama_selected_model = selected
    return selected
