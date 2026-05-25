# GKFS Operational Dashboard User Documentation

This document explains every part of the **GKFS Operational Dashboard**: what each section shows, how each **metric** is calculated, and how each **plot** is built. Use it to interpret the dashboard and understand the numbers and charts.

---

## 1. Dashboard Overview

The Operational Dashboard is a **Streamlit** app that:

- Loads **18 months** of plant data (from `data/processed/parquet/` or `data/processed/`).
- Splits data **temporally** into Train (70%), Validation (15%), and Test (15%).
- Trains or loads a **RidgeLightGBM ensemble** model to predict power **6 blocks (1.5 hours) ahead**.
- Generates **predictions** and **ML scheduling** by choosing, for each 15minute block, which FSP (Forecast Service Provider) forecast to use so that scheduled power is closest to the ML prediction.
- Compares **ML-scheduled** vs **manual-scheduled** power and shows **DSM (Deviation Settlement Mechanism) penalty** estimates.

**Workflow:** Select Plant Load Data Train or Load Model Generate Predictions View three tabs (Quantile Forecasts, Block-wise Analysis, Test Set Aggregates) and DSM Penalty Comparison at the bottom.

---

## 2. Sidebar

| Element | Meaning |
|--------|--------|
| **Data Loaded** | Whether 18month plant data has been loaded and split into train/val/test. |
| **Model Loaded** | Whether a saved RidgeLightGBM ensemble was loaded from disk. |
| **Model Trained** | Whether the model was trained in this session. |
| **Predictions Generated** | Whether predictions and ML scheduling have been run. |
| **Ridge Weight** | Ensemble weight for Ridge (default **40%**); LightGBM gets the remaining **60%**. |
| **Features** | Number of input features used by the model. |

---

## 3. Step 0: Select Plant

- **Select Plant**: Dropdown of plants discovered from `*_dataset.parquet` files (e.g. Plant Alpha, Plant Beta, Plant Gamma, Sample Plant, Plant Delta).
- **Auto-load model from model_savesss**: If checked, when you change the plant, the app tries to load data and then load the model from the `model_savesss` folder for that plant.

---

## 4. Step 1: Load Data

After clicking **Load Data**:

- Data is loaded from `data/processed/parquet/<plant>_dataset.parquet` (or root `data/processed/`), pivoted for FSPs, limited to the last **18 months**, and cleaned (actual power and at least one FSP non-null).
- **Target**: `actual_power` 6 blocks ahead (`target_horizon = actual_power.shift(-6)`).
- **Split**: 70% train, 15% validation, 15% test (temporal, no shuffle).

### 4.1 Data Summary Metrics

| Metric | Calculation |
|--------|-------------|
| **Total Rows** | Number of rows in the pivoted dataframe after the 18month filter and cleaning (rows used for train+val+test before split). |
| **Train Set** | Number of rows in the **training** set (first 70% temporally). |
| **Validation Set** | Number of rows in the **validation** set (next 15%). |
| **Test Set** | Number of rows in the **test** set (last 15%). |
| **Input Features** | Count of feature columns used for the model (from feature engineering: time features, rolling stats, encoded categoricals, etc.), excluding target and excluded patterns. |

---

## 5. Step 2: Train or Load Model

- **Train Model**: Trains Ridge + LightGBM, then combines with **Ridge weight = 0.4** and **LightGBM weight = 0.6**. Saves to `outputs/saved_models/<PLANT>/ridge_lightgbm_ensemble/v1/`.
- **Load Trained Model**: Loads from `outputs/saved_models/` or, if the model_savesss option is used, from `model_savesss/`.

### 5.1 Model Summary Metrics (after load or train)

| Metric | Calculation |
|--------|-------------|
| **Model Type** | Fixed: "Ridge-LightGBM Ensemble". |
| **Ridge Weight** | Fixed: 40% (0.4). |
| **Features** | Number of feature columns the model was trained with (from `feature_columns.json` or session state). |
| **Test Accuracy (R2)** | R2 between **test set** `target_horizon` (actual power 6 blocks ahead) and **ensemble predictions** on test. |

### 5.2 Model Training Statistics (from saved config)

| Metric | Calculation |
|--------|-------------|
| **Train Size** | Number of training rows. |
| **Val Size** | Number of validation rows. |
| **Test Size** | Number of test rows. |
| **Val Accuracy (R2)** | R2(validation actual, ensemble prediction on validation). |
| **Val RMSE** | (mean((y_val pred_val)2)), in MW. |
| **Test RMSE** | (mean((y_test pred_test)2)), in MW. |

**R2 (Accuracy) formula (used everywhere in the app):**

- `accuracy = r2_score(y_true, y_pred)` (sklearn), after dropping rows where either series is NaN.
- R2 = 1 (SS_res / SS_tot); higher is better; 1 = perfect fit.

---

## 6. Step 3: Generate Predictions

When you click **Show Predictions** (or predictions are already generated):

1. **Model** predicts **6blockahead power** for each test row (ensemble output `ml_predicted_power`).
2. **FSP selection**: For each 15minute block, the FSP whose **forecast is closest to the ML prediction** (in absolute MW) is chosen; then a **minimum 6block constraint per day** is applied so the chosen FSP does not change more often than every 6 blocks within the same day.
3. **ML scheduled power** = forecast value of the selected FSP for that block.
4. **Selection confidence** (see below) and error columns are computed.

So:

- **ml_predicted_power** = ensemble prediction (MW) for that block (horizon +6).
- **ml_scheduled_power** = FSP forecast (MW) of the FSP selected for that block.
- **ml_selected_fsp** = name of that FSP.

**Selection confidence (per block):**

- For the block, errors are computed: `error_fsp = |FSP_forecast ML_prediction|` for each FSP.
- Best FSP = argmin of these errors.
- `min_err = min(errors)`, `max_err = max(errors)`.
- **Confidence** = `(max_err min_err) / (max_err + 1e-8)` if `max_err > 0`, else `0.5`.
  So confidence is higher when one FSP is clearly closer to the prediction than the others.

**Note:** In the prediction dataframe, **actual_power** is the **target_horizon** (actual power 6 blocks ahead at that timestamp), not same-block actual. Manual schedule is from **schedule_power** (manual scheduled power).

---

## 7. Tab 1: Quantile Forecasts

### 7.1 Daily Quantile Ribbon (Blocks 196)

**What it is:** One day of data: 96 blocks (15min each). Shows uncertainty bands around the ML prediction and where actual, ML scheduled, and optional manual scheduled lie.

**Quantile bands (residual-based, normal assumption):**

- Residuals: `residual = actual_power ml_predicted_power` (over the prediction dataframe).
- `residual_std = std(residuals)` (with a small minimum to avoid division by zero).
- For quantiles F10, F25, F50, F75, F90 (0.10, 0.25, 0.50, 0.75, 0.90):
  - `z = norm.ppf(quantile)`
  - **Fxx** = `ml_predicted_power + z * residual_std`
- **F50** = median forecast = ML predicted (dashed line).
- **80% CI** = band between F10 and F90 (light fill).
- **50% CI** = band between F25 and F75 (darker fill).

**Percentiles (for hover):**

- **Actual percentile** = `norm.cdf((actual_power ml_predicted_power) / residual_std) * 100`, clipped to [0, 100].
- **ML scheduled percentile** = same formula with `ml_scheduled_power` instead of actual.

**Plot elements:**

- F10F90 bands, F50 line, Actual power line, ML Scheduled (markers colored by selected FSP), optional Manual Scheduled.
- Optional Show all FSP forecast lines overlays each FSP forecast with its percentile vs the band.

**Date:** Chosen via Select Date for Daily Ribbon; only that days 96 blocks are plotted.

---

### 7.2 Forecast Heatmap (optional, after Show Forecast Heatmap)

- **Rows**: FSPs. **Columns**: Blocks B01B96.
- **Cell value**: Forecasted power (MW) for that FSP in that block.
- **Colors**: Gradient by MW (e.g. low high).
- **Highlight**: Squares show which FSP was **scheduled** (selected by ML) in each block.

So you see all FSP forecasts and which one was chosen per block.

---

### 7.3 FSP Selection Waterfall (All 96 Blocks)

- **Stacked bars** per block: each segment is the **ML scheduled power** for the FSP that was selected in that block (other FSPs show 0 for that block).
- **Lines**: Actual power (black, dotted); ML scheduled (green, dotted).

So you see block-by-block how much each FSP contributed to the schedule and how actual and ML scheduled compare.

---

### 7.4 Metrics Below the Waterfall (for the selected day)

| Metric | Calculation |
|--------|-------------|
| **FSPs Used** | Number of distinct FSPs that were selected in at least one block that day. |
| **Total Scheduled** | Sum of `ml_scheduled_power` over the 96 blocks (MW). |
| **Daily Accuracy (R2)** | R2(actual_power, ml_scheduled_power) over the 96 blocks of the selected day. |
| **Avg Confidence** | Mean of `selection_confidence` over the 96 blocks. |

---

### 7.5 Overall Day Accuracy (same day)

| Metric | Calculation |
|--------|-------------|
| **ML Scheduled Accuracy (R2)** | R2(actual_power, ml_scheduled_power) over the selected days 96 blocks. |
| **Manual Scheduled Accuracy (R2)** | R2(actual_power, manual_scheduled_power) over the same 96 blocks. |
| **Improvement % (Error Reduction)** | `(manual_MAE ml_MAE) / manual_MAE * 100`, where MAE = mean absolute error (actual vs scheduled). Positive % = ML has lower error than manual for that day. |

---

### 7.6 Block-wise Accuracy & Improvement Table

- **Block**: 196.
- **Actual (MW)**: Actual power (target_horizon) for that block on the selected day.
- **ML Scheduled (MW)** / **Manual Scheduled (MW)**: Scheduled power for that block.
- **ML/Manual Scheduled Accuracy (R2)**: R2 for **that block number** computed over **all test-set rows** that fall in that block (so many days), not just the selected day.
- **ML Scheduled Delta (MW)** = Actual ML Scheduled for that block (that day).
- **Manual Delta (MW)** = Actual Manual Scheduled for that block (that day).
- **Improvement % (vs Manual)** = For **that block on that day**: `(|Manual Delta| |ML Delta|) / |Manual Delta| * 100`. Positive means ML error is smaller. If ML Scheduled = Manual Scheduled for the block, improvement is 0%.

---

### 7.7 FSP Selection Distribution (pie chart)

- **Data**: Count of blocks (across **all test data**) where each FSP was selected as `ml_selected_fsp`.
- **Chart**: Pie of these counts (percent + label). So you see the overall share of blocks each FSP gets.

---

### 7.8 Block-wise Power Comparison (line plot)

- **X**: Block 196. **Y**: Power (MW).
- **Lines**: Actual, Manual Scheduled (if present), ML Scheduled, and optionally each FSP forecast (toggle in legend).
  So you can compare actual vs manual vs ML schedule vs individual FSPs across the day.

---

## 8. Tab 2: Block-wise Analysis

This tab uses the **full test set** and breaks results by **block number** (196).

### 8.1 Block-wise Metrics (how theyre computed)

For each block number (e.g. 24 = 06:0006:15):

- All test rows with that block number are grouped.
- **Accuracy (R2)** for ML scheduled = R2(actual_power, ml_scheduled_power) over that group.
- **Accuracy (R2)** for Manual scheduled = R2(actual_power, manual_scheduled_power) over that group.
- **Improvement %** = `(accuracy_ml accuracy_manual) / |accuracy_manual| * 100` (when denominator = 0).
- **Mean delta** = mean(actual scheduled) over that blocks group (bias).

### 8.2 Plots

1. **Accuracy (R2) by Time Block**
   - X: Block 196. Y: R2.
   - Two lines: ML Scheduled Accuracy (R2), Manual Scheduled Accuracy (R2).
   - Lets you see which time-of-day blocks are easier/harder for ML vs manual.

2. **Improvement % by Time Block**
   - X: Block 196. Y: Improvement % (ML vs Manual).
   - Filled line; baseline at 0. Positive = ML better for that block on average.

### 8.3 Overall Day Accuracy (test set)

| Metric | Calculation |
|--------|-------------|
| **Overall ML Scheduled Accuracy (R2)** | R2(actual_power, ml_scheduled_power) over the **entire test set**. |
| **Overall Manual Scheduled Accuracy (R2)** | R2(actual_power, manual_scheduled_power) over the **entire test set**. |
| **Overall Improvement** | `(overall_ml_R2 overall_manual_R2) / |overall_manual_R2| * 100`. |

### 8.4 Block-wise Summary Table

- **Avg/Max/Min** of ML and Manual R2 across blocks.
- **Avg/Max/Min** of Improvement % across blocks.
- **Avg/Max/Min** of mean delta (MW) across blocks.

All computed from the per-block metrics above.

---

## 9. Tab 3: Test Set Aggregates

Everything here is over the **full test set** (all blocks, all days).

### 9.1 Aggregate Power Summary Table

| Row | Value (MW) | Percentage (%) | Difference from Actual (MW) |
|-----|------------|----------------|-----------------------------|
| Total Actual Power | Sum(actual_power) | 100 | |
| Total ML-Predicted Power | Sum(ml_predicted_power) | (sum_ml_pred / sum_actual)*100 | sum_ml_pred sum_actual |
| Total ML-Scheduled Power | Sum(ml_scheduled_power) | (sum_ml_sched / sum_actual)*100 | sum_ml_sched sum_actual |
| Total Manual-Scheduled Power | Sum(manual_scheduled_power) | (sum_manual / sum_actual)*100 | sum_manual sum_actual |

### 9.2 Performance Metrics (R2 and comparison)

| Metric | Calculation |
|--------|-------------|
| **ML Scheduled Accuracy (R2)** | R2(actual_power, ml_scheduled_power) on full test set. |
| **Manual Scheduled Accuracy (R2)** | R2(actual_power, manual_scheduled_power) on full test set. |
| **ML Improvement** | `(R2_ml R2_manual) / |R2_manual| * 100`. |
| **ML Better Cases** | Percentage of **rows** where `ml_scheduled_error < manual_error` (absolute errors). |

### 9.3 Power Comparison Chart

- Bar chart: **Actual**, **ML Predicted**, **ML Scheduled**, **Manual Scheduled** each bar = total MW (sum over test set).

### 9.4 Error Distribution Comparison

- **Box plots** of:
  - **ML Scheduled Error** = |actual_power  ml_scheduled_power| (per row).
  - **Manual Error** = |actual_power  manual_scheduled_power| (per row).
  So you compare the distribution of absolute errors for ML vs manual.

---

## 10. DSM Penalty Comparison (bottom of dashboard)

This section uses **deviation band config** from `deviation_band_configs.json` for the selected plant (by sscode). It compares **estimated DSM penalty** for one **selected date** (96 blocks).

### 10.1 Deviation and penalty (per 15min block)

- **Deviation %** = `(actual scheduled) / scheduled * 100` (when scheduled > 0; else 0).
- **Band**: Config has bands with `from`, `to`, `penalty` (Rs per unit). The band that contains the deviation % is chosen.
- **Block duration** = 0.25 h (15 min).
- **Penalty (Rs) per block** = `band_penalty_rate * |actual scheduled| (MW) * 0.25`.
  So penalty is proportional to absolute MW deviation and the band rate.

**Total penalty** = sum of these block penalties over the 96 blocks of the selected day.

### 10.2 Metrics

| Metric | Calculation |
|--------|-------------|
| **Estimated Penalty (ML Scheduled)** | Sum of block penalties when **scheduled = ml_scheduled_power**. |
| **Estimated Penalty (Manual Scheduled)** | Sum of block penalties when **scheduled = manual_scheduled_power** (if available). |
| **Penalty Reduction (Rs)** | Manual total penalty ML total penalty (for that day). |
| **Penalty Reduction (%)** | `(Manual_penalty ML_penalty) / Manual_penalty * 100` (when Manual > 0). |

### 10.3 Monthly & Annual Penalty (same period as selected date)

Monthly and annual totals use the **same penalty formula** as the daily section, summed over days in the test set.

**Computation:**

- For **every day** in the prediction set, daily penalty (Rs) is computed the same way as for the selected date: per-block penalty = band rate |actual scheduled| (MW) 0.25, then summed over that days blocks.
- **Monthly Penalty (ML)** = sum of daily ML penalties for all dates in the **same month and year** as the selected date (e.g. all days in February 2026 if the selected date is in Feb 2026).
- **Annual Penalty (ML)** = sum of daily ML penalties for all dates in the **same year** as the selected date (e.g. all days in 2026 in the test set).
- **Monthly/Annual Penalty (Manual)** = same sums using manual scheduled power (only for days that have manual schedule data).
- **Monthly Reduction (Rs)** = Monthly Penalty (Manual) Monthly Penalty (ML).
- **Monthly Reduction (%)** = (Monthly Manual Monthly ML) / Monthly Manual 100 (when Manual > 0).
- **Annual Reduction (Rs)** = Annual Penalty (Manual) Annual Penalty (ML).
- **Annual Reduction (%)** = (Annual Manual Annual ML) / Annual Manual 100 (when Manual > 0).

**Note:** Monthly and annual figures are based only on **test set** dates (no extrapolation to full calendar month/year).

### 10.4 Within band (out of 96 points)

- **ML scheduled within band** = Percentage of the 96 blocks where **ML** schedule has **zero penalty** (deviation falls in the no-penalty band).
  Formula: `(number of blocks with penalty_ml  1e-9) / 96 * 100`.
- **Manual scheduled within band** = Same for **manual** schedule: `(number of blocks with penalty_manual 1e-9) / 96 * 100`.

### 10.5 Bar chart

- If manual schedule exists and either penalty is > 0: grouped bar of **ML Scheduled** total penalty vs **Manual Scheduled** total penalty (Rs) for the selected date.

---

## 11. Summary of Key Formulas

| Term | Formula |
|------|--------|
| **R2 (Accuracy)** | `r2_score(y_true, y_pred)` after dropping NaN; R2 = 1 SS_res/SS_tot. |
| **RMSE** | (mean((y_true y_pred)2)), in MW. |
| **MAE** | mean(\|y_true y_pred\|). |
| **Improvement % (error reduction)** | `(manual_MAE ml_MAE) / manual_MAE * 100`. |
| **Improvement % (R2)** | `(R2_ml R2_manual) / |R2_manual| * 100`. |
| **Selection confidence** | `(max_err min_err) / (max_err + 1e-8)` where errors = \|FSP_forecast ML_pred\|. |
| **Quantile Fxx** | `ml_predicted_power + norm.ppf(xx/100) * residual_std`. |
| **DSM penalty (per block)** | Band rate (Rs/unit) \|actual scheduled\| (MW) 0.25 h. |

---

## 12. File and Config References

- **Data**: `data/processed/parquet/<plant>_dataset.parquet` or `data/processed/<plant>_dataset.parquet`.
- **Models**: `outputs/saved_models/<PLANT>/ridge_lightgbm_ensemble/v1/` or `model_savesss/`.
- **DSM bands**: `deviation_band_configs.json` (plant identified by sscode, e.g. SAMPLE_PSS).
- **Constants**: DATA_MONTHS=18, TRAIN=0.70, VAL=0.15, TEST=0.15, Ridge weight=0.4, PREDICTION_HORIZON=6 blocks, BLOCK_HOURS=0.25.

If you need the exact code location for any metric or plot, it lives in `operational_app.py` and (for FSP selection and prediction dataframe) in `app/pages/predictions_viz.py` (`select_best_fsp_by_prediction`, `create_prediction_dataframe`).
