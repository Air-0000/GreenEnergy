# GreenEnergy — Multi-Energy Complementary Dispatch System

**GreenEnergy** is a comprehensive simulation and optimization system for coordinating **green electricity**, **hydrogen production**, and **combined heat and power (CHP)** in a multi-energy complementary microgrid. It combines LSTM-based time-series forecasting with a rule-based dispatch engine to maximize renewable energy utilization, minimize curtailment, and optimize economic and environmental outcomes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
  - [Data Loader](#1-data-loader)
  - [LSTM Predictor](#2-lstm-predictor)
  - [Dispatch Engine](#3-dispatch-engine)
  - [LLM Analyst](#4-llm-analyst)
  - [System Evaluator](#5-system-evaluator)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Technologies](#technologies)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Modern energy systems face a critical challenge: how to efficiently integrate intermittent renewable sources (wind and solar) with flexible loads and storage. **GreenEnergy** tackles this by simulating a multi-energy system that:

1. **Forecasts** wind and solar power generation using LSTM neural networks trained on real European power data.
2. **Dispatches** green electricity according to a priority rule: **first to electrolytic hydrogen production, surplus to CHP**, with grid export as the final fallback.
3. **Evaluates** system performance across energy efficiency, economic benefit, and carbon reduction metrics.
4. **Analyzes** dispatch strategies using optional LLM integration (Claude / GPT) for intelligent commentary and suggestions.

The system was originally developed as a course project for an Artificial Intelligence fundamentals course.

---

## Features

- **LSTM Time-Series Forecasting** — Separate LSTM models (2-layer, 64 hidden units) for wind and solar power prediction with early stopping and dropout regularization.
- **Rule-Based Dispatch Engine** — Prioritizes electrolytic hydrogen production, routes surplus to combined heat and power, and manages hydrogen storage as a buffer.
- **Peak/Off-Peak Pricing** — Automatically adjusts dispatch economics based on time-of-use electricity prices.
- **Carbon Accounting** — Computes CO₂ emission savings relative to conventional thermal generation.
- **LLM Integration** — Optionally uses Claude or GPT APIs to produce natural-language strategy analysis and improvement recommendations. Falls back to a built-in local analysis when no API key is available.
- **Visualization** — Generates comprehensive plots: power forecast timelines, dispatch allocation stack charts, hydrogen storage levels, hourly economic benefit, and carbon reduction.
- **Structured Output** — Saves results as JSON, Markdown reports, and PNG figures for downstream consumption.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   GreenEnergy System                              │
│                                                                  │
│  🌬️  Wind Forecast (LSTM) ──┐                                   │
│  ☀️  Solar Forecast (LSTM) ──┼──▶ 🔋 Electrolyzer (H₂)          │
│                              │         │                         │
│                              │         ▼                         │
│                              │    ⚡ CHP (Heat & Power)            │
│                              │         │                         │
│                              └──▶ 🔌 Grid Export / Import         │
│                                       │                         │
│                                       ▼                         │
│                              💧 H₂ Storage Buffer                 │
│                                       │                         │
│                                       ▼                         │
│                              📊 Evaluator                        │
│                              └──▶ 📈 Reports & Plots             │
│                              └──▶ 🤖 LLM Analysis (optional)     │
└──────────────────────────────────────────────────────────────────┘
```

### Dispatch Priority Rules

1. Green electricity is used **first** for electrolytic hydrogen production.
2. Any remaining power is directed to **Combined Heat and Power (CHP)**.
3. Any excess beyond both capacities is **exported to the grid**.
4. The hydrogen storage tank acts as a buffer; overflow beyond capacity is recorded.
5. Time-of-use pricing (peak 08:00–20:00, off-peak otherwise) influences economic calculations.

---

## Project Structure

```
GreenEnergy/
├── src/                        # Source code
│   ├── main.py                 # Main entry point: orchestrates the full pipeline
│   ├── data_loader.py          # OPSD data loading, preprocessing, and feature engineering
│   ├── lstm_model.py           # LSTM model definition and training/prediction wrapper
│   ├── dispatcher.py           # Rule-based dispatch engine
│   ├── llm_analyst.py          # LLM-based strategy analysis (Claude / GPT / fallback)
│   └── evaluator.py            # System performance evaluation and visualization
├── data/                       # Data directory (OPSD time-series CSV expected here)
│   └── opsd_time_series.csv    # Open Power System Data (hourly, ~124 MB)
│   └── SOLETE/                 # Optional SOLETE Danish wind/solar dataset
├── results/                    # Output directory (generated at runtime)
│   ├── wind_lstm.pth           # Trained wind LSTM model weights
│   ├── solar_lstm.pth          # Trained solar LSTM model weights
│   ├── dispatch_overview.png   # Combined power forecast + allocation + H₂ storage plot
│   ├── economics_carbon.png    # Economic benefit and carbon reduction plot
│   ├── results.json            # Structured evaluation results
│   ├── llm_analysis.txt        # LLM-generated analysis (if configured)
│   └── analysis_report.md      # Full analysis report
├── models/                     # Additional model storage (optional)
├── reports/                    # Additional reports (optional)
└── requirements.txt            # Python dependencies
```

---

## Data Sources

### Primary: Open Power System Data (OPSD)

- **Source**: [Open Power System Data](https://open-power-system-data.org/) — Time Series
- **Range**: 2014-12-31 to 2020-09-30 (hourly resolution)
- **Coverage**: 32 European countries; wind, solar, load, and price data
- **File**: `data/opsd_time_series.csv` (~124 MB, not included in repo; must be downloaded separately)

### Secondary: SOLETE Dataset (optional)

Danish wind and solar data from the SOLETE project can be placed in `data/SOLETE/`.

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Air-0000/GreenEnergy.git
cd GreenEnergy

# Install dependencies
pip install -r requirements.txt

# Download the OPSD dataset (required for training)
# Place it at: data/opsd_time_series.csv
# You can download from: https://data.open-power-system-data.org/time_series/
```

### Dependencies

| Package       | Version   | Purpose                        |
|---------------|-----------|--------------------------------|
| torch         | ≥ 2.0.0   | LSTM model (PyTorch)           |
| numpy         | ≥ 1.24.0  | Numerical computing            |
| pandas        | ≥ 2.0.0   | Data manipulation              |
| scikit-learn  | ≥ 1.3.0   | Scaling, metrics               |
| matplotlib    | ≥ 3.7.0   | Plotting                       |
| seaborn       | ≥ 0.12.0  | Statistical visualizations     |
| anthropic     | ≥ 0.25.0  | Claude API (optional)          |
| openai        | ≥ 1.0.0   | OpenAI API (optional)          |

---

## Quick Start

### Run the full pipeline

```bash
python src/main.py
```

This will:

1. Load and preprocess the OPSD dataset.
2. Train LSTM models for wind and solar power prediction.
3. Generate 168-hour (one week) forecasts.
4. Execute the rule-based dispatch engine.
5. Evaluate system performance and produce visualizations.
6. Optionally call an LLM for strategy analysis (if `OPENAI_API_KEY` or `CLAUDE_API_KEY` is set).
7. Save all outputs to the `results/` directory.

### Run individual modules

```bash
# Test the data loader
python src/data_loader.py

# Test the LSTM predictor
python src/lstm_model.py

# Test the dispatch engine
python src/dispatcher.py

# Test the LLM analyst (requires API key)
python src/llm_analyst.py

# Test the system evaluator
python src/evaluator.py
```

---

## Module Reference

### 1. Data Loader

`src/data_loader.py` — Loads, cleans, and prepares OPSD time-series data.

```python
from data_loader import DataLoader

loader = DataLoader('data/opsd_time_series.csv')
loader.load_and_preprocess()          # Load CSV, extract wind/solar/load/price columns
loader.explore_data()                 # Print summary statistics

# Create sliding-window sequences
X, y = loader.create_sequences(loader.wind_data, lookback=24)
# X shape: (samples, 24, 1) — past 24 hours
# y shape: (samples,)       — next hour value
```

- Automatically selects German data (most complete) by default.
- Extracts columns matching wind/solar/load/price patterns.
- Creates time-based features (hour, day, month, cyclical encodings) and lagged features.

### 2. LSTM Predictor

`src/lstm_model.py` — Defines and trains an LSTM neural network for univariate time-series forecasting.

```python
from lstm_model import LSTMPredictor

predictor = LSTMPredictor(
    name='Wind',
    input_dim=1,
    hidden_dim=64,
    num_layers=2,
    lookback=24
)

# Train with early stopping (patience=10)
predictor.train(X, y, epochs=50)

# Predict and evaluate
predictions = predictor.predict(X_test)
results = predictor.evaluate(X_test, y_test)
# {'rmse': ..., 'mae': ..., 'mape': ...}

# Save / load model weights
predictor.save('results/wind_lstm.pth')
predictor.load('results/wind_lstm.pth')
```

**Model Architecture**: 2-layer LSTM → FC(64→32) → FC(32→1) with ReLU activation and dropout (0.2).

### 3. Dispatch Engine

`src/dispatcher.py` — Rule-based energy dispatch optimization.

```python
from dispatcher import DispatchEngine

config = {
    'electrolyzer_capacity': 500,        # kW
    'electrolyzer_efficiency': 0.7,      # kWh/kg H₂
    'chp_capacity': 300,                 # kW
    'chp_heat_ratio': 0.4,               # thermal:electric ratio
    'h2_storage_capacity': 1000,         # kg
    'electricity_price_low': 0.3,        # ¥/kWh (off-peak)
    'electricity_price_high': 0.8,       # ¥/kWh (peak)
}

dispatcher = DispatchEngine(config)

# Single-step dispatch
result = dispatcher.dispatch(wind_power=300, solar_power=200, timestamp=...)

# Batch simulation
results = dispatcher.run_simulation(power_forecast_df)

# Summarize results
summary = dispatcher.summarize_results(results)
```

**Key dispatch outputs per time step**:

| Output               | Unit    | Description                        |
|----------------------|---------|------------------------------------|
| electrolyzer_power   | kW      | Power to hydrogen production       |
| h2_produced          | kg/h    | Hydrogen produced                  |
| h2_storage_level     | kg      | Current hydrogen storage level     |
| chp_power            | kW      | Power to CHP                       |
| chp_heat             | kW      | Thermal output from CHP            |
| grid_import / export | kW      | Grid power exchange                |
| net_economic_benefit | ¥       | Net economic benefit               |
| carbon_savings       | kg CO₂  | Estimated CO₂ reduction            |

### 4. LLM Analyst

`src/llm_analyst.py` — Optional large language model integration for strategy interpretation.

```python
from llm_analyst import LLMAnalyst

# Using environment variable (OPENAI_API_KEY or CLAUDE_API_KEY)
analyst = LLMAnalyst()

# Analyze dispatch scenario
analysis = analyst.analyze_scenario(scenario_summary)
```

- Supports **Claude** (via Anthropic SDK) and **GPT** (via OpenAI API).
- Falls back to a built-in **local analysis** if no API key is available.
- Analysis dimensions: strategy evaluation, efficiency improvement suggestions, and upgrade roadmap.

### 5. System Evaluator

`src/evaluator.py` — Comprehensive system performance assessment and visualization.

```python
from evaluator import SystemEvaluator

evaluator = SystemEvaluator('results')
metrics = evaluator.evaluate(dispatch_results, wind_forecast, solar_forecast)
```

**Evaluation metrics**:

| Category      | Metric               | Description                         |
|---------------|----------------------|-------------------------------------|
| Efficiency    | overall_efficiency   | System-level energy efficiency (%)  |
| Efficiency    | self_use_rate        | Green power self-consumption rate   |
| Efficiency    | curtailment_rate     | Curtailment rate (%)                |
| Production    | h2_production        | Total hydrogen produced (kg)        |
| Economic      | net_economic_benefit | Net economic benefit (¥)            |
| Economic      | h2_revenue           | Hydrogen sales revenue (¥)          |
| Environmental | carbon_savings       | CO₂ emission savings (kg CO₂)       |

---

## Configuration

System parameters can be customized by editing the `self.config` dictionary in `src/main.py`:

```python
self.config = {
    'electrolyzer_capacity': 500,          # kW
    'electrolyzer_efficiency': 0.7,        # kWh/kg H₂
    'chp_capacity': 300,                   # kW
    'chp_heat_ratio': 0.4,                 # thermal:electric ratio
    'chp_efficiency': 0.85,                # CHP electric efficiency
    'h2_storage_capacity': 1000,           # kg
    'h2_storage_level': 500,               # initial storage level (kg)
    'electricity_price_low': 0.3,          # ¥/kWh (off-peak)
    'electricity_price_high': 0.8,         # ¥/kWh (peak)
    'lookback_window': 24,                 # hours of history for LSTM
    'forecast_horizon': 24,                # hours ahead to forecast
}
```

Model hyperparameters (hidden dimension, number of layers, etc.) can also be adjusted when constructing `LSTMPredictor` instances in `main.py`.

---

## Outputs

After running the pipeline, the `results/` directory contains:

| File                     | Description                                          |
|--------------------------|------------------------------------------------------|
| `wind_lstm.pth`          | Trained wind LSTM checkpoint                         |
| `solar_lstm.pth`         | Trained solar LSTM checkpoint                        |
| `dispatch_overview.png`  | Three-panel plot: forecasts, allocation, H₂ storage  |
| `economics_carbon.png`   | Two-panel plot: hourly revenue + carbon savings      |
| `results.json`           | Structured JSON with all metrics                     |
| `llm_analysis.txt`       | LLM strategy analysis (if API configured)            |
| `analysis_report.md`     | Full formatted analysis report                       |

---

## Technologies

- **Python** — Core language
- **PyTorch** — LSTM neural network implementation
- **scikit-learn** — Data preprocessing and evaluation metrics
- **pandas / numpy** — Data manipulation and numerical computation
- **matplotlib / seaborn** — Results visualization
- **Anthropic / OpenAI APIs** — Optional LLM integration

---

## Future Work

The project includes several planned upgrade paths:

- **Short-term**: Upgrade to reinforcement learning dispatch (DQN / PPO).
- **Medium-term**: Introduce multi-objective optimization (NSGA-II) for Pareto-optimal trade-offs between economic and environmental goals.
- **Long-term**: Build a full thermal-electric-hydrogen multi-energy flow coupling model with sensitivity analysis.
- Additional improvements: incorporate Transformer attention mechanisms for better forecasting, stochastic programming for uncertainty handling, and real-time data feeds.

---

## License

This project was developed as an educational course project. No license is specified — please contact the author for usage terms.

---

## Topics / Tags

Suggested GitHub topics for this repository:

- `renewable-energy`
- `hydrogen-production`
- `lstm`
- `time-series-forecasting`
- `multi-energy-system`
- `combined-heat-and-power`
- `energy-dispatch`
- `green-hydrogen`
- `machine-learning`
- `power-system-optimization`
- `sustainable-energy`
- `opsd`
- `energy-management-system`
- `deep-learning`
- `carbon-emission-reduction`
