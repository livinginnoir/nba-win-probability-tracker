# 🏀 NBA Win Probability Model

A full-stack data science project that predicts real-time NBA win probabilities using XGBoost, trained on ~2,500 games from the 2023–24 and 2024–25 regular seasons. The model achieved a **Brier Score of 0.1127** and **84.14% test accuracy**, placing it within the range of professional-grade analytics tools (ESPN's BPI targets sub-0.15).

An interactive **Streamlit dashboard** lets users replay historical games, identify pivotal momentum shifts, and simulate hypothetical "what-if" scenarios.

---

## Table of Contents

- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Features](#features)
- [Dashboard](#dashboard)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [AI Usage Citations](#ai-usage-citations)

---

## Key Results

| Metric | Base Model | Tuned Model |
|--------|-----------|-------------|
| **Brier Score** | 0.1747 | **0.1127** |
| **Test Accuracy** | — | **84.14%** |

The model is highly calibrated: when it forecasts a 75% win probability, the home team wins approximately 75% of the time. SHAP analysis confirms that `score_margin`, `avg_margin_diff` (team strength), and `total_seconds_remaining` are the dominant predictive features — aligning with basketball intuition.

---

## Project Structure

```
├── notebooks/
│   ├── 01_Data_Acquisition_Cleaning_and_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Validation_and_Simulation_Testing.ipynb
├── scripts/
│   ├── data_processor.py          # Cleaning, feature engineering, ELO-style team strengths
│   ├── model_trainer.py           # XGBoost training with Optuna hyperparameter optimization
│   ├── inference_engine.py        # Single-state and batch prediction wrapper
│   └── simulation_utils.py        # Game replay, pivotal play detection, dropdown formatting
├── models/
│   └── nba_win_probability_model.json   # Serialized XGBoost Booster
├── data/
│   ├── nba_metadata_2500.parquet        # Full metadata (team names, descriptions) for the app UI
│   └── nba_wp_model_ready_2500.parquet  # Numeric features only, model-ready
├── app.py                         # Streamlit dashboard (entry point)
└── README.md
```

---

## Notebooks

The project follows a sequential notebook pipeline. Each notebook builds on the outputs of the previous one.

**01 — Data Acquisition, Cleaning & EDA:** Connects to the NBA Stats API, samples and cleans play-by-play data, forward-fills scores, optimizes memory via downcasting, and performs exploratory analysis (score distributions, temporal density, target balance).

**02 — Feature Engineering:** Builds the predictive feature set including a global game clock (`total_seconds_remaining`), lead change tracking, a "smart" possession indicator that filters neutral actions (fouls, timeouts), clutch/garbage time flags, and an ELO-style `avg_margin_diff` team strength metric capped at ±20 points.

**03 — Model Training & Bayesian Optimization:** Trains a base XGBoost classifier, then runs 100-trial Optuna optimization with early stopping and pruning callbacks. Compares base vs. tuned performance on a chronological 1.5-season train / 0.5-season test split.

**04 — Validation, Simulation & Interpretability:** Verifies calibration with a reliability diagram, stress-tests the model on comeback games to confirm sensitivity to momentum shifts, and interprets feature importance with SHAP (TreeExplainer).

---

## Features

The model uses 11 engineered features derived from raw play-by-play data:

| Feature | Description |
|---------|-------------|
| `period` | Current game period (1–4, 5+ for OT) |
| `score_margin` | Home score minus away score |
| `total_seconds_remaining` | Unified game clock across all periods |
| `is_home` | Whether the acting team is the home team |
| `total_lead_changes` | Cumulative lead changes up to the current play |
| `possession_indicator` | Which team has the ball (neutral actions filtered) |
| `is_clutch` | Score within 5 points in the final 5 minutes |
| `garbage_time_flag` | Blowout situations with low leverage |
| `avg_margin_diff` | Home team strength minus away team strength (ELO-style) |
| `home_games_played` | Games played by the home team (season context) |
| `away_games_played` | Games played by the away team (season context) |

---

## Dashboard

The Streamlit app provides three interactive modes:

**Overview** — A non-technical summary of the model's approach, methodology, and performance benchmarks.

**Historical Replay** — Select any game from the 2023–24 or 2024–25 season and scrub through the timeline to see how win probability evolved play-by-play. A reality line shows the actual outcome, and the model's prediction line should converge toward it as the game progresses. A "Top 3 Pivotal Plays" table surfaces the largest win probability swings with human-readable play descriptions.

**What-If Simulator** — Manually set score margin, time remaining, possession, and team competitiveness to see how the model responds to hypothetical scenarios. Includes an explainer on edge-case behavior (e.g., why a tied game at 0:00 doesn't always show exactly 50%).

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install pandas numpy xgboost scikit-learn optuna plotly streamlit
```

### Data

The raw play-by-play CSVs (`nbastatsv3_2023.csv`, `nbastatsv3_2024.csv`) should be placed in a `data/` directory. Run the processing pipeline to generate the parquet files:

```bash
cd scripts
python data_processor.py
```

This produces `nba_metadata_2500.parquet` and `nba_wp_model_ready_2500.parquet` in `data/`.

### Model Training (Optional)

To retrain the model from scratch:

```bash
python scripts/model_trainer.py
```

The pre-trained model is included at `models/nba_win_probability_model.json`.

---

## Usage

### Run the Dashboard

```bash
streamlit run app.py
```

### Use the Inference Engine Directly

```python
from scripts.inference_engine import NBAInferenceEngine

engine = NBAInferenceEngine("models/nba_win_probability_model.json")

state = {
    "period": 4,
    "score_margin": 5,
    "total_seconds_remaining": 300,
    "is_home": 1,
    "total_lead_changes": 12,
    "possession_indicator": 1,
    "is_clutch": 1,
    "garbage_time_flag": 0,
    "avg_margin_diff": 1.2,
    "home_games_played": 30,
    "away_games_played": 30,
}

print(f"Home Win Probability: {engine.predict_probability(state):.2%}")
```

---

## AI Usage Citations

This project uses AI-assisted code generation in specific, documented areas. All citations are preserved inline in the relevant files:

- **`model_trainer.py` / Notebook 03:** Gemini assisted with Optuna hyperparameter optimization setup and final model training with the native XGBoost API.
- **`simulation_utils.py`:** Gemini implemented dropdown label formatting, game state extraction, and pivotal play detection. Index-based join logic was manually refined.
- **`app.py`:** Gemini provided the foundational Streamlit framework and drafted the interactive UI components. CSS styling, page copy, and pivotal plays table rendering were manually adjusted.

---

## Author

**Thomas Lee** — M.S. Data Science, Eastern University
