# CODE CITATION HERE

import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.inference_engine import NBAInferenceEngine
from scripts.simulation_utils import NBASimulator

# 1. INITIALIZATION
st.set_page_config(page_title="NBA Win Probability Dashboard", layout="wide")
st.title("🏀 NBA Win Probability: Analytics Dashboard")

# Load our Back-End Engines
@st.cache_resource
def load_engines():
    engine = NBAInferenceEngine('models/nba_win_probability_model.json')
    simulator = NBASimulator('data/nba_wp_model_ready_2500.parquet')
    return engine, simulator

engine, simulator = load_engines()

# 2. SIDEBAR - Mode Selection
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Select Dashboard Mode", 
                         ["Historical Replay", "What-If Simulator", "Model vs. Reality"])

# --- MODE 1: HISTORICAL REPLAY ---
if mode == "Historical Replay":
    st.header("🕰️ Historical Game Replay")
    
    # Game Selector
    game_list = simulator.get_available_games()
    selected_game = st.selectbox("Select a Game from the 2024-25 Season", game_list)
    
    # Load and Predict the whole game for the chart
    full_game = simulator.get_full_game_data(selected_game)
    full_game['win_prob'] = engine.predict_batch(full_game)
    
    # The Slider (Scrubbing through time)
    st.subheader("Game Timeline")
    time_step = st.select_slider(
        "Scrub through the game (Seconds Remaining)",
        options=full_game['total_seconds_remaining'].tolist()
    )
    
    # Get state at selected time
    current_state = simulator.get_game_state_at_time(full_game, time_step)
    
    # Visual Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Score Margin", f"{current_state['score_margin']}")
        # We grab the probability for the specific 'sliced' state
        st.metric("Live Win Probability", f"{current_state['win_prob']:.2%}")
        st.write(f"Period: {current_state['period']}")
    
    with col2:
        fig = px.line(full_game, x='total_seconds_remaining', y='win_prob', 
                      title=f"Win Probability Curve: {selected_game}")
        fig.add_vline(x=time_step, line_dash="dash", line_color="red")
        fig.update_xaxes(autorange="reversed") # Game flows toward 0
        st.plotly_chart(fig, use_container_width=True)

    # --- PIVOTAL PLAYS SECTION (NEW) ---
    st.divider()
    st.subheader("🔑 Top 3 Pivotal Momentum Shifts")
    st.info("The moments where the Home Win Probability swung most dramatically.")

    # 1. Calculate Win Probability Added (WPA)
    # Sorting ensures diff() calculates the change from the previous chronological play
    full_game = full_game.sort_values('total_seconds_remaining', ascending=False)
    full_game['WPA'] = full_game['win_prob'].diff().fillna(0)
    
    # 2. Identify top 3 absolute swings
    pivotal_plays = full_game.reindex(full_game['WPA'].abs().sort_values(ascending=False).index).head(3)
    
    # 3. Format and Display
    display_cols = ['period', 'score_margin', 'total_seconds_remaining', 'WPA', 'win_prob']
    pivotal_table = pivotal_plays[display_cols].copy()
    
    # Convert seconds to MM:SS
    pivotal_table['Time'] = pivotal_table['total_seconds_remaining'].apply(
        lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
    )
    
    # Professional Renaming
    pivotal_table = pivotal_table.rename(columns={
        'period': 'Prd',
        'score_margin': 'Margin',
        'WPA': 'Prob. Swing',
        'win_prob': 'New Prob.'
    })

    # Render table
    st.table(pivotal_table[['Prd', 'Time', 'Margin', 'Prob. Swing', 'New Prob.']])

# --- MODE 2: WHAT-IF SIMULATOR ---
elif mode == "What-If Simulator":
    st.header("🎮 'What-If' Coaching Tool")
    
    with st.expander("Adjust Game Scenario", expanded=True):
        col1, col2, col3 = st.columns(3)
        margin = col1.number_input("Score Margin (Home - Away)", value=0)
        time_rem = col2.slider("Seconds Remaining", 0, 2880, 600)
        possession = col3.selectbox("Possession", ["Home Team", "Away Team"])
        
    # Format for inference
    manual_state = {
        'period': 4 if time_rem <= 720 else 3, # Simplification for demo
        'score_margin': margin,
        'total_seconds_remaining': time_rem,
        'possession_indicator': 1 if possession == "Home Team" else 0,
        # Defaulting other features for the manual sim
        'is_home': 1, 'total_lead_changes': 5, 'is_clutch': 0, 
        'garbage_time_flag': 0, 'avg_margin_diff': 0, 
        'home_games_played': 41, 'away_games_played': 41
    }
    
    prob = engine.predict_probability(manual_state)
    st.markdown(f"<h1 style='text-align: center;'>{prob:.1%}</h1>", unsafe_allow_html=True)
    st.progress(prob)

# --- MODE 3: MODEL VS. REALITY ---
elif mode == "Model vs. Reality":
    st.header("⚖️ Model Confidence vs. Game Outcome")
    st.info("This view compares the model's 'Confidence' (Probability) against the 'Truth' (Who actually won).")

    # Game Selector
    game_list = simulator.get_available_games()
    selected_game = st.selectbox("Select Game to Validate", game_list)
    
    # 1. Fetch Game Data
    full_game = simulator.get_full_game_data(selected_game)
    
    # 2. Generate Predictions
    full_game['win_prob'] = engine.predict_batch(full_game)
    
    # 3. Define Reality (The outcome is the same for every row of a specific game)
    actual_winner = full_game['HOME_WINS'].iloc[0]
    full_game['Reality'] = actual_winner
    
    # 4. Plotting the Comparison
    fig = px.line(full_game, x='total_seconds_remaining', y=['win_prob', 'Reality'],
                  labels={'value': 'Probability / Outcome', 'total_seconds_remaining': 'Seconds Remaining'},
                  title=f"Prediction Accuracy for Game {selected_game}",
                  color_discrete_map={"win_prob": "#1f77b4", "Reality": "#ff7f0e"})

    # Reverse X-axis so game flows from left (start) to right (finish)
    fig.update_xaxes(autorange="reversed")
    
    # Add a horizontal line at 50% to show the "toss-up" threshold
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="50/50 Threshold")

    st.plotly_chart(fig, use_container_width=True)

    # 5. Summary Statistics
    col1, col2 = st.columns(2)
    winner_text = "Home Team" if actual_winner == 1 else "Away Team"
    col1.success(f"Actual Winner: {winner_text}")
    
    # Calculate Mean Absolute Error for this specific game
    mae = (full_game['win_prob'] - full_game['Reality']).abs().mean()
    col2.metric("Game Prediction Error (MAE)", f"{mae:.4f}")
