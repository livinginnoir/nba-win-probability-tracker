# AI USAGE CITATION
# Tool: Gemini
# Prompt: Gemini was provided headings and text for each page, as well as the
# general design direction for the app itself, and it returns most of the code
# below.
# Usage: Provided the foundational framework for Streamlit page. Generated the
# visualization and drafted the interactive UI components for the manual game 
# simulation tool. I manually adjusted the CSS styling, page copy, and the specific 
# logic for the 'Pivotal Plays' table rendering.

import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.inference_engine import NBAInferenceEngine
from scripts.simulation_utils import NBASimulator

# 1. GLOBAL SETTINGS (Must be first)
st.set_page_config(page_title="Thomas Lee | NBA Win Probability Project", layout="wide")

# Load Engines once for the whole app
@st.cache_resource
def load_engines():
    engine = NBAInferenceEngine('models/nba_win_probability_model.json')
    simulator = NBASimulator(
        'data/nba_wp_model_ready_2500.parquet', 
        'data/nba_metadata_2500.parquet'
    )
    return engine, simulator

engine, simulator = load_engines()

# 2. NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Bio & Intro", "Resume", "NBA Win Probability Model", "Other Projects"])

# --- PAGE: BIO & INTRO ---
if page == "Bio & Intro":
    st.title("Biographical Homepage")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # st.image("your_photo.jpg") # Optional: Add a professional photo
        st.subheader("Thomas Lee")
        st.write("Korean-American Aspiring Data Scientist") #
        st.write("Masters in Data Science @ Eastern University") #

    with col2:
        st.header("Hello Everyone")
        st.write("""
        I am a graduate student completing my final semester of the Master's in Data Science program at Eastern University. 
        My academic journey began with a Bachelor's in Sociology at University of California, Irvine, which gives me a unique perspective 
        on data through a human-centric lens.
        """) #
        st.write("""
        Before pivoting to data science, I worked several years as a digital marketing specialist, where 
        I translated business data into actionable insights to drive KPIs across Content, SEO, and Paid Media.
        """) #
        
        st.subheader("Travels & Interests")
        st.write("My wife and I travel regularly to Japan and South Korea. We're big on exploring new " \
        "coffee, food, and nature at home and abroad.") #
        st.write("Outside of coding, I like music, books, horror movies, and the NBA.") #

# --- PAGE: RESUME ---
elif page == "Resume":
    st.title("Resume")
    st.subheader("Education")
    st.write("**M.S. Data Science at Eastern University** | Expected March 2026") #
    st.write("**B.A. Sociology at University of California, Irvine**") #
    
    st.subheader("Technical Skills")
    st.image("assets/data_image.png", width=200) 
    st.write("Proficient in Python (Pandas, Scikit-Learn, XGBoost), R (Tidyverse), and SQL.") #
    
    st.write("To view my full resume for details on work experience and more, click the button below:") #
    st.link_button("View Resume", "https://docs.google.com/document/d/1S8obSR4Qj_-s9p0ekHeRkgZ-Umdo41VRu8XfyAEEzP8/edit?usp=sharing")

# --- PAGE: NBA WIN PROBABILITY MODEL ---
elif page == "NBA Win Probability Model":
    st.title("🏀 NBA Win Probability Dashboard")
    
    mode = st.sidebar.radio("Select Dashboard Mode", 
                            ["Overview", "Historical Replay", "What-If Simulator"])

    # --- MODE 1: HISTORICAL REPLAY ---
    if mode == "Overview":
        st.header("📝 Overview")
        # NON-TECHNICAL OVERVIEW
        st.markdown("""
        **The Problem:** NBA games are fast-paced and volatile. A single play can monumentally shift the win prospects of either team, making it hard for fans and analysts to know the *true* state of a game.
        
        **The Approach:** This model analyzes ~2,500 historical games using **XGBoost**, which can be described as a team of "specialist" decision trees that are built one after the other, where each new tree focuses exclusively on correcting the specific mistakes made by the previous ones. By looking primarily at the score, time remaining, and ball possession, in addition to team strengths, it calculates the statistical likelihood of victory for the Home Team at any given moment.
        
        **The Results:** The model achieved a **Brier Score of 0.1127**. In predictive modeling, a lower Brier Score indicates higher accuracy, with 0 being a perfect prediction and 0.25 representing a random 50/50 guess. This score of 0.1127 means the model is highly calibrated with reality; when it forecasts a 75% win probability, the team wins approximately 75% of the time. For comparison, professional industry standards for NBA win probability models (e.g. ESPN) typically aim for a Brier Score between **0.10 and 0.13**, placing this model within the range of high-performance analytical tools.
        """)

    # --- MODE 2: HISTORICAL REPLAY ---
    elif mode == "Historical Replay":
        st.header("🕰️ Historical Game Replay & Validation")

        # Overview Text
        st.write("""
        Choose a game from the 2023-2024 or 2024-2025 regular season. Scrub the timeline to see how the model's predictions evolved play-by-play. 
        
        The **Orange Line (Reality)** represents the binary ground truth of the game. A constant 1.0 means that the home team won in reality, while a constant 0.0 means
        the away team won in reality.
            
        As the game progresses (moving left to right toward 0 seconds), the **Blue Line (Model prediction)** should *ideally* converge with the Orange Line, showing the model has correctly 'solved' 
        the game's outcome before the final buzzer.
        """)
        st.divider()
        
        # 1. Selection Logic
        game_options = simulator.get_available_games_with_names()
        label_to_id = {g['label']: g['id'] for g in game_options}
        selected_label = st.selectbox("Select a Game (GameId: Home vs Away)", list(label_to_id.keys()))
        selected_game_id = label_to_id[selected_label]
        
        # 2. Extract Home Team for labeling
        # Splitting "gameId: HOME vs AWAY" to get "HOME"
        home_team = selected_label.split(": ")[1].split(" vs ")[0]
        
        # 3. Data Preparation
        full_game = simulator.get_full_game_data(selected_game_id)
        full_game['win_prob'] = engine.predict_batch(full_game)
        
        # Define Reality Line (1 if Home Won, 0 if Away Won)
        actual_winner_val = full_game['HOME_WINS'].iloc[0]
        full_game['Reality'] = actual_winner_val
        winner_text = home_team if actual_winner_val == 1 else selected_label.split(" vs ")[1]

        # 4. Timeline Slider
        st.subheader(f"Timeline: {home_team} Win Probability")
        time_step = st.select_slider(
            "Scrub through the game (Seconds Remaining)",
            options=full_game['total_seconds_remaining'].tolist()
        )
        current_state = simulator.get_game_state_at_time(full_game, time_step)
        
        # 5. Visual Layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Score Margin", f"{current_state['score_margin']}")
            st.metric(f"{home_team} Win Prob.", f"{current_state['win_prob']:.2%}")
            st.write(f"Period: {current_state['period']}")
            st.success(f"🏆 Actual Winner: {winner_text}")
        
        with col2:
            # Chart with dual lines: win_prob and Reality
            fig = px.line(full_game, x='total_seconds_remaining', y=['win_prob', 'Reality'],
                        labels={'value': 'Probability', 'total_seconds_remaining': 'Seconds Remaining'},
                        title=f"Model Prediction vs. Outcome: {selected_label}",
                        color_discrete_map={"win_prob": "#1f77b4", "Reality": "#ff7f0e"})
            
            fig.add_vline(x=time_step, line_dash="dash", line_color="red", annotation_text="Current Play")
            fig.update_xaxes(autorange="reversed")
            fig.update_yaxes(range=[-0.05, 1.05])
            st.plotly_chart(fig, use_container_width=True)

        # --- PIVOTAL PLAYS SECTION ---
        st.divider()
        st.subheader(f"🔑 Top 3 Pivotal Momentum Shifts for {home_team}")

        # Use the simulator method to get the data AND the description join
        pivotal_plays = simulator.get_pivotal_plays(full_game)

        if pivotal_plays is not None:
            pivotal_table = pivotal_plays.copy()
            
            # Formatting Time
            pivotal_table['Time'] = pivotal_table['total_seconds_remaining'].apply(
                lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
            )
            
            # Rename columns for the UI
            pivotal_table = pivotal_table.rename(columns={
                'period': 'Period',
                'score_margin': 'Score Margin',
                'description': 'Play Description', # Match what was joined in Step 1
                'WPA': 'Prob. Swing',
                'win_prob': 'New Prob.'
            })

            # Render table with only the requested columns
            st.table(pivotal_table[['Period', 'Time', 'Score Margin', 'Play Description', 'Prob. Swing', 'New Prob.']])

    # --- MODE 2: WHAT-IF SIMULATOR ---
    elif mode == "What-If Simulator":
        st.header("🎮 'What-If' Game Simulation Tool")
        
        # Overview for non-technical audience
        st.write("""
        This tool allows you to simulate hypothetical game scenarios. By adjusting the score, time, current possession, 
        and team strength, you can see how the model's win probability reacts to different high-pressure 
        situations in real-time.
        """)
        st.divider()

        with st.expander("Adjust Game Scenario", expanded=True):
            col1, col2 = st.columns(2)
            margin = col1.number_input("Score Margin (Home - Away)", value=0)
            time_rem = col2.slider("Seconds Remaining", 0, 2880, 600)
            
            col3, col4 = st.columns(2)
            possession = col3.selectbox("Current Possession", ["Home Team", "Away Team"])
            
            # --- COMPETITIVENESS DROPDOWN ---
            # Mapping descriptive labels to numerical avg_margin_diff values
            comp_options = {
                "Major Mismatch (Home advantage)": 10.0,
                "Moderate Mismatch (Home advantage)": 5.0,
                "Minor Mismatch (Home advantage)": 2.0,
                "Even (Teams at similar level)": 0.0,
                "Minor Mismatch (Away advantage)": -2.0,
                "Moderate Mismatch (Away advantage)": -5.0,
                "Major Mismatch (Away advantage)": -10.0
            }
            
            selected_comp = col4.selectbox("Competitiveness", list(comp_options.keys()), index=3)
            comp_value = comp_options[selected_comp]

        # Format for inference
        manual_state = {
            'period': 4 if time_rem <= 720 else 3, 
            'score_margin': margin,
            'total_seconds_remaining': time_rem,
            'possession_indicator': 1 if possession == "Home Team" else 0,
            'avg_margin_diff': comp_value, # Now controlled by the dropdown
            # Static defaults for manual simulation
            'is_home': 1, 
            'total_lead_changes': 5, 
            'is_clutch': 1 if (abs(margin) <= 5 and time_rem <= 300) else 0, 
            'garbage_time_flag': 0, 
            'home_games_played': 41, 
            'away_games_played': 41
        }
        
        prob = engine.predict_probability(manual_state)
        
        # Centered Result Display
        st.markdown(f"<h3 style='text-align: center;'>Predicted Home Win Probability</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        st.progress(prob)

        # --- MODEL CALIBRATION & EDGE CASES ---
        st.divider()
        with st.expander("🔍 Deep Dive: Why is the probability not 50/50 in a tie?"):
            st.write("""
            In your simulation, you may notice that a tied game with 0 seconds remaining doesn't always 
            result in a perfect 50% probability. Here is the technical breakdown of why:
            """)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**1. Data Sparsity at 'Zero'**")
                st.write("""
                Machine Learning models like XGBoost learn from patterns. In the training dataset of ~2,500 games, 
                there are very few instances where the clock is at exactly 0.0 but the game hasn't officially 
                ended. The model is essentially extrapolating its best guess based on 
                late-game trends rather than a large sample of 0-second ties.
                """)

            with col_b:
                st.markdown("**2. The Overtime Bias**")
                st.write("""
                In a tie game at the buzzer, the 'Reality' is usually Overtime. Historically, home teams win 
                about 52-53% of OT games. If the model is showing a dip (e.g., 43%), it may be picking up on 
                specific 'Underdog' effects or noise within the 2023-2024 and/or 2024-2025 season data that suggests the 
                away team had a slight momentum edge in close finishes.
                """)

# --- PAGE: OTHER PROJECTS ---
elif page == "Other Projects":
    st.title("General Projects Portfolio")
    st.info("A collection of my other work in Data Science.")

    # NBA Draft Project
    st.subheader("🏀 NBA Draft Players Performance Project (WIP)")
    col1, col2 = st.columns([1, 2])
    with col1:
        # A placeholder or relevant graphic for the draft project
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/315px-National_Basketball_Association_logo.svg.png", width=100)
    with col2:
        st.write("""
        This project analyzes the relationship between NBA Draft positions and actual on-court performance. 
        It involves scraping historical draft data and player statistics to identify 'steals' and 'busts' 
        using statistical analysis in R.
        """)
        st.link_button("View GitHub Repository", "https://github.com/livinginnoir/nba-draft-players-performance-project")