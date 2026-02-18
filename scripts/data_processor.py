import pandas as pd
import numpy as np
import re
import os

# --- HELPER UTILITIES ---

def parse_nba_clock(clock_str):
    """Parses NBA clock string into total seconds."""
    if not isinstance(clock_str, str): return 0
    match = re.search(r'PT(\d+)M(\d+)', clock_str)
    if match:
        return (int(match.group(1)) * 60) + int(match.group(2))
    return 0

def get_total_seconds(row):
    """Calculates seconds remaining in the regulation/OT game."""
    period = row['period']
    sec_rem = row['seconds_remaining']
    return (4 - period) * 720 + sec_rem if period <= 4 else sec_rem

def check_garbage_time(row):
    """Identifies low-leverage garbage time situations."""
    margin, sec, period = abs(row['score_margin']), row['total_seconds_remaining'], row['period']
    if period >= 4:
        if (sec <= 120 and margin > 10) or (sec <= 360 and margin > 20) or (margin > 30):
            return 1
    return 0

# --- CORE PROCESSING ENGINES ---

def nba_engine_processor(df):
    """Primary cleaning and basic feature engineering pipeline."""
    # 1. CLEANING & FORMATTING
    dropped_columns = ['teamTricode', 'personId', 'playerName', 'playerNameI', 
                       'xLegacy', 'yLegacy', 'shotDistance', 'isFieldGoal', 
                       'pointsTotal', 'subType', 'videoAvailable', 'shotValue', 'actionNumber']
    
    df = df.drop(columns=[c for c in dropped_columns if c in df.columns])
    df[['scoreHome', 'scoreAway']] = df[['scoreHome', 'scoreAway']].replace('', np.nan)
    df = df.sort_values(['gameId', 'actionId'])
    
    # Fill scores forward within game
    df['scoreHome'] = df.groupby('gameId')['scoreHome'].ffill().fillna(0)
    df['scoreAway'] = df.groupby('gameId')['scoreAway'].ffill().fillna(0)
    
    # Type Casting for memory efficiency
    dtype_mapping = {'actionId': 'int32', 'period': 'int8', 'scoreHome': 'int16', 
                     'scoreAway': 'int16', 'location': 'category', 'actionType': 'category', 
                     'shotResult': 'category'}
    df = df.astype({k: v for k, v in dtype_mapping.items() if k in df.columns})
    
    # 2. CORE TIME & SCORE FEATURES
    df['seconds_remaining'] = df['clock'].apply(parse_nba_clock).astype('int16')
    df['total_seconds_remaining'] = df.apply(get_total_seconds, axis=1).astype('int16')
    df['score_margin'] = df['scoreHome'] - df['scoreAway']
    df['is_home'] = (df['location'] == 'h').astype(int)
    
    # Lead Changes
    df['margin_sign'] = np.sign(df['score_margin'])
    df['lead_changed'] = (df['margin_sign'] != df['margin_sign'].shift(1)) & (df['score_margin'] != 0)
    df['lead_changed'] = (df['lead_changed'] & (df['gameId'] == df['gameId'].shift(1))).astype(int)
    df['total_lead_changes'] = df.groupby('gameId')['lead_changed'].cumsum()
    
    # 3. SMART POSSESSION
    neutral_actions = ['S.FOUL', 'P.FOUL', 'Timeout', 'Period', 'Jump Ball']
    is_neutral = df['description'].str.contains('|'.join(neutral_actions), case=False, na=False)
    
    df['possession_indicator'] = np.nan
    df.loc[~is_neutral & (df['location'] == 'h'), 'possession_indicator'] = 1
    df.loc[~is_neutral & (df['location'] == 'v'), 'possession_indicator'] = 0
    
    df['possession_indicator'] = df.groupby(['gameId', 'period'])['possession_indicator'].bfill()
    df['possession_indicator'] = df.groupby(['gameId', 'period'])['possession_indicator'].ffill().fillna(0).astype('int8')
    
    # 4. LEVERAGE FLAGS
    df['is_clutch'] = ((df['score_margin'].abs() <= 5) & (df['total_seconds_remaining'] <= 300)).astype(int)
    df['garbage_time_flag'] = df.apply(check_garbage_time, axis=1).astype('int8')
    
    # 5. TARGET LABEL (HOME_WINS)
    win_map = df.groupby('gameId')['score_margin'].last().gt(0).astype(int).to_dict()
    df['HOME_WINS'] = df['gameId'].map(win_map)
    
    return df.drop(columns=['margin_sign'])

def calculate_strengths(season_df):
    """Calculates ELO-style average margin differentials for teams."""
    mapping = season_df.groupby('gameId').apply(lambda x: pd.Series({
        'teamIdHome': x.loc[x['location'] == 'h', 'teamId'].iloc[0],
        'teamIdAway': x.loc[x['location'] == 'v', 'teamId'].iloc[0]
    }), include_groups=False).reset_index()
    
    results = season_df.groupby('gameId').agg({'score_margin': 'last'}).reset_index()
    results = results.merge(mapping, on='gameId').sort_values('gameId')
    results['capped_margin'] = np.clip(results['score_margin'], -20, 20)
    
    h = results[['gameId', 'teamIdHome', 'capped_margin']].rename(columns={'teamIdHome': 'teamId', 'capped_margin': 'margin'})
    v = results[['gameId', 'teamIdAway', 'capped_margin']].rename(columns={'teamIdAway': 'teamId', 'capped_margin': 'margin'})
    v['margin'] = -v['margin']
    
    hist = pd.concat([h, v]).sort_values(['teamId', 'gameId'])
    hist['avg_margin'] = hist.groupby('teamId')['margin'].apply(lambda x: x.expanding().mean().shift(1)).reset_index(level=0, drop=True)
    hist['games_played'] = hist.groupby('teamId').cumcount()
    
    results = results.merge(
        hist[['gameId', 'teamId', 'avg_margin', 'games_played']], 
        left_on=['gameId', 'teamIdHome'], right_on=['gameId', 'teamId'], how='left'
    ).rename(columns={'avg_margin': 'h_s', 'games_played': 'home_games_played'}).drop(columns='teamId')

    results = results.merge(
        hist[['gameId', 'teamId', 'avg_margin', 'games_played']], 
        left_on=['gameId', 'teamIdAway'], right_on=['gameId', 'teamId'], how='left'
    ).rename(columns={'avg_margin': 'a_s', 'games_played': 'away_games_played'}).drop(columns='teamId')

    results['avg_margin_diff'] = results['h_s'] - results['a_s']
    
    merged_df = season_df.merge(
        results[['gameId', 'avg_margin_diff', 'home_games_played', 'away_games_played']], 
        on='gameId', how='left'
    )
    
    cols_to_fill = ['avg_margin_diff', 'home_games_played', 'away_games_played']
    merged_df[cols_to_fill] = merged_df[cols_to_fill].fillna(0)
    
    return merged_df

# --- EXECUTION PIPELINE ---

def run_main_pipeline():
    """End-to-end execution of the script."""
    print("Starting NBA Data Processor...")

    # 1. Load Data
    df_23 = pd.read_csv('../data/nbastatsv3_2023.csv')
    df_24 = pd.read_csv('../data/nbastatsv3_2024.csv')

    # 2. Process Seasons
    p_23 = calculate_strengths(nba_engine_processor(df_23))
    p_24 = calculate_strengths(nba_engine_processor(df_24))

    # 3. Concatenate and Clean for XGBoost
    df_model = pd.concat([p_23, p_24], ignore_index=True)
    cols_to_drop = ['clock', 'description', 'location', 'actionType', 'shotResult', 
                    'teamId', 'actionId', 'seconds_remaining']
    df_model = df_model.drop(columns=[c for c in cols_to_drop if c in df_model.columns])

    # 4. Final Optimization and Save
    optimized_dtypes = {
        'is_home': 'int8', 'lead_changed': 'int8', 'total_lead_changes': 'int16',
        'is_clutch': 'int8', 'HOME_WINS': 'int8', 'home_games_played': 'int16',
        'away_games_played': 'int16', 'avg_margin_diff': 'float32', 'gameId': 'object'
    }
    df_model = df_model.astype(optimized_dtypes)
    
    df_model.to_parquet('../data/nba_wp_model_ready_2500.parquet')
    print("Done! Model-ready Parquet saved to ../data/")

if __name__ == "__main__":
    run_main_pipeline()