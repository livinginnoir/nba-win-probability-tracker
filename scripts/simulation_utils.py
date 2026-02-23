# AI USAGE CITATION 
# Tool: Gemini 
# Prompt: Gemini was provided directions on the functionality of certain
# features, including dropdown labels for games, extraction of specific game
# states by timestamp, and the ability to identify the top "pivotal" plays
# based on win probability swings.
# Usage: In addition to implementing the functions described above, I manually
# refined the get_pivotal_plays logic to ensure the index-based join accurately
# mapped the WPA swings to the correct text descriptions.

import pandas as pd

class NBASimulator:
    def __init__(self, model_data_path, metadata_path):
        """Loads two versions: one for the math, one for the labels."""
        self.model_df = pd.read_parquet(model_data_path)
        self.meta_df = pd.read_parquet(metadata_path)

    def get_available_games_with_names(self):
        """Dynamically identifies Home and Away teams and formats the dropdown label."""
        # Use meta_df because it contains 'location' and 'teamTricode'
        subset = self.meta_df[['gameId', 'location', 'teamTricode']].drop_duplicates()
        
        # Group to find which tricode belongs to 'h' (Home) and 'v' (Visitor)
        game_teams = subset.groupby(['gameId', 'location'])['teamTricode'].first().unstack()
        
        game_list = []
        for game_id, row in game_teams.iterrows():
            home = row.get('h', 'Home')
            away = row.get('v', 'Away')
            
            # --- UPDATED LABEL FORMAT ---
            # This adds the gameId prefix you requested
            display_label = f"{game_id}: {home} vs {away}"
            
            game_list.append({
                'id': game_id,
                'label': display_label
            })
        return game_list

    def get_full_game_data(self, game_id):
        """Returns the model-ready data for the Inference Engine."""
        # Pulls from model_df to ensure high-performance math
        return self.model_df[self.model_df['gameId'] == game_id].sort_values(
            'total_seconds_remaining', ascending=False
        )

    def get_game_state_at_time(self, game_df, seconds_remaining):
        """Finds the 'frame' of the game at a specific timestamp."""
        state = game_df[game_df['total_seconds_remaining'] >= seconds_remaining].iloc[-1]
        return state.to_dict()

    def get_pivotal_plays(self, game_df, top_n=3):
        """
        Identifies momentum shifts and merges with metadata for human-readable labels.
        """
        if 'win_prob' not in game_df.columns:
            return None
        
        game_df = game_df.copy()
        # Calculate Win Probability Added (WPA)
        game_df['WPA'] = game_df['win_prob'].diff().fillna(0)
        
        # Get indices of the top absolute swings
        pivotal_indices = game_df['WPA'].abs().sort_values(ascending=False).index[:top_n]
        
        # 1. Start with the math data from game_df
        pivotal_math = game_df.loc[pivotal_indices]
        
        # 2. Join with the human-readable metadata using the shared index
        # We only need the 'description' from the meta_df
        pivotal_final = pivotal_math.join(
            self.meta_df[['description']], 
            how='left'
        )
        
        return pivotal_final

# --- Updated Testing for the Dual-Load Slicer ---
if __name__ == "__main__":
    MODEL_DATA = '../data/nba_wp_model_ready_2500.parquet'
    META_DATA = '../data/nba_metadata_2500.parquet'
    
    sim = NBASimulator(MODEL_DATA, META_DATA)
    
    # Test dropdown labels
    games = sim.get_available_games_with_names()
    print(f"First game label: {games[0]['label']}")
    
    # Test pivotal plays with metadata join
    game_id = games[0]['id']
    full_game = sim.get_full_game_data(game_id)
    full_game['win_prob'] = 0.5 # Dummy prob for test
    
    pivotal = sim.get_pivotal_plays(full_game)
    print("Top Pivotal Plays with Action labels:")
    print(pivotal[['actionType', 'WPA']])