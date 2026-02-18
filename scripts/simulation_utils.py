import pandas as pd

class NBASimulator:
    def __init__(self, data_path):
        """Load the processed dataset."""
        self.df = pd.read_parquet(data_path)

    def get_available_games(self):
        """Returns unique Game IDs for the Streamlit selection menu."""
        return self.df['gameId'].unique().tolist()

    def get_full_game_data(self, game_id):
        """Returns the entire play-by-play for a game, sorted chronologically."""
        return self.df[self.df['gameId'] == game_id].sort_values(
            'total_seconds_remaining', ascending=False
        )

    def get_game_state_at_time(self, game_df, seconds_remaining):
        """
        SLICING LOGIC: 
        Finds the exact 'frame' of the game at a specific timestamp.
        Used to update the 'Live Score' and 'Current Probability' in the app.
        """
        # Find the row closest to the requested time without going past it
        state = game_df[game_df['total_seconds_remaining'] >= seconds_remaining].iloc[-1]
        return state.to_dict()

    def get_pivotal_plays(self, game_df, top_n=3):
        """Identifies the biggest momentum shifts (Win Probability Added)."""
        if 'win_prob' not in game_df.columns:
            return None
        
        # Calculate the absolute change in probability from the previous play
        game_df = game_df.copy()
        game_df['prob_delta'] = game_df['win_prob'].diff().abs()
        
        return game_df.nlargest(top_n, 'prob_delta')

# --- Testing the Slicer ---
if __name__ == "__main__":
    SIM_DATA = '../data/nba_wp_model_ready_2500.parquet'
    sim = NBASimulator(SIM_DATA)
    
    # Grab a game and slice it at exactly 5 minutes (300 seconds) left
    game_id = sim.get_available_games()[0]
    full_game = sim.get_full_game_data(game_id)
    
    five_min_mark = sim.get_game_state_at_time(full_game, 300)
    print(f"At 5:00 remaining, the score margin was: {five_min_mark['score_margin']}")