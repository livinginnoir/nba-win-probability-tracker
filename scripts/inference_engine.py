import xgboost as xgb
import pandas as pd

class NBAInferenceEngine:
    def __init__(self, model_path):
        """Loads the saved Native Booster."""
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Hardcoded features to ensure consistency with the trainer
        self.features = [
            'period', 'score_margin', 'total_seconds_remaining', 
            'is_home', 'total_lead_changes', 'possession_indicator', 
            'is_clutch', 'garbage_time_flag', 'avg_margin_diff', 
            'home_games_played', 'away_games_played'
        ]

    def predict_probability(self, game_state_dict):
        """Returns the Home Win Probability (0.0 to 1.0)."""
        # Convert dict to DF and ensure feature order
        df = pd.DataFrame([game_state_dict])
        df = df[self.features] 
        
        # Convert to Native DMatrix
        dmatrix = xgb.DMatrix(df)
        
        # Inference
        prob = self.model.predict(dmatrix)[0]
        return float(prob)
    
    def predict_batch(self, df):
        """
        Used for replaying a whole game. 
        Processes an entire DataFrame at once for the Streamlit chart.
        """
        # Ensure we only pass the columns the model expects, in the right order
        dmatrix = xgb.DMatrix(df[self.features])
        
        # Returns an array of probabilities
        return self.model.predict(dmatrix)

if __name__ == "__main__":
    # Test the engine with a hypothetical scenario
    engine = NBAInferenceEngine('../models/nba_win_probability_model.json')
    
    sample_state = {
        'period': 4, 'score_margin': 5, 'total_seconds_remaining': 300,
        'is_home': 1, 'total_lead_changes': 12, 'possession_indicator': 1,
        'is_clutch': 1, 'garbage_time_flag': 0, 'avg_margin_diff': 1.2,
        'home_games_played': 30, 'away_games_played': 30
    }
    
    print(f"Home Win Prob: {engine.predict_probability(sample_state):.2%}")