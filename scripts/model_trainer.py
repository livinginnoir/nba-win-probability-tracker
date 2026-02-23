import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import log_loss
from optuna.integration import XGBoostPruningCallback

def load_and_split(file_path):
    """Loads the parquet and splits based on a specific game count index."""
    df = pd.read_parquet(file_path)
    unique_games = df['gameId'].unique()
    
    # 1.5 Seasons for training, Remaining 0.5 for testing
    train_games = unique_games[:1845] 
    test_games = unique_games[1845:]
    
    train_df = df[df['gameId'].isin(train_games)]
    test_df = df[df['gameId'].isin(test_games)]
    # ---------------------------
    
    features = [col for col in df.columns if col not in ['HOME_WINS', 'gameId']]
    
    print(f"Split Summary: {len(train_games)} games for Training, {len(test_games)} games for Testing")
    
    return (train_df[features], train_df['HOME_WINS'], 
            test_df[features], test_df['HOME_WINS'])

def objective(trial, dtrain, dtest, y_test):
    """Optuna objective function for Native XGBoost API."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
    }

    pruning_callback = XGBoostPruningCallback(trial, 'test-logloss')

    bst = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dtest, 'test')], early_stopping_rounds=50,
        callbacks=[pruning_callback], verbose_eval=False
    )
    
    preds = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    return log_loss(y_test, preds)

def main():
    DATA_PATH = '../data/nba_wp_model_ready_2500.parquet'
    MODEL_PATH = '../models/nba_win_probability_model.json'

    # Load and Split
    X_train, y_train, X_test, y_test = load_and_split(DATA_PATH)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 1. Run Optuna Study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, dtrain, dtest, y_test), n_trials=100)

    # 2. Final Training with Optimized Parameters
    final_params = study.best_params
    final_params.update({
        'objective': 'binary:logistic', 
        'eval_metric': 'logloss', 
        'tree_method': 'hist'
    })

    final_model = xgb.train(
        params=final_params, dtrain=dtrain, num_boost_round=1000,
        evals=[(dtest, 'test')], early_stopping_rounds=50, verbose_eval=100
    )

    # 3. Export Model
    final_model.save_model(MODEL_PATH)
    print(f"Success: Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()