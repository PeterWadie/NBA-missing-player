# models/trainer.py
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from utils.data_utils import load_dataframe, load_json, save_json
import xgboost as xgb


def load_encoded_data(input_dir: str) -> tuple:
    """
    Loads the encoded binary data and returns features and target.
    """
    df_encoded = load_dataframe(input_dir, "encoded_binary_data")
    feature_cols = (
        [f"home_{i}" for i in range(4)]
        + [f"away_{i}" for i in range(5)]
        + [
            "home_team",
            "away_team",
            "starting_min",
            "candidate_player",
            "candidate_overall_win_rate",
            "candidate_home_win_rate",
            "candidate_away_win_rate",
            "candidate_avg_start_min",
            "candidate_avg_synergy_with_present",
            "candidate_avg_head2head",
            "present_4_avg_synergy",
        ]
    )
    label_col = "label"
    X = df_encoded[feature_cols]
    y = df_encoded[label_col]
    return X, y


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "n_jobs": -1,
        "seed": 42,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train with early stopping
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Return validation logloss
    return model.best_score


def train_xgboost(input_dir: str, output_dir: str = None) -> None:
    """
    Trains an XGBoost model with automated hyperparameter tuning.
    
    Args:
        input_dir (str): Directory containing the training data
        output_dir (str, optional): Directory to save the model. If None, use input_dir.
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy necessary files from input_dir to output_dir
    for file_name in ["player_encoder.pkl", "team_encoder.pkl", "players_stats.json", "team_rosters.json"]:
        source_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        if os.path.exists(source_path) and input_dir != output_dir:
            shutil.copy(source_path, dest_path)
    
    # Load data
    X, y = load_encoded_data(input_dir)
    
    # Check if we should perform hyperparameter tuning or use existing parameters
    best_params_path = os.path.join(input_dir, "best_xgb_params.json")
    
    # Use predefined parameters if modeling for 2016 (based on 2015 data)
    if "2016" in output_dir and os.path.exists(best_params_path):
        print("Using 2015's best parameters for 2016 predictions")
        best_params = load_json(input_dir, "best_xgb_params")
    # Check if we already have optimized parameters
    elif os.path.exists(best_params_path) and not "2016" in output_dir:
        print(f"Loading existing optimized parameters from {best_params_path}")
        best_params = load_json(input_dir, "best_xgb_params")
    else:
        print("Performing hyperparameter optimization with Optuna")
        # Split data for hyperparameter tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create Optuna study for hyperparameter optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=50
        )
        
        # Get the best parameters
        best_params = study.best_params
        
        # Add fixed parameters
        best_params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "n_jobs": -1,
            "seed": 42,
        })
        
        # Save the best parameters
        save_json(best_params, output_dir, "best_xgb_params")
        print(f"Best hyperparameters: {best_params}")

    # Use best parameters to train final model on all data
    dtrain = xgb.DMatrix(X, label=y)
    
    # Remove non-XGBoost parameters if present
    xgb_params = best_params.copy()
    if "direction" in xgb_params:
        xgb_params.pop("direction")
        
    final_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
    )
    
    # Save model
    model_path = os.path.join(output_dir, "best_xgb_model.json")
    final_model.save_model(model_path)
    print(f"Model saved to {model_path}")
