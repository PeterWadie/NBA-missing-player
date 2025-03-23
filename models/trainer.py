# models/trainer.py
import os
from utils.data_utils import load_dataframe, load_json
import xgboost as xgb


def load_encoded_data(output_dir: str) -> tuple:
    """
    Loads the encoded binary data and returns features and target.
    """
    df_encoded = load_dataframe(output_dir, "encoded_binary_data")
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


def train_xgboost(output_dir: str) -> None:
    """
    Trains an XGBoost model using the best parameters.
    """
    X, y = load_encoded_data(output_dir)
    best_params = load_json(output_dir, "best_xgb_params")
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "n_jobs": -1,
            "seed": 42,
        }
    )
    final_model = xgb.train(
        best_params,
        xgb.DMatrix(X, label=y),
        num_boost_round=1000,
    )
    final_model.save_model(os.path.join(output_dir, "best_xgb_model.json"))
