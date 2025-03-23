# models/evaluator.py
import os
import pickle
import json
import random
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from utils.data_utils import load_matchup_data
from utils.feature_utils import build_candidate_features


def evaluate_models(year: int, output_dir: str):
    """
    Evaluates the trained models on a test scenario.
    The test set is built from segments (where home team won) with one home player randomly removed.
    """
    # Load encoders and team rosters
    with open(os.path.join(output_dir, "player_encoder.pkl"), "rb") as f:
        player_encoder = pickle.load(f)
    with open(os.path.join(output_dir, "team_encoder.pkl"), "rb") as f:
        team_encoder = pickle.load(f)
    with open(os.path.join(output_dir, "team_rosters.json"), "r") as f:
        team_rosters = json.load(f)

    # Load player stats
    players_stats_path = os.path.join(output_dir, "players_stats.json")
    if os.path.exists(players_stats_path):
        with open(players_stats_path, "r") as f:
            players_stats = json.load(f)
    else:
        players_stats = {}

    # Load matchup data for the year
    df = load_matchup_data(year)

    # Build test scenario: for each segment where the home team won, remove one home player randomly.
    test_rows = []
    for _, row in df.iterrows():
        if row["outcome"] == 1:
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_lineup = [row[f"home_{i}"] for i in range(5)]
            away_lineup = [row[f"away_{i}"] for i in range(5)]
            missing_index = random.choice(range(5))
            missing_player = home_lineup[missing_index]
            present_4 = home_lineup[:missing_index] + home_lineup[missing_index + 1 :]
            test_rows.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "starting_min": row["starting_min"],
                    "home_present": present_4,
                    "away_lineup": away_lineup,
                    "true_missing_player": missing_player,
                }
            )
    df_test = pd.DataFrame(test_rows)
    print(f"Test scenario rows: {len(df_test)}")

    # Build candidate feature rows for each test segment
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

    X_test_rows = []
    y_test_truth = (
        []
    )  # List to store true missing player info along with candidate name

    for _, row in df_test.iterrows():
        home_team_name = row["home_team"]
        away_team_name = row["away_team"]
        home_present = row["home_present"]
        away_lineup = row["away_lineup"]
        true_missing_player = row["true_missing_player"]

        # Helper encoding functions
        def encode_player(name):
            return (
                player_encoder.transform([name])[0]
                if name in player_encoder.classes_
                else -1
            )

        def encode_team(name):
            return (
                team_encoder.transform([name])[0]
                if name in team_encoder.classes_
                else -1
            )

        enc_home_present = [encode_player(p) for p in home_present]
        enc_away = [encode_player(p) for p in away_lineup]
        enc_home_team = encode_team(home_team_name)
        enc_away_team = encode_team(away_team_name)
        minute = row["starting_min"]

        roster_candidates = team_rosters.get(home_team_name, [])
        for candidate in roster_candidates:
            if candidate not in player_encoder.classes_:
                continue
            enc_candidate = player_encoder.transform([candidate])[0]
            # Build candidate features using the helper function.
            candidate_features = build_candidate_features(
                candidate, home_present, away_lineup, players_stats
            )
            feature_vector = (
                enc_home_present
                + enc_away
                + [
                    enc_home_team,
                    enc_away_team,
                    minute,
                    enc_candidate,
                    candidate_features["candidate_overall_win_rate"],
                    candidate_features["candidate_home_win_rate"],
                    candidate_features["candidate_away_win_rate"],
                    candidate_features["candidate_avg_start_min"],
                    candidate_features["candidate_avg_synergy_with_present"],
                    candidate_features["candidate_avg_head2head"],
                    candidate_features["present_4_avg_synergy"],
                ]
            )
            X_test_rows.append(feature_vector)
            y_test_truth.append(
                {"true_player": true_missing_player, "candidate": candidate}
            )

    X_test = pd.DataFrame(X_test_rows, columns=feature_cols)

    # Load trained models
    lgb_model = lgb.Booster(model_file=os.path.join(output_dir, "best_lgb_model.txt"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(output_dir, "best_xgb_model.json"))
    catboost_model = cb.CatBoostClassifier()
    catboost_model.load_model(os.path.join(output_dir, "best_cb_model.cbm"))

    # Get predictions from each model
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict_proba(X_test)
    catboost_preds = catboost_model.predict_proba(X_test)

    # Ensure we have probability for class 1
    if lgb_preds.ndim == 2 and lgb_preds.shape[1] == 2:
        lgb_preds = lgb_preds[:, 1]
    if xgb_preds.ndim == 2 and xgb_preds.shape[1] == 2:
        xgb_preds = xgb_preds[:, 1]
    if catboost_preds.ndim == 2 and catboost_preds.shape[1] == 2:
        catboost_preds = catboost_preds[:, 1]

    def evaluate_predictions(pred_probs):
        index = 0
        correct = 0
        total = 0
        for _, row in df_test.iterrows():
            home_team_name = row["home_team"]
            roster_candidates = team_rosters.get(home_team_name, [])
            # Filter out candidates not in the encoder vocabulary.
            roster_candidates = [
                c for c in roster_candidates if c in player_encoder.classes_
            ]
            n_candidates = len(roster_candidates)
            segment_probs = pred_probs[index : index + n_candidates]
            segment_truth = y_test_truth[index : index + n_candidates]
            index += n_candidates
            best_candidate_idx = segment_probs.argmax()
            best_candidate = segment_truth[best_candidate_idx]["candidate"]
            true_player = segment_truth[0]["true_player"]
            if best_candidate == true_player:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0

    lgb_accuracy = evaluate_predictions(lgb_preds)
    xgb_accuracy = evaluate_predictions(xgb_preds)
    catboost_accuracy = evaluate_predictions(catboost_preds)

    results = {
        "LightGBM Accuracy": f"{lgb_accuracy * 100:.2f}%",
        "XGBoost Accuracy": f"{xgb_accuracy * 100:.2f}%",
        "CatBoost Accuracy": f"{catboost_accuracy * 100:.2f}%",
    }

    results_file = os.path.join(output_dir, "model_comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Model evaluation complete. Results saved to {results_file}")
