# _04_evaluate_models.py
import pickle
import json
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import random
from itertools import combinations

# Set random seed for reproducibility
random.seed(42)


def evaluate_models(year):
    output_dir = f"./{year}"
    # Load encoders & team rosters
    with open(f"{output_dir}/player_encoder.pkl", "rb") as f:
        player_encoder = pickle.load(f)
    with open(f"{output_dir}/team_encoder.pkl", "rb") as f:
        team_encoder = pickle.load(f)
    with open(f"{output_dir}/team_rosters.json", "r") as f:
        team_rosters = json.load(f)

    # Load player stats (generated previously)
    stats_file = f"{output_dir}/players_stats.json"
    if stats_file:
        with open(stats_file, "r") as f:
            players_stats = json.load(f)
    else:
        players_stats = {}

    # Helper functions to extract stats (same as in _00_get_players_stats.py)
    def get_player_metric(player, metric):
        if player in players_stats:
            return players_stats[player].get(metric, 0.0)
        return 0.0

    def get_synergy(p1, p2):
        if p1 in players_stats:
            synergy_dict = players_stats[p1].get("synergy_with_teammates", {})
            return synergy_dict.get(p2, 0.0)
        return 0.0

    def get_head2head(home_p, away_p):
        if home_p in players_stats:
            h2h_dict = players_stats[home_p].get("head2head_against", {})
            return h2h_dict.get(away_p, 0.0)
        return 0.0

    # Load trained models
    lgb_model = lgb.Booster(model_file=f"{output_dir}/best_lgb_model.txt")

    # For XGBoost, load probabilities using predict_proba()
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{output_dir}/best_xgb_model.json")

    # For CatBoost, load probabilities using predict_proba()
    catboost_model = cb.CatBoostClassifier()
    catboost_model.load_model(f"{output_dir}/best_cb_model.cbm")

    # Load original dataset for test scenario
    df = pd.read_csv(f"../input-dataset/matchups-{year}.csv")

    # Define feature columns (order must match training)
    # Note: We now include the extra synergy/stat features.
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

    # Build a test set from segments where outcome==1 (home team won)
    # and randomly remove one home player.
    test_rows = []
    for idx, row in df.iterrows():
        if row["outcome"] == 1:
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_lineup = [row[f"home_{i}"] for i in range(5)]
            away_lineup = [row[f"away_{i}"] for i in range(5)]
            # Randomly choose one player to be "missing"
            missing_index = random.choice(range(5))
            missing_player = home_lineup[missing_index]
            # Present players: all except the randomly removed one
            present_4 = home_lineup[:missing_index] + home_lineup[missing_index + 1 :]

            test_rows.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "starting_min": row["starting_min"],
                    "home_present_0": present_4[0],
                    "home_present_1": present_4[1],
                    "home_present_2": present_4[2],
                    "home_present_3": present_4[3],
                    "away_0": away_lineup[0],
                    "away_1": away_lineup[1],
                    "away_2": away_lineup[2],
                    "away_3": away_lineup[3],
                    "away_4": away_lineup[4],
                    "true_missing_player": missing_player,
                }
            )

    df_test_scenario = pd.DataFrame(test_rows)
    print(f"Test scenario rows: {len(df_test_scenario)}")

    # Encoding functions for test data
    def encode_player(name):
        return (
            player_encoder.transform([name])[0]
            if name in player_encoder.classes_
            else -1
        )

    def encode_team(name):
        return (
            team_encoder.transform([name])[0] if name in team_encoder.classes_ else -1
        )

    # Build candidate feature rows for each test segment
    X_test_rows = []
    y_test_truth = (
        []
    )  # List of dicts storing the true missing player and candidate name

    for idx, row in df_test_scenario.iterrows():
        home_team_name = row["home_team"]
        away_team_name = row["away_team"]
        home_present = [
            row["home_present_0"],
            row["home_present_1"],
            row["home_present_2"],
            row["home_present_3"],
        ]
        away_players = [
            row["away_0"],
            row["away_1"],
            row["away_2"],
            row["away_3"],
            row["away_4"],
        ]
        true_missing_player = row["true_missing_player"]

        # Encode basic features
        enc_home_present = [encode_player(p) for p in home_present]
        enc_away = [encode_player(p) for p in away_players]
        enc_home_team = encode_team(home_team_name)
        enc_away_team = encode_team(away_team_name)
        minute = row["starting_min"]

        # For each candidate in the roster of the home team, compute full feature vector
        roster_candidates = team_rosters.get(home_team_name, [])
        for candidate in roster_candidates:
            if candidate not in player_encoder.classes_:
                continue
            enc_candidate = player_encoder.transform([candidate])[0]

            # Compute extra candidate stats using the same helper functions:
            overall_win_rate = get_player_metric(candidate, "overall_win_rate")
            home_win_rate = get_player_metric(candidate, "home_win_rate")
            away_win_rate = get_player_metric(candidate, "away_win_rate")
            avg_start_min = get_player_metric(candidate, "avg_start_min")
            # Average synergy with the present 4 players:
            synergy_vals = [get_synergy(candidate, p) for p in home_present]
            avg_synergy_with_present = (
                sum(synergy_vals) / len(synergy_vals) if synergy_vals else 0.0
            )
            # Average head-to-head vs away players:
            h2h_vals = [get_head2head(candidate, opp) for opp in away_players]
            avg_head2head = sum(h2h_vals) / len(h2h_vals) if h2h_vals else 0.0
            # Average synergy among the present 4 themselves:
            if len(home_present) >= 2:
                synergy_sum = 0.0
                count = 0
                for p1, p2 in combinations(home_present, 2):
                    synergy_sum += get_synergy(p1, p2)
                    count += 1
                present_4_avg_synergy = synergy_sum / count if count > 0 else 0.0
            else:
                present_4_avg_synergy = 0.0

            # Construct feature vector (order must match training)
            feature_vector = (
                enc_home_present
                + enc_away
                + [
                    enc_home_team,
                    enc_away_team,
                    minute,
                    enc_candidate,
                    overall_win_rate,
                    home_win_rate,
                    away_win_rate,
                    avg_start_min,
                    avg_synergy_with_present,
                    avg_head2head,
                    present_4_avg_synergy,
                ]
            )

            X_test_rows.append(feature_vector)
            y_test_truth.append(
                {"true_player": true_missing_player, "candidate": candidate}
            )

    # Convert candidate rows to DataFrame
    X_test = pd.DataFrame(X_test_rows, columns=feature_cols)

    # --- Get Predictions from all models ---
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict_proba(X_test)
    catboost_preds = catboost_model.predict_proba(X_test)

    # For binary classification, extract probability for class 1
    if lgb_preds.ndim == 2 and lgb_preds.shape[1] == 2:
        lgb_preds = lgb_preds[:, 1]
    if xgb_preds.ndim == 2 and xgb_preds.shape[1] == 2:
        xgb_preds = xgb_preds[:, 1]
    if catboost_preds.ndim == 2 and catboost_preds.shape[1] == 2:
        catboost_preds = catboost_preds[:, 1]

    # --- Evaluation Function ---
    def evaluate_predictions(pred_probs):
        index = 0
        correct = 0
        total = 0
        for idx, row in df_test_scenario.iterrows():
            home_team_name = row["home_team"]
            roster_candidates = team_rosters.get(home_team_name, [])
            # Filter out candidates not in training vocabulary
            roster_candidates = [
                c for c in roster_candidates if c in player_encoder.classes_
            ]
            n_candidates = len(roster_candidates)
            segment_probs = pred_probs[index : index + n_candidates]
            segment_truth_info = y_test_truth[index : index + n_candidates]
            index += n_candidates
            best_candidate_idx = segment_probs.argmax()
            best_candidate_name = segment_truth_info[best_candidate_idx]["candidate"]
            true_player = segment_truth_info[0]["true_player"]
            if best_candidate_name == true_player:
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

    with open(f"{output_dir}/model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(
        f"Step 4 complete for {year}: model comparison results saved to model_comparison_results.json"
    )
