# models/evaluator.py
import os
import pandas as pd
import xgboost as xgb
from config import OUTPUT_BASE_DIR, TEST_DATA_DIR, TARGET_YEARS
from utils.data_utils import load_json, load_encoder, load_dataframe
from utils.feature_utils import build_candidate_features


def evaluate_model() -> dict:
    """
    Evaluates the trained models on the test data.

    Test data is loaded from TEST_DATA_DIR:
      - NBA_test.csv: Contains test samples with a missing home player (indicated by '?')
      - NBA_test_labels.csv: Contains the removed player's name (column "removed_value")

    For each test sample:
      - The season (year) is identified.
      - For years in TARGET_YEARS (2008-2016), models are loaded from data/<year> directory
        These models are trained on previous year's data. For example, the 2008 model is trained on 2007 data.
      - The four present home players are determined (ignoring the '?' among home_0 to home_4).
      - The candidate pool is built from the home team's roster (all players not in the present four).
      - For each candidate, candidate features are computed (using build_candidate_features) and a feature row is constructed.
      - The candidate sample is encoded with the season’s LabelEncoders.
      - The season's XGBoost model is used to predict the probability of the candidate being the correct missing player.
      - The candidate with the highest predicted probability is selected.
      - The prediction is compared to the ground truth from NBA_test_labels.csv.

    Results are aggregated per season and cumulatively, and then saved as evaluation_results.json.
    """

    # Load test data and corresponding ground truth labels.
    test_df = load_dataframe(TEST_DATA_DIR, "NBA_test")
    test_labels_df = load_dataframe(TEST_DATA_DIR, "NBA_test_labels")

    results = {}
    cumulative_correct = 0
    cumulative_incorrect = 0
    cumulative_total = 0

    # Process test samples grouped by season.
    for year in test_df["season"].unique():
        if year not in TARGET_YEARS:
            continue
            
        season_test_df = test_df[test_df["season"] == year].copy()

        # Use the data directory for loading models
        target_dir = os.path.join(OUTPUT_BASE_DIR, "data", str(year))
        
        # If directory doesn't exist, skip this year
        if not os.path.exists(target_dir):
            print(f"Warning: Data directory for year {year} not found. Skipping.")
            continue
        
        # Load model and encoders from the directory
        model = xgb.Booster()
        model_path = os.path.join(target_dir, "best_xgb_model.json")
        if not os.path.exists(model_path):
            print(f"Warning: Model file for year {year} not found at {model_path}. Skipping.")
            continue
            
        model.load_model(model_path)

        player_encoder = load_encoder(target_dir, "player_encoder")
        team_encoder = load_encoder(target_dir, "team_encoder")
        players_stats = load_json(target_dir, "players_stats")
        team_rosters = load_json(target_dir, "team_rosters")

        season_correct = 0
        season_incorrect = 0
        season_total = len(season_test_df)

        # Process each test sample for the season.
        for idx, row in season_test_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            starting_min = row["starting_min"]
            away_lineup = [row[f"away_{i}"] for i in range(5)]

            # Identify the four present home players (ignoring the '?' placeholder)
            present_home_players = []
            for i in range(5):
                player = row[f"home_{i}"]
                if player != "?":
                    present_home_players.append(player)

            # Build the candidate pool: all players in the team's roster that are not already in the present lineup.
            roster = team_rosters.get(home_team, [])
            candidate_pool = list(set(roster) - set(present_home_players))

            if not candidate_pool:
                # If no candidate is available, mark as incorrect and move to next sample.
                season_incorrect += 1
                continue

            best_prob = -1
            best_candidate = None

            # For each candidate, create a feature sample.
            for candidate in candidate_pool:
                sample_features = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "starting_min": starting_min,
                    "candidate_player": candidate,
                }
                # Use the order of home players as they appear in the test sample (excluding the '?')
                for i in range(4):
                    sample_features[f"home_{i}"] = present_home_players[i]
                for i in range(5):
                    sample_features[f"away_{i}"] = away_lineup[i]

                # Compute candidate-specific features using player statistics.
                sample_features.update(
                    build_candidate_features(
                        candidate, present_home_players, away_lineup, players_stats
                    )
                )

                # Create a one-row DataFrame.
                sample_df = pd.DataFrame([sample_features])

                # Encode player columns: candidate_player, home_0 ... home_3, away_0 ... away_4.
                player_cols = (
                    ["candidate_player"]
                    + [f"home_{i}" for i in range(4)]
                    + [f"away_{i}" for i in range(5)]
                )
                
                # Handle players not in encoder
                for col in player_cols:
                    if sample_df.loc[0, col] not in player_encoder.classes_:
                        sample_df[col] = -1
                        continue
                    sample_df[col] = player_encoder.transform(sample_df[col])

                # Encode team columns
                for col in ["home_team", "away_team"]:
                    if sample_df.loc[0, col] not in team_encoder.classes_:
                        sample_df[col] = -1
                        continue
                    sample_df[col] = team_encoder.transform(sample_df[col])

                # Reorder columns as expected by the model.
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
                sample_df = sample_df[feature_cols]

                # Create a DMatrix and predict the probability.
                dmatrix = xgb.DMatrix(sample_df)
                prob = model.predict(dmatrix)[0]

                if prob > best_prob:
                    best_prob = prob
                    best_candidate = candidate

            # Compare the predicted candidate with the ground truth label.
            actual_candidate = test_labels_df.loc[idx, "removed_value"]
            if best_candidate == actual_candidate:
                season_correct += 1
            else:
                season_incorrect += 1

        season_accuracy = season_correct / season_total if season_total > 0 else 0.0
        results[str(year)] = {
            "accuracy": season_accuracy,
            "correct": season_correct,
            "incorrect": season_incorrect,
            "total": season_total,
        }

        cumulative_correct += season_correct
        cumulative_incorrect += season_incorrect
        cumulative_total += season_total

    cumulative_accuracy = (
        cumulative_correct / cumulative_total if cumulative_total > 0 else 0.0
    )
    results["cumulative"] = {
        "accuracy": cumulative_accuracy,
        "correct": cumulative_correct,
        "incorrect": cumulative_incorrect,
        "total": cumulative_total,
    }

    return results
