# 03-evaluate_models.py
import pickle
import json
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Load encoders & team rosters
with open("player_encoder.pkl", "rb") as f:
    player_encoder = pickle.load(f)
with open("team_encoder.pkl", "rb") as f:
    team_encoder = pickle.load(f)
with open("team_rosters.json", "r") as f:
    team_rosters = json.load(f)

# Load trained models
lgb_model = lgb.Booster(model_file="best_lgb_model.txt")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("best_xgb_model.json")
catboost_model = cb.CatBoostClassifier()
catboost_model.load_model("best_cb_model.cbm")

# Load original dataset for test scenario
df = pd.read_csv("../input-dataset/matchups-2007.csv")

# Define feature columns
feature_cols = (
    [f"home_{i}" for i in range(4)]
    + [f"away_{i}" for i in range(5)]
    + ["home_team", "away_team", "starting_min", "candidate_player"]
)

# We'll build a test set focusing on real home wins again OR your own scenario:
# For instance, pick segments where the real home team is known to have won,
# but we only keep 4 players from the 5. Let's call that the "actual test scenario."

test_rows = []
for idx, row in df.iterrows():
    # Suppose you want to test only those rows where outcome=1 (home team won).
    # Then you pretend you only know 4 of the 5 home players, and you want the model to pick the 5th.
    if row["outcome"] == 1:
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_lineup = [row[f"home_{i}"] for i in range(5)]
        away_lineup = [row[f"away_{i}"] for i in range(5)]

        # Let's say we remove the last home player (home_4) and keep the first four as "present."
        # The correct missing player is home_4. You could do random removal if you like.
        missing_player = home_lineup[4]
        present_4 = home_lineup[:4]

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


# Encoding functions
def encode_player(name):
    return (
        player_encoder.transform([name])[0] if name in player_encoder.classes_ else -1
    )


def encode_team(name):
    return team_encoder.transform([name])[0] if name in team_encoder.classes_ else -1


# Prepare test dataset
X_test_rows = []
y_test_truth = []

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

    # Encode the known 4 home players and 5 away
    enc_home_present = [encode_player(p) for p in home_present]
    enc_away = [encode_player(p) for p in away_players]
    enc_home_team = encode_team(home_team_name)
    enc_away_team = encode_team(away_team_name)
    minute = row["starting_min"]

    # Build a candidate row for each possible player in the home_team's roster
    # (as derived from the dictionary we built or your own data).
    # For simplicity, let's rely on the same "team_rosters" approach from training:
    # We'll rebuild it or keep it in memory:
    roster_candidates = team_rosters.get(home_team_name, set())

    # We build a feature row for each candidate. Then we'll pick the one with highest prob.
    for candidate in roster_candidates:
        # We'll encode candidate
        if candidate not in player_encoder.classes_:
            # skip if it's not in the training vocabulary
            continue

        enc_candidate = player_encoder.transform([candidate])[0]

        # Construct the feature vector in the same order as training
        feature_vector = (
            enc_home_present
            + enc_away
            + [enc_home_team, enc_away_team, minute, enc_candidate]
        )

        X_test_rows.append(feature_vector)
        # We'll store the "true missing player" to check correctness
        # in a parallel array, along with the candidate name.
        y_test_truth.append(
            {"true_player": true_missing_player, "candidate": candidate}
        )

# Convert X_test_rows to a DataFrame for prediction
X_test = pd.DataFrame(X_test_rows, columns=feature_cols)

# Predict probabilities with the models
lgb_preds = lgb_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
catboost_preds = catboost_model.predict(X_test)

# Convert predictions to probabilities if necessary
# (LightGBM binary -> 2 columns usually: [prob_of_class0, prob_of_class1])
# So we might do pred_probs[:,1] if it's a 2-column output.
if lgb_preds.ndim == 2 and lgb_preds.shape[1] == 2:
    # the second column is probability of label=1
    lgb_preds = lgb_preds[:, 1]
if xgb_preds.ndim == 2 and xgb_preds.shape[1] == 2:
    xgb_preds = xgb_preds[:, 1]
if catboost_preds.ndim == 2 and catboost_preds.shape[1] == 2:
    catboost_preds = catboost_preds[:, 1]

# Now we have a list of probabilities, one for each (segment, candidate).
# We'll reconstruct which candidate belongs to which segment, pick the best candidate, and measure accuracy.

# Evaluate accuracy
def evaluate_predictions(pred_probs, model_name):
    index = 0
    correct = 0
    total = 0

    # We'll regroup by each segment. For each segment, we have len(roster_candidates) rows.
    for idx, row in df_test_scenario.iterrows():
        home_team_name = row["home_team"]
        roster_candidates = list(team_rosters.get(home_team_name, []))
        # Filter out those not in player_encoder.classes_ if needed
        roster_candidates = [
            c for c in roster_candidates if c in player_encoder.classes_
        ]

        # We'll slice out a chunk of pred_probs for this segment
        n_candidates = len(roster_candidates)
        segment_probs = pred_probs[index : index + n_candidates]
        segment_truth_info = y_test_truth[index : index + n_candidates]
        index += n_candidates

        # Identify the candidate with the highest probability
        best_candidate_idx = segment_probs.argmax()
        best_candidate_name = segment_truth_info[best_candidate_idx]["candidate"]
        # The true missing player for this entire segment is the same in all y_test_truth entries
        true_player = segment_truth_info[0]["true_player"]

        if best_candidate_name == true_player:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


# Compute and compare accuracy for each model
lgb_accuracy = evaluate_predictions(lgb_preds, "LightGBM")
xgb_accuracy = evaluate_predictions(xgb_preds, "XGBoost")
catboost_accuracy = evaluate_predictions(catboost_preds, "CatBoost")

# Display model comparison
results = {
    "LightGBM Accuracy": f"{lgb_accuracy*100:.2f}%",
    "XGBoost Accuracy": f"{xgb_accuracy*100:.2f}%",
    "CatBoost Accuracy": f"{catboost_accuracy*100:.2f}%",
}

print("\n📊 Model Performance Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc}")
