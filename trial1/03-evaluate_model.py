# 03-evaluate_model.py
import json
import pickle
import pandas as pd
import lightgbm as lgb

with open("../input-dataset/team-players.json", "r") as f:
    team_rosters = json.load(f)

with open("player_encoder.pkl", "rb") as f:
    player_encoder = pickle.load(f)

with open("team_encoder.pkl", "rb") as f:
    team_encoder = pickle.load(f)

# Define feature columns: four home players, five away players, plus the extra features
feature_cols = [f"home_{i}" for i in range(4)] + [f"away_{i}" for i in range(5)] + ["home_team", "away_team", "starting_min"]

test_data = pd.read_csv("02-test_data.csv")
X_test = test_data[feature_cols]
y_test = test_data["removed_player"]

model = lgb.Booster(model_file="lightgbm_model.txt")

correct = 0
total = 0

# Iterate over each test sample
for idx, row in X_test.iterrows():
    # Directly retrieve the encoded home team and decode it
    home_team_encoded = row["home_team"]
    home_team_name = team_encoder.inverse_transform([home_team_encoded])[0]
    
    # Retrieve the candidate player names for this team and filter out any players not in the encoder
    candidate_players = team_rosters[home_team_name]
    candidate_players = [player for player in candidate_players if player in player_encoder.classes_]
    
    # If no valid candidate players are found, skip this sample
    if not candidate_players:
        continue
    
    encoded_candidates = player_encoder.transform(candidate_players)
    
    # Get prediction probabilities for this sample (reshape for a single sample)
    sample_features = row.values.reshape(1, -1)
    pred_probs = model.predict(sample_features)[0]  # shape: [num_classes]
    
    # Restrict predictions to the home team's candidates
    candidate_probs = {candidate: pred_probs[candidate] for candidate in encoded_candidates}
    
    # Select the candidate with the highest probability
    predicted_label = max(candidate_probs, key=candidate_probs.get)
    
    # Compare the filtered prediction with the true label
    true_label = y_test.iloc[idx]
    if predicted_label == true_label:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f"Filtered Model Accuracy: {accuracy * 100:.2f}%")
