import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load your encoders
with open("player_encoder.pkl", "rb") as f:
    player_encoder = pickle.load(f)

with open("team_encoder.pkl", "rb") as f:
    team_encoder = pickle.load(f)

# Load your rosters for filtering
with open("../input-dataset/team-players.json", "r") as f:
    team_rosters = json.load(f)

# Load the neural network model
model = keras.models.load_model("nn_model.h5")

# Define feature columns
feature_cols = (
    [f"home_{i}" for i in range(4)]
    + [f"away_{i}" for i in range(5)]
    + ["home_team", "away_team", "starting_min"]
)

# Read the test data
test_data = pd.read_csv("02-test_data.csv")
X_test = test_data[feature_cols]
y_test = test_data["removed_player"].values  # true labels (encoded)

correct = 0
total = 0

for i, row in X_test.iterrows():
    # 1) figure out which team we are dealing with
    home_team_encoded = row["home_team"]
    home_team_name = team_encoder.inverse_transform([home_team_encoded])[0]

    # 2) get candidate players for home_team
    candidate_players = team_rosters[home_team_name]

    # Filter those candidates to ones the model actually knows
    candidate_players = [
        p for p in candidate_players 
        if p in player_encoder.classes_  # ensure it exists in the encoder
    ]
    if not candidate_players:
        continue  # skip if no valid candidates

    encoded_candidates = player_encoder.transform(candidate_players)

    # 3) get model predictions (logits or probabilities)
    sample_features = row.values.reshape(1, -1)  # shape (1, input_dim)
    pred_probs = model.predict(sample_features)[0]  # shape [num_classes]

    # 4) restrict to the probabilities of candidate players
    candidate_probs = {cand: pred_probs[cand] for cand in encoded_candidates}

    predicted_label = max(candidate_probs, key=candidate_probs.get)
    true_label = y_test[i]

    if predicted_label == true_label:
        correct += 1
    total += 1

accuracy = correct / total if total else 0
print(f"Filtered NN Accuracy: {accuracy * 100:.2f}%")
