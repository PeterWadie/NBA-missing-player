# 01-encode_data.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

chunk_size = 100000
# Feature columns: four home players, five away players, and additional features
feature_cols = [f"home_{i}" for i in range(4)] + [f"away_{i}" for i in range(5)] + ["home_team", "away_team", "starting_min"]

# Columns with player names (features + target)
player_cols = [f"home_{i}" for i in range(4)] + [f"away_{i}" for i in range(5)] + ["removed_player"]

all_players = set()
all_teams = set()

# Collect all unique players and teams for encoding
for chunk in pd.read_csv("00-expanded_data.csv", chunksize=chunk_size):
    for col in player_cols:
        all_players.update(chunk[col].unique())
    all_teams.update(chunk["home_team"].unique())
    all_teams.update(chunk["away_team"].unique())

# Create and fit the LabelEncoder for players
player_encoder = LabelEncoder()
player_encoder.fit(list(all_players))

# Save the player encoder for later use
with open("player_encoder.pkl", "wb") as f:
    pickle.dump(player_encoder, f)

# Create and fit the LabelEncoder for teams
team_encoder = LabelEncoder()
team_encoder.fit(list(all_teams))

# Save the team encoder for later use
with open("team_encoder.pkl", "wb") as f:
    pickle.dump(team_encoder, f)

# Process and encode the data in chunks
encoded_chunks = []
for chunk in pd.read_csv("00-expanded_data.csv", chunksize=chunk_size):
    for col in player_cols:
        chunk[col] = player_encoder.transform(chunk[col]).astype(np.int16)
    # Encode team columns
    chunk["home_team"] = team_encoder.transform(chunk["home_team"]).astype(np.int16)
    chunk["away_team"] = team_encoder.transform(chunk["away_team"]).astype(np.int16)
    # starting_min is numeric, so leave as is
    encoded_chunks.append(chunk)

df_encoded = pd.concat(encoded_chunks, ignore_index=True)
df_encoded.to_csv("01-encoded_data.csv", index=False)
print("Data Encoded and Saved as 01-encoded_data.csv")

# Split into training and test sets (80% train, 20% test)
X = df_encoded[feature_cols]
y = df_encoded["removed_player"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for later evaluation
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv("02-test_data.csv", index=False)
print("Test Data Saved as 02-test_data.csv")
