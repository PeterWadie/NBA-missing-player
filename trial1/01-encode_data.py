# 01-encode_data.py
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Feature columns: four home players, five away players, and additional features
feature_cols = (
    [f"home_{i}" for i in range(4)]
    + [f"away_{i}" for i in range(5)]
    + ["home_team", "away_team", "starting_min"]
)

# Columns with player names (features + target)
player_cols = (
    [f"home_{i}" for i in range(4)]
    + [f"away_{i}" for i in range(5)]
    + ["removed_player"]
)

# Load the entire dataset
df = pd.read_csv("00-expanded_data.csv")

# Collect all unique players and teams for encoding
all_players = set(df[player_cols].values.ravel())
all_teams = set(df["home_team"].unique()).union(set(df["away_team"].unique()))

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

# Encode the dataset
for col in player_cols:
    df[col] = player_encoder.transform(df[col])

# Encode team columns
df["home_team"] = team_encoder.transform(df["home_team"])
df["away_team"] = team_encoder.transform(df["away_team"])

# Save the encoded data
df.to_csv("01-encoded_data.csv", index=False)
print("Data Encoded and Saved as 01-encoded_data.csv")

# Split into training and test sets (80% train, 20% test)
X = df[feature_cols]
y = df["removed_player"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save test data for later evaluation
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv("02-test_data.csv", index=False)
print("Test Data Saved as 02-test_data.csv")
