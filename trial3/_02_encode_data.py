# _02_encode_data.py
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def encode_data(year):
    output_dir = f"./{year}"
    df_binary = pd.read_csv(f"{output_dir}/binary_data.csv")

    # Identify columns with player names
    home_present_cols = [f"home_{i}" for i in range(4)]
    away_cols = [f"away_{i}" for i in range(5)]
    player_cols = home_present_cols + away_cols + ["candidate_player"]

    # Build sets of unique players and unique teams
    all_players = set()
    all_teams = set()

    for col in player_cols:
        all_players.update(df_binary[col].unique())
    all_teams.update(df_binary["home_team"].unique())
    all_teams.update(df_binary["away_team"].unique())

    all_players = list(all_players)
    all_teams = list(all_teams)

    player_encoder = LabelEncoder()
    team_encoder = LabelEncoder()

    player_encoder.fit(all_players)
    team_encoder.fit(all_teams)

    # Encode columns in df_binary
    for col in player_cols:
        df_binary[col] = player_encoder.transform(df_binary[col])
    df_binary["home_team"] = team_encoder.transform(df_binary["home_team"])
    df_binary["away_team"] = team_encoder.transform(df_binary["away_team"])

    # Save encoders
    with open(f"{output_dir}/player_encoder.pkl", "wb") as f:
        pickle.dump(player_encoder, f)
    with open(f"{output_dir}/team_encoder.pkl", "wb") as f:
        pickle.dump(team_encoder, f)

    df_binary.to_csv(f"{output_dir}/encoded_binary_data.csv", index=False)
    print(f"Step 1 complete for {year}: Encoded data saved to encoded_binary_data.csv")
