# models/encoder.py
import os
import random
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.feature_utils import build_candidate_features


def expand_binary_data(
    df: pd.DataFrame, players_stats: dict, team_rosters: dict, random_seed: int = 42
) -> pd.DataFrame:
    """
    Expands the matchup data into binary samples (positive and negative) with candidate features.
    """
    random.seed(random_seed)
    binary_rows = []

    # Use only rows where the home team won (outcome == 1)
    df_win = df[df["outcome"] == 1].copy()

    for _, row in df_win.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        starting_min = row["starting_min"]
        home_lineup = [row[f"home_{i}"] for i in range(5)]
        away_lineup = [row[f"away_{i}"] for i in range(5)]

        # Positive examples: each player in the home lineup is considered a candidate.
        for candidate in home_lineup:
            present_players = [p for p in home_lineup if p != candidate]
            new_row = {
                "home_team": home_team,
                "away_team": away_team,
                "starting_min": starting_min,
                "candidate_player": candidate,
                "label": 1,
            }
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p
            for i, opp in enumerate(away_lineup):
                new_row[f"away_{i}"] = opp
            # Build candidate features using the helper function
            new_row.update(
                build_candidate_features(
                    candidate, present_players, away_lineup, players_stats
                )
            )
            binary_rows.append(new_row)

        # Negative examples: candidates who are in the team roster but not in the current lineup.
        roster_candidates = set(team_rosters.get(home_team, []))
        missing_candidates = roster_candidates - set(home_lineup)
        for candidate in missing_candidates:
            # Randomly remove one player from the current lineup for negative example.
            remove_index = random.choice(range(5))
            present_players = [
                p for i, p in enumerate(home_lineup) if i != remove_index
            ]
            new_row = {
                "home_team": home_team,
                "away_team": away_team,
                "starting_min": starting_min,
                "candidate_player": candidate,
                "label": 0,
            }
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p
            for i, opp in enumerate(away_lineup):
                new_row[f"away_{i}"] = opp
            new_row.update(
                build_candidate_features(
                    candidate, present_players, away_lineup, players_stats
                )
            )
            binary_rows.append(new_row)

    return pd.DataFrame(binary_rows)


def encode_binary_data(df_binary: pd.DataFrame, output_dir: str):
    """
    Encodes player and team columns using LabelEncoder.
    Saves the encoders and the encoded DataFrame.
    Returns (encoded DataFrame, player_encoder, team_encoder).
    """
    home_present_cols = [f"home_{i}" for i in range(4)]
    away_cols = [f"away_{i}" for i in range(5)]
    player_cols = home_present_cols + away_cols + ["candidate_player"]

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

    for col in player_cols:
        df_binary[col] = player_encoder.transform(df_binary[col])
    df_binary["home_team"] = team_encoder.transform(df_binary["home_team"])
    df_binary["away_team"] = team_encoder.transform(df_binary["away_team"])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "player_encoder.pkl"), "wb") as f:
        pickle.dump(player_encoder, f)
    with open(os.path.join(output_dir, "team_encoder.pkl"), "wb") as f:
        pickle.dump(team_encoder, f)

    df_binary.to_csv(os.path.join(output_dir, "encoded_binary_data.csv"), index=False)
    return df_binary, player_encoder, team_encoder
