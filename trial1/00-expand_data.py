# 00-expand_data.py
import pandas as pd
import os
import numpy as np

chunk_size = 100000
expanded_rows = []


def process_chunk(chunk):
    global expanded_rows
    # Loop through each game segment (row)
    for _, row in chunk.iterrows():
        # Depending on outcome, determine which team "won" and use the corresponding lineups
        if row["outcome"] == 1:
            winning_home_team = row["home_team"]
            winning_away_team = row["away_team"]
            winning_home_players = [row[f"home_{i}"] for i in range(5)]
            winning_away_players = [row[f"away_{i}"] for i in range(5)]
        else:
            winning_home_team = row["away_team"]
            winning_away_team = row["home_team"]
            winning_home_players = [row[f"away_{i}"] for i in range(5)]
            winning_away_players = [row[f"home_{i}"] for i in range(5)]

        # For each player in the winning home lineup, remove that player to simulate a missing player
        for removed_player in winning_home_players:
            new_row = {
                "removed_player": removed_player,
                "home_team": winning_home_team,
                "away_team": winning_away_team,
                "starting_min": row["starting_min"],
            }

            # Keep the remaining four home players as features
            present_players = [p for p in winning_home_players if p != removed_player]
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p

            # Include all five away players as features
            for i, opp in enumerate(winning_away_players):
                new_row[f"away_{i}"] = opp

            expanded_rows.append(new_row)

        # Save in chunks to avoid memory issues
        if len(expanded_rows) >= chunk_size:
            df_chunk = pd.DataFrame(expanded_rows)
            df_chunk.to_csv(
                "00-expanded_data.csv",
                mode="a",
                header=not os.path.exists("00-expanded_data.csv"),
                index=False,
            )
            expanded_rows = []


# Define column data types for efficient reading
dtype_dict = {
    "home_team": "category",
    "away_team": "category",
    "starting_min": np.int8,
    "end_min": np.int8,
    "home_0": "category",
    "home_1": "category",
    "home_2": "category",
    "home_3": "category",
    "home_4": "category",
    "away_0": "category",
    "away_1": "category",
    "away_2": "category",
    "away_3": "category",
    "away_4": "category",
    "outcome": np.int8,
}

# Read the input CSV file in chunks and process each one
df_chunks = pd.read_csv(
    "../input-dataset/matchups-2007.csv", chunksize=chunk_size, dtype=dtype_dict
)
for chunk in df_chunks:
    process_chunk(chunk)

# Write out any remaining rows
if expanded_rows:
    pd.DataFrame(expanded_rows).to_csv(
        "00-expanded_data.csv",
        mode="a",
        header=not os.path.exists("00-expanded_data.csv"),
        index=False,
    )

print("Data Expanded and Saved as 00-expanded_data.csv")
