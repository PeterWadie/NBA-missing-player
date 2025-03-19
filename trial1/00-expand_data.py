import pandas as pd

# Load the entire dataset
df = pd.read_csv( "../input-dataset/matchups-2007.csv")

expanded_rows = []

# Process each row
for _, row in df.iterrows():
    # Determine winning team and players
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

    # Remove one player at a time and store new rows
    for removed_player in winning_home_players:
        new_row = {
            "removed_player": removed_player,
            "home_team": winning_home_team,
            "away_team": winning_away_team,
            "starting_min": row["starting_min"],
        }

        # Keep the remaining four home players
        present_players = [p for p in winning_home_players if p != removed_player]
        for i, p in enumerate(present_players):
            new_row[f"home_{i}"] = p

        # Include all five away players
        for i, opp in enumerate(winning_away_players):
            new_row[f"away_{i}"] = opp

        expanded_rows.append(new_row)

# Convert to DataFrame and save to CSV
pd.DataFrame(expanded_rows).to_csv("00-expanded_data.csv", index=False)
print("Data Expanded and Saved as 00-expanded_data.csv")
