# 00-expand_data.py
import pandas as pd
import json

# Read the original data where real home_team and away_team are known
# outcome = 1 => real home_team "won" this segment
df = pd.read_csv("../input-dataset/matchups-2007.csv")

# We'll keep only rows where the real home team won (outcome=1).
# We only use these for training because our test scenario focuses on the real home team winning.
df_win = df[df["outcome"] == 1].copy()

# Build a mapping of team -> set of all players who ever appeared for that team
# This acts as our "roster" for each team.
team_rosters = {}


def add_to_roster(team, player):
    if team not in team_rosters:
        team_rosters[team] = set()
    team_rosters[team].add(player)


# Collect rosters by scanning all the home columns of df (and away columns if needed).
for idx, row in df.iterrows():
    ht = row["home_team"]
    at = row["away_team"]
    for i in range(5):
        add_to_roster(ht, row[f"home_{i}"])
        add_to_roster(at, row[f"away_{i}"])

# Now build a "binary_data" DataFrame:
# For each row in df_win, we have 5 actual home players. Each of them gets a "label = 1".
# Then for all other players on home_team's roster who are NOT in the lineup, label=0.
# We'll store: 4 "present" home players, 5 away players, 1 candidate, plus "home_team", "away_team", "starting_min", and a label.

binary_rows = []
for idx, row in df_win.iterrows():
    home_team = row["home_team"]
    away_team = row["away_team"]
    starting_min = row["starting_min"]

    # The actual 5 players who played
    home_lineup = [row[f"home_{i}"] for i in range(5)]
    away_lineup = [row[f"away_{i}"] for i in range(5)]

    # We'll iterate over each of the 5 to create "positive" examples
    for candidate_player in home_lineup:
        # The 4 "present" others
        present_players = [p for p in home_lineup if p != candidate_player]
        # Build a row
        new_row = {
            "home_team": home_team,
            "away_team": away_team,
            "starting_min": starting_min,
            "candidate_player": candidate_player,
            "label": 1,  # Because this candidate was truly in the lineup
        }
        # Put the 4 present players in columns
        for i, p in enumerate(present_players):
            new_row[f"home_{i}"] = p
        # Put the 5 away players
        for i, opp in enumerate(away_lineup):
            new_row[f"away_{i}"] = opp

        binary_rows.append(new_row)

    # Now create "negative" examples for each roster player who was NOT in the actual 5
    all_candidates_for_team = team_rosters[
        home_team
    ]  # set of all players for that home_team
    missing_from_lineup = set(all_candidates_for_team) - set(home_lineup)
    for candidate_player in missing_from_lineup:
        new_row = {
            "home_team": home_team,
            "away_team": away_team,
            "starting_min": starting_min,
            "candidate_player": candidate_player,
            "label": 0,
        }
        # The present 4 can be any 4 from the actual 5, but typically we pick exactly the 4 that remain.
        # We'll just pick the "first 4" in the row for consistency.
        # Alternatively, you could create multiple negative examples with different "4-of-5" combos,
        # but let's keep it simple: we'll always pick "home_0..3" from the row, ignoring "home_4".
        # (Pick whichever 4 you like; as long as it's consistent, the model can learn.)
        new_row["home_0"] = row["home_0"]
        new_row["home_1"] = row["home_1"]
        new_row["home_2"] = row["home_2"]
        new_row["home_3"] = row["home_3"]

        # 5 away players
        for i, opp in enumerate(away_lineup):
            new_row[f"away_{i}"] = opp

        binary_rows.append(new_row)

# Convert to DataFrame
df_binary = pd.DataFrame(binary_rows)
df_binary.to_csv("binary_data.csv", index=False)
# Convert sets to lists before saving to JSON
with open("team_rosters.json", "w") as f:
    json.dump({team: list(players) for team, players in team_rosters.items()}, f)
print("Step 0 complete: Built binary_data.csv with positive/negative samples.")
