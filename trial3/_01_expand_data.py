# _01_expand_data.py
import pandas as pd
import json
import random

# Set random seed for reproducibility
random.seed(42)


def expand_data(year):
    output_dir = f"./{year}"
    # Read the original data where real home_team and away_team are known
    # outcome = 1 => real home_team "won" this segment
    df = pd.read_csv(f"../input-dataset/matchups-{year}.csv")

    # We'll keep only rows where the real home team won (outcome=1).
    # We only use these for training because our test scenario focuses on the real home team winning.
    df_win = df[df["outcome"] == 1].copy()

    # Build a mapping of team -> set of all players who ever appeared for that team.
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

    # Build a "binary_data" DataFrame:
    # For each row in df_win, we have 5 actual home players.
    # For positive examples, we simulate the removal of each actual player (the remaining 4 are used as features).
    # For negative examples, for every candidate in the team's roster that did not appear,
    # we randomly remove one player from the winning lineup (thus simulating a random missing spot).
    binary_rows = []
    for idx, row in df_win.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        starting_min = row["starting_min"]

        # The actual 5 players who played (winning lineup)
        home_lineup = [row[f"home_{i}"] for i in range(5)]
        away_lineup = [row[f"away_{i}"] for i in range(5)]

        # --- Positive examples ---
        # For each candidate in the winning lineup, simulate a removal.
        for candidate_player in home_lineup:
            # The present 4 are the winning lineup minus the candidate.
            present_players = [p for p in home_lineup if p != candidate_player]
            new_row = {
                "home_team": home_team,
                "away_team": away_team,
                "starting_min": starting_min,
                "candidate_player": candidate_player,
                "label": 1,  # candidate actually played
            }
            # Save the 4 present players in columns home_0 .. home_3
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p
            # Save the 5 away players
            for i, opp in enumerate(away_lineup):
                new_row[f"away_{i}"] = opp

            binary_rows.append(new_row)

        # --- Negative examples ---
        # For candidates that did NOT play in the winning lineup,
        # simulate a scenario by randomly removing one of the winning players.
        all_candidates_for_team = team_rosters[
            home_team
        ]  # set of all players for that home_team
        missing_from_lineup = set(all_candidates_for_team) - set(home_lineup)
        for candidate_player in missing_from_lineup:
            # Randomly choose one of the 5 winning players to "remove"
            remove_index = random.choice(range(5))
            present_players = [
                p for idx, p in enumerate(home_lineup) if idx != remove_index
            ]
            new_row = {
                "home_team": home_team,
                "away_team": away_team,
                "starting_min": starting_min,
                "candidate_player": candidate_player,
                "label": 0,  # candidate did not play
            }
            # Save the randomly selected 4 present players
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p
            # Save the 5 away players
            for i, opp in enumerate(away_lineup):
                new_row[f"away_{i}"] = opp

            binary_rows.append(new_row)

    # Convert the collected rows into a DataFrame and save
    df_binary = pd.DataFrame(binary_rows)
    df_binary.to_csv(f"{output_dir}/binary_data.csv", index=False)
    # Save team rosters as JSON (convert sets to lists)
    with open(f"{output_dir}/team_rosters.json", "w") as f:
        json.dump({team: list(players) for team, players in team_rosters.items()}, f)
    print(
        f"Step 0 complete for {year}: Built binary_data.csv with positive/negative samples."
    )
