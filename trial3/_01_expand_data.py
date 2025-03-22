# _01_expand_data.py
import pandas as pd
import json
import random
import os
from itertools import combinations

# Set random seed for reproducibility
random.seed(42)


def get_time_range_bucket(minutes):
    if minutes < 12:
        return "1_12"
    elif minutes < 24:
        return "12_24"
    elif minutes < 36:
        return "24_36"
    else:
        return "36_48"


def expand_data(year):
    output_dir = f"./{year}"
    # Read the original data where real home_team and away_team are known
    # outcome = 1 => real home_team "won" this segment
    df = pd.read_csv(f"../input-dataset/matchups-{year}.csv")

    # We'll keep only rows where the real home team won (outcome=1).
    df_win = df[df["outcome"] == 1].copy()

    # Build a mapping of team -> set of all players who ever appeared for that team.
    team_rosters = {}

    def add_to_roster(team, player):
        if team not in team_rosters:
            team_rosters[team] = set()
        team_rosters[team].add(player)

    for idx, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        for i in range(5):
            add_to_roster(ht, row[f"home_{i}"])
            add_to_roster(at, row[f"away_{i}"])

    # Load player stats (make sure _00_get_players_stats.py has been run and saved this file)
    stats_file = f"{output_dir}/players_stats.json"
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            players_stats = json.load(f)
    else:
        print(f"Warning: {stats_file} not found. Player stats features will be zeros.")
        players_stats = {}

    # Helper functions to extract stats
    def get_player_metric(player, metric):
        if player in players_stats:
            return players_stats[player].get(metric, 0.0)
        return 0.0

    def get_synergy(p1, p2):
        if p1 in players_stats:
            synergy_dict = players_stats[p1].get("synergy_with_teammates", {})
            return synergy_dict.get(p2, 0.0)
        return 0.0

    def get_head2head(home_p, away_p):
        if home_p in players_stats:
            h2h_dict = players_stats[home_p].get("head2head_against", {})
            return h2h_dict.get(away_p, 0.0)
        return 0.0

    # Build binary_data rows with additional synergy and stat features
    binary_rows = []
    for idx, row in df_win.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        starting_min = row["starting_min"]

        # The actual 5 players who played (winning lineup)
        home_lineup = [row[f"home_{i}"] for i in range(5)]
        away_lineup = [row[f"away_{i}"] for i in range(5)]

        # --- Positive examples ---
        for candidate_player in home_lineup:
            present_players = [p for p in home_lineup if p != candidate_player]
            new_row = {
                "home_team": home_team,
                "away_team": away_team,
                "starting_min": starting_min,
                "candidate_player": candidate_player,
                "label": 1,
            }
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p
            for i, opp in enumerate(away_lineup):
                new_row[f"away_{i}"] = opp

            # --- Add candidate's basic stats ---
            new_row["candidate_overall_win_rate"] = get_player_metric(
                candidate_player, "overall_win_rate"
            )
            new_row["candidate_home_win_rate"] = get_player_metric(
                candidate_player, "home_win_rate"
            )
            new_row["candidate_away_win_rate"] = get_player_metric(
                candidate_player, "away_win_rate"
            )
            new_row["candidate_avg_start_min"] = get_player_metric(
                candidate_player, "avg_start_min"
            )

            # --- Candidate's average synergy with present players ---
            synergy_vals = [get_synergy(candidate_player, p) for p in present_players]
            new_row["candidate_avg_synergy_with_present"] = (
                sum(synergy_vals) / len(synergy_vals) if synergy_vals else 0.0
            )

            # --- Candidate's average head-to-head vs away players ---
            h2h_vals = [get_head2head(candidate_player, opp) for opp in away_lineup]
            new_row["candidate_avg_head2head"] = (
                sum(h2h_vals) / len(h2h_vals) if h2h_vals else 0.0
            )

            # --- Average synergy among the present 4 themselves ---
            if len(present_players) >= 2:
                synergy_sum = 0.0
                count = 0
                for p1, p2 in combinations(present_players, 2):
                    synergy_sum += get_synergy(p1, p2)
                    count += 1
                new_row["present_4_avg_synergy"] = (
                    synergy_sum / count if count > 0 else 0.0
                )
            else:
                new_row["present_4_avg_synergy"] = 0.0

            binary_rows.append(new_row)

        # --- Negative examples ---
        all_candidates_for_team = team_rosters[home_team]
        missing_from_lineup = set(all_candidates_for_team) - set(home_lineup)
        for candidate_player in missing_from_lineup:
            remove_index = random.choice(range(5))
            present_players = [
                p for idx, p in enumerate(home_lineup) if idx != remove_index
            ]
            new_row = {
                "home_team": home_team,
                "away_team": away_team,
                "starting_min": starting_min,
                "candidate_player": candidate_player,
                "label": 0,
            }
            for i, p in enumerate(present_players):
                new_row[f"home_{i}"] = p
            for i, opp in enumerate(away_lineup):
                new_row[f"away_{i}"] = opp

            # --- Add candidate's basic stats ---
            new_row["candidate_overall_win_rate"] = get_player_metric(
                candidate_player, "overall_win_rate"
            )
            new_row["candidate_home_win_rate"] = get_player_metric(
                candidate_player, "home_win_rate"
            )
            new_row["candidate_away_win_rate"] = get_player_metric(
                candidate_player, "away_win_rate"
            )
            new_row["candidate_avg_start_min"] = get_player_metric(
                candidate_player, "avg_start_min"
            )

            # --- Candidate's average synergy with present players ---
            synergy_vals = [get_synergy(candidate_player, p) for p in present_players]
            new_row["candidate_avg_synergy_with_present"] = (
                sum(synergy_vals) / len(synergy_vals) if synergy_vals else 0.0
            )

            # --- Candidate's average head-to-head vs away players ---
            h2h_vals = [get_head2head(candidate_player, opp) for opp in away_lineup]
            new_row["candidate_avg_head2head"] = (
                sum(h2h_vals) / len(h2h_vals) if h2h_vals else 0.0
            )

            # --- Average synergy among the present 4 themselves ---
            if len(present_players) >= 2:
                synergy_sum = 0.0
                count = 0
                for p1, p2 in combinations(present_players, 2):
                    synergy_sum += get_synergy(p1, p2)
                    count += 1
                new_row["present_4_avg_synergy"] = (
                    synergy_sum / count if count > 0 else 0.0
                )
            else:
                new_row["present_4_avg_synergy"] = 0.0

            binary_rows.append(new_row)

    # Convert rows into a DataFrame and save
    df_binary = pd.DataFrame(binary_rows)
    df_binary.to_csv(f"{output_dir}/binary_data.csv", index=False)
    # Save team rosters as JSON (convert sets to lists)
    with open(f"{output_dir}/team_rosters.json", "w") as f:
        json.dump({team: list(players) for team, players in team_rosters.items()}, f)
    print(
        f"Step 1 complete for {year}: Built binary_data.csv with positive/negative samples and added player stats."
    )
