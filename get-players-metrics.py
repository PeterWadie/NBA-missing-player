import pandas as pd
import json
from itertools import combinations

#############################
#  HELPER FUNCTIONS
#############################

def get_time_range_bucket(minutes):
    """
    Returns one of four labels based on the starting_min of the segment:
      - '1_12'   for 0 <= minutes < 12
      - '12_24'  for 12 <= minutes < 24
      - '24_36'  for 24 <= minutes < 36
      - '36_48'  for 36 <= minutes <= 48
    Adjust these thresholds as needed.
    """
    if minutes < 12:
        return "1_12"
    elif minutes < 24:
        return "12_24"
    elif minutes < 36:
        return "24_36"
    else:
        return "36_48"

def initialize_player_stats():
    """Return a fresh stats dictionary for a new player."""
    return {
        "total_segments": 0,
        "total_wins": 0,
        "sum_start_min": 0.0,  # We'll use this to get avg_start_min later

        # Home vs Away splits
        "home_segments": 0,
        "home_wins": 0,
        "away_segments": 0,
        "away_wins": 0,

        # Home time-range buckets
        "home_segments_by_range": {
            "1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0
        },
        "home_wins_by_range": {
            "1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0
        },

        # Away time-range buckets
        "away_segments_by_range": {
            "1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0
        },
        "away_wins_by_range": {
            "1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0
        },
    }

def synergy_key(p1, p2):
    """
    Create a unique key for a pair of players on the same side,
    ensuring the order doesn't matter (tuple is always sorted).
    """
    return tuple(sorted([p1, p2]))

def head2head_key(home_player, away_player):
    """
    Unique key for a home-vs-away pair.
    We'll keep it as (home_player, away_player).
    """
    return (home_player, away_player)


#############################
#  MAIN SCRIPT
#############################

def main():
    # 1. Load the CSV
    df = pd.read_csv("./input-dataset/combined.csv")

    # 2. Dictionaries to store metrics
    player_stats = {}         # Individual stats keyed by player
    synergy_stats = {}        # Same-side synergy keyed by (p1, p2)
    head2head_stats = {}      # Opposite-side matchups keyed by (home_p, away_p)

    def update_player_stats(player_name, is_home, outcome, start_min):
        """
        Update the stats for a single player:
        - is_home: True if the player was on the home side in this segment
        - outcome: 1 if home side won, -1 if away side won
        """
        if player_name not in player_stats:
            player_stats[player_name] = initialize_player_stats()

        stats = player_stats[player_name]
        stats["total_segments"] += 1
        stats["sum_start_min"] += float(start_min)

        # Determine which bucket (e.g., '1_12', '12_24', etc.)
        bucket = get_time_range_bucket(start_min)

        if is_home:
            # Player is on home side
            stats["home_segments"] += 1
            stats["home_segments_by_range"][bucket] += 1

            # If outcome=1 => home won => player "won"
            if outcome == 1:
                stats["total_wins"] += 1
                stats["home_wins"] += 1
                stats["home_wins_by_range"][bucket] += 1
        else:
            # Player is on away side
            stats["away_segments"] += 1
            stats["away_segments_by_range"][bucket] += 1

            # If outcome=-1 => away won => player "won"
            if outcome == -1:
                stats["total_wins"] += 1
                stats["away_wins"] += 1
                stats["away_wins_by_range"][bucket] += 1

    # 3. Process each row
    for idx, row in df.iterrows():
        outcome = row["outcome"]  # 1 for home win, -1 for away win
        start_min = row["starting_min"]

        # Home and away players
        home_players = [
            row["home_0"], 
            row["home_1"], 
            row["home_2"], 
            row["home_3"], 
            row["home_4"]
        ]
        away_players = [
            row["away_0"], 
            row["away_1"], 
            row["away_2"], 
            row["away_3"], 
            row["away_4"]
        ]

        # Update player-level stats
        for h_player in home_players:
            update_player_stats(h_player, is_home=True, outcome=outcome, start_min=start_min)

        for a_player in away_players:
            update_player_stats(a_player, is_home=False, outcome=outcome, start_min=start_min)

        # Update synergy (same-side pairwise)
        # For the home side
        for p1, p2 in combinations(home_players, 2):
            pair_key = synergy_key(p1, p2)
            if pair_key not in synergy_stats:
                synergy_stats[pair_key] = {"segments": 0, "wins": 0}
            synergy_stats[pair_key]["segments"] += 1
            if outcome == 1:  # home won
                synergy_stats[pair_key]["wins"] += 1

        # For the away side
        for p1, p2 in combinations(away_players, 2):
            pair_key = synergy_key(p1, p2)
            if pair_key not in synergy_stats:
                synergy_stats[pair_key] = {"segments": 0, "wins": 0}
            synergy_stats[pair_key]["segments"] += 1
            if outcome == -1:  # away won
                synergy_stats[pair_key]["wins"] += 1

        # Update head-to-head (home vs away)
        for h in home_players:
            for a in away_players:
                pair_key = head2head_key(h, a)
                if pair_key not in head2head_stats:
                    head2head_stats[pair_key] = {"segments": 0, "wins": 0}
                head2head_stats[pair_key]["segments"] += 1
                if outcome == 1:  # home win
                    head2head_stats[pair_key]["wins"] += 1

    # 4. Compute final metrics for each player
    for player_name, stats in player_stats.items():
        total_segments = stats["total_segments"]
        total_wins = stats["total_wins"]

        # Overall Win Rate
        stats["overall_win_rate"] = (
            total_wins / total_segments if total_segments > 0 else 0.0
        )

        # Home/Away Win Rates
        if stats["home_segments"] > 0:
            stats["home_win_rate"] = stats["home_wins"] / stats["home_segments"]
        else:
            stats["home_win_rate"] = 0.0

        if stats["away_segments"] > 0:
            stats["away_win_rate"] = stats["away_wins"] / stats["away_segments"]
        else:
            stats["away_win_rate"] = 0.0

        # Average Starting Minute
        if total_segments > 0:
            stats["avg_start_min"] = stats["sum_start_min"] / total_segments
        else:
            stats["avg_start_min"] = 0.0

        # Now compute the new time-range stats
        # We'll produce home_win_rate_1_12, home_win_rate_12_24, home_win_rate_24_36, home_win_rate_36_48,
        # and similarly for away.
        for rng in ["1_12", "12_24", "24_36", "36_48"]:
            home_segs = stats["home_segments_by_range"][rng]
            home_wins = stats["home_wins_by_range"][rng]
            away_segs = stats["away_segments_by_range"][rng]
            away_wins = stats["away_wins_by_range"][rng]

            # e.g., home_win_rate_1_12
            stats[f"home_win_rate_{rng}"] = (home_wins / home_segs) if home_segs > 0 else 0.0
            # e.g., away_win_rate_1_12
            stats[f"away_win_rate_{rng}"] = (away_wins / away_segs) if away_segs > 0 else 0.0

        # Remove intermediate fields we don't need in final JSON:
        del stats["sum_start_min"]
        del stats["home_segments_by_range"]
        del stats["home_wins_by_range"]
        del stats["away_segments_by_range"]
        del stats["away_wins_by_range"]

    # 5. Build synergy metrics per player
    for player_name in player_stats.keys():
        player_stats[player_name]["synergy_with_teammates"] = {}
        player_stats[player_name]["head2head_against"] = {}

    # Compute synergy (same-side) win rate and attach to each player's stats
    for (p1, p2), record in synergy_stats.items():
        segs = record["segments"]
        wins = record["wins"]
        synergy_rate = wins / segs if segs > 0 else 0.0

        # Attach synergy to both p1 and p2
        if p1 in player_stats:
            player_stats[p1]["synergy_with_teammates"][p2] = synergy_rate
        if p2 in player_stats:
            player_stats[p2]["synergy_with_teammates"][p1] = synergy_rate

    # Compute head-to-head (home vs away) from the home player's perspective
    for (home_p, away_p), record in head2head_stats.items():
        segs = record["segments"]
        wins = record["wins"]
        h2h_rate = wins / segs if segs > 0 else 0.0

        if home_p in player_stats:
            player_stats[home_p]["head2head_against"][away_p] = h2h_rate

    # 6. Save to JSON
    with open("./input-dataset/player-stats.json", "w") as f:
        json.dump(player_stats, f, indent=2)

if __name__ == "__main__":
    main()
