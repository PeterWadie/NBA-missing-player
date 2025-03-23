# utils/stats_utils.py
import os
import json
from itertools import combinations


def get_time_range_bucket(minutes: float) -> str:
    """
    Returns one of four labels based on the starting minute.
    """
    if minutes < 12:
        return "1_12"
    elif minutes < 24:
        return "12_24"
    elif minutes < 36:
        return "24_36"
    else:
        return "36_48"


def initialize_player_stats() -> dict:
    """
    Returns a fresh stats dictionary for a new player.
    """
    return {
        "total_segments": 0,
        "total_wins": 0,
        "sum_start_min": 0.0,
        # Home vs Away splits
        "home_segments": 0,
        "home_wins": 0,
        "away_segments": 0,
        "away_wins": 0,
        # Home time-range buckets
        "home_segments_by_range": {"1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0},
        "home_wins_by_range": {"1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0},
        # Away time-range buckets
        "away_segments_by_range": {"1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0},
        "away_wins_by_range": {"1_12": 0, "12_24": 0, "24_36": 0, "36_48": 0},
    }


def synergy_key(p1: str, p2: str) -> tuple:
    """
    Returns a sorted tuple key for two players (order independent).
    """
    return tuple(sorted([p1, p2]))


def head2head_key(home_player: str, away_player: str) -> tuple:
    """
    Returns a key for a home-vs-away player pair.
    """
    return (home_player, away_player)


def update_player_stats(
    player_stats: dict, player: str, is_home: bool, outcome: int, start_min: float
):
    """
    Updates the stats for a single player.
    """
    if player not in player_stats:
        player_stats[player] = initialize_player_stats()

    stats = player_stats[player]
    stats["total_segments"] += 1
    stats["sum_start_min"] += float(start_min)
    bucket = get_time_range_bucket(start_min)
    if is_home:
        stats["home_segments"] += 1
        stats["home_segments_by_range"][bucket] += 1
        if outcome == 1:
            stats["total_wins"] += 1
            stats["home_wins"] += 1
            stats["home_wins_by_range"][bucket] += 1
    else:
        stats["away_segments"] += 1
        stats["away_segments_by_range"][bucket] += 1
        if outcome == -1:
            stats["total_wins"] += 1
            stats["away_wins"] += 1
            stats["away_wins_by_range"][bucket] += 1


def compute_player_stats(df) -> dict:
    """
    Processes the matchup DataFrame to compute individual player stats,
    same-side synergy, and head-to-head metrics.
    Returns a dictionary keyed by player.
    """
    player_stats = {}
    synergy_stats = {}
    head2head_stats = {}

    for _, row in df.iterrows():
        outcome = row["outcome"]
        start_min = row["starting_min"]
        home_players = [row[f"home_{i}"] for i in range(5)]
        away_players = [row[f"away_{i}"] for i in range(5)]

        # Update player-level stats
        for player in home_players:
            update_player_stats(player_stats, player, True, outcome, start_min)
        for player in away_players:
            update_player_stats(player_stats, player, False, outcome, start_min)

        # Update synergy (same-side pairs)
        for p1, p2 in combinations(home_players, 2):
            key = synergy_key(p1, p2)
            synergy_stats.setdefault(key, {"segments": 0, "wins": 0})
            synergy_stats[key]["segments"] += 1
            if outcome == 1:
                synergy_stats[key]["wins"] += 1
        for p1, p2 in combinations(away_players, 2):
            key = synergy_key(p1, p2)
            synergy_stats.setdefault(key, {"segments": 0, "wins": 0})
            synergy_stats[key]["segments"] += 1
            if outcome == -1:
                synergy_stats[key]["wins"] += 1

        # Update head-to-head (home vs away)
        for h in home_players:
            for a in away_players:
                key = head2head_key(h, a)
                head2head_stats.setdefault(key, {"segments": 0, "wins": 0})
                head2head_stats[key]["segments"] += 1
                if outcome == 1:
                    head2head_stats[key]["wins"] += 1

    # Compute final metrics for each player
    for player, stats in player_stats.items():
        total_segments = stats["total_segments"]
        total_wins = stats["total_wins"]
        stats["overall_win_rate"] = (
            total_wins / total_segments if total_segments > 0 else 0.0
        )
        stats["home_win_rate"] = (
            stats["home_wins"] / stats["home_segments"]
            if stats["home_segments"] > 0
            else 0.0
        )
        stats["away_win_rate"] = (
            stats["away_wins"] / stats["away_segments"]
            if stats["away_segments"] > 0
            else 0.0
        )
        stats["avg_start_min"] = (
            stats["sum_start_min"] / total_segments if total_segments > 0 else 0.0
        )

        for rng in ["1_12", "12_24", "24_36", "36_48"]:
            home_segs = stats["home_segments_by_range"][rng]
            home_wins = stats["home_wins_by_range"][rng]
            away_segs = stats["away_segments_by_range"][rng]
            away_wins = stats["away_wins_by_range"][rng]
            stats[f"home_win_rate_{rng}"] = (
                home_wins / home_segs if home_segs > 0 else 0.0
            )
            stats[f"away_win_rate_{rng}"] = (
                away_wins / away_segs if away_segs > 0 else 0.0
            )

        # Remove intermediate fields
        for key in [
            "sum_start_min",
            "home_segments_by_range",
            "home_wins_by_range",
            "away_segments_by_range",
            "away_wins_by_range",
        ]:
            del stats[key]

    # Attach synergy and head-to-head metrics to each player's stats
    for player in player_stats.keys():
        player_stats[player]["synergy_with_teammates"] = {}
        player_stats[player]["head2head_against"] = {}

    for (p1, p2), record in synergy_stats.items():
        synergy_rate = (
            record["wins"] / record["segments"] if record["segments"] > 0 else 0.0
        )
        player_stats[p1]["synergy_with_teammates"][p2] = synergy_rate
        player_stats[p2]["synergy_with_teammates"][p1] = synergy_rate

    for (home_p, away_p), record in head2head_stats.items():
        h2h_rate = (
            record["wins"] / record["segments"] if record["segments"] > 0 else 0.0
        )
        player_stats[home_p]["head2head_against"][away_p] = h2h_rate

    return player_stats


def build_team_rosters(df) -> dict:
    """
    Builds a mapping of team names to the set of all players who ever appeared for that team.
    """
    team_rosters = {}

    def add_to_roster(team, player):
        team_rosters.setdefault(team, set()).add(player)

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        for i in range(5):
            add_to_roster(home_team, row[f"home_{i}"])
            add_to_roster(away_team, row[f"away_{i}"])

    # Convert sets to lists for serialization
    return {team: list(players) for team, players in team_rosters.items()}


def save_player_stats(stats: dict, output_dir: str) -> None:
    """
    Saves the computed player statistics as a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "players_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
