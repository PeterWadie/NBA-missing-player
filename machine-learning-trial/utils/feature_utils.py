# utils/feature_utils.py
from itertools import combinations


def build_candidate_features(
    candidate: str, present_players: list, away_lineup: list, players_stats: dict
) -> dict:
    """
    Builds and returns a dictionary of candidate feature values using the given player's stats.

    The returned dictionary contains:
      - candidate_overall_win_rate
      - candidate_home_win_rate
      - candidate_away_win_rate
      - candidate_avg_start_min
      - candidate_avg_synergy_with_present
      - candidate_avg_head2head
      - present_4_avg_synergy
    """

    def get_metric(player, metric):
        return players_stats.get(player, {}).get(metric, 0.0)

    def get_synergy(p1, p2):
        return players_stats.get(p1, {}).get("synergy_with_teammates", {}).get(p2, 0.0)

    def get_head2head(home_p, away_p):
        return (
            players_stats.get(home_p, {}).get("head2head_against", {}).get(away_p, 0.0)
        )

    features = {}
    features["candidate_overall_win_rate"] = get_metric(candidate, "overall_win_rate")
    features["candidate_home_win_rate"] = get_metric(candidate, "home_win_rate")
    features["candidate_away_win_rate"] = get_metric(candidate, "away_win_rate")
    features["candidate_avg_start_min"] = get_metric(candidate, "avg_start_min")

    synergy_vals = [get_synergy(candidate, p) for p in present_players]
    features["candidate_avg_synergy_with_present"] = (
        sum(synergy_vals) / len(synergy_vals) if synergy_vals else 0.0
    )

    h2h_vals = [get_head2head(candidate, opp) for opp in away_lineup]
    features["candidate_avg_head2head"] = (
        sum(h2h_vals) / len(h2h_vals) if h2h_vals else 0.0
    )

    # Compute average synergy among the present players themselves
    if len(present_players) >= 2:
        synergy_sum = 0.0
        count = 0
        for p1, p2 in combinations(present_players, 2):
            synergy_sum += get_synergy(p1, p2)
            count += 1
        features["present_4_avg_synergy"] = synergy_sum / count if count > 0 else 0.0
    else:
        features["present_4_avg_synergy"] = 0.0

    return features
