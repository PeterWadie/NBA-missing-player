import json
import requests
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import google.auth.transport.requests
from google.oauth2 import service_account
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Google Vertex AI Configuration -------------------
PROJECT_ID = "project_id"  # Replace with your Google Cloud Project ID
LOCATION = "us-east1"  # Change if needed
MODEL_ID = "text-embedding-005"

# Load Google Cloud credentials
CREDENTIALS_PATH = "path"  # Replace with your service account key file path
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# ------------------- File Paths -------------------
COMBINED_DATA_FILE = Path("input-dataset/combined_filtered.csv")
TEAM_NAMES_FILE = Path("input-dataset/teams-full-name.json")  
PLAYER_STATS_FILE = Path("input-dataset/player-stats.json")  
EMBEDDINGS_CACHE_DIR = Path("./embeddings_cache")  
LOGS_DIR = Path("./logs")  

# Ensure directories exist
EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load mappings
with TEAM_NAMES_FILE.open() as f:
    team_name_mapping = json.load(f)  

with PLAYER_STATS_FILE.open() as f:
    player_stats = json.load(f)  

# ------------------- Embedding Functions -------------------

def get_google_vertex_embedding(text, task_type="SEMANTIC_SIMILARITY", output_dimensionality=768):
    """Fetches embeddings from Google Vertex AI's text-embedding-005 model."""
    try:
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:predict"

        # Ensure token is fresh
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)

        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }

        data = {
            "instances": [{"task_type": task_type, "content": text}],
            "parameters": {"output_dimensionality": output_dimensionality}
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["predictions"][0]["embeddings"]["values"]
        else:
            raise Exception(f"Error fetching embedding: {response.text}")

    except Exception as e:
        print(f"Error in get_google_vertex_embedding: {e}")
        return None

def load_embedding(file_path: Path):
    """Loads embeddings from cache if available."""
    if file_path.exists():
        with file_path.open("r") as f:
            return np.array(json.load(f))
    return None

def save_embedding(file_path: Path, embedding):
    """Saves embeddings to cache."""
    with file_path.open("w") as f:
        json.dump(embedding, f)  # No need to call .tolist(), Vertex AI already returns a list

def get_player_embedding(player_name: str):
    """Fetches cached player embedding or computes a new one using Google Vertex AI."""
    player_hash = hashlib.md5(player_name.encode()).hexdigest()
    player_embedding_file = EMBEDDINGS_CACHE_DIR / f"player_{player_hash}.json"

    player_embedding = load_embedding(player_embedding_file)
    if player_embedding is None:
        # Use RETRIEVAL_DOCUMENT for player embeddings
        player_embedding = get_google_vertex_embedding(player_name, task_type="RETRIEVAL_DOCUMENT")
        if player_embedding:
            save_embedding(player_embedding_file, player_embedding)  # Save to cache

    return np.array(player_embedding)

def get_lineup_embedding(lineup_text: str):
    """Fetches cached lineup embedding or computes a new one using Google Vertex AI."""
    lineup_hash = hashlib.md5(lineup_text.encode()).hexdigest()
    lineup_embedding_file = EMBEDDINGS_CACHE_DIR / f"lineup_{lineup_hash}.json"

    lineup_embedding = load_embedding(lineup_embedding_file)
    if lineup_embedding is None:
        # Use QUESTION_ANSWERING for lineup embeddings
        lineup_embedding = get_google_vertex_embedding(lineup_text, task_type="QUESTION_ANSWERING")
        if lineup_embedding:
            save_embedding(lineup_embedding_file, lineup_embedding)  # Save to cache

    return np.array(lineup_embedding)

# ------------------- Game Processing Functions -------------------

def compute_stat_similarity(player_name: str, teammates: list, opponents: list, lineup_start_min: float):
    """Computes a stat-based similarity score for a player."""
    stats = player_stats.get(player_name, {})

    avg_start_min = stats.get("avg_start_min", 0)
    start_min_diff = 1 - abs(avg_start_min - lineup_start_min) / 48  # Normalize between [0,1]

    synergy_values = [stats.get("synergy_with_teammates", {}).get(teammate, 0) for teammate in teammates]
    avg_synergy = np.mean(synergy_values) if synergy_values else 0

    head2head_values = [stats.get("head2head_against", {}).get(opponent, 0) for opponent in opponents]
    avg_head2head = np.mean(head2head_values) if head2head_values else 0

    home_win_rate = stats.get("home_win_rate", 0.5)
    away_win_rate = stats.get("away_win_rate", 0.5)
    overall_win_rate = (home_win_rate + away_win_rate) / 2  

    stat_score = (
        0.4 * start_min_diff +
        0.3 * avg_synergy +
        0.2 * avg_head2head +
        0.1 * overall_win_rate
    )

    return stat_score  # Returns a value between [0,1]

# ------------------- Main Processing Loop -------------------

correct_predictions = 0
total_predictions = 0
df = pd.read_csv(COMBINED_DATA_FILE)

for index, row in df.iterrows():
    home_team = team_name_mapping.get(row["home_team"], row["home_team"])
    away_team = team_name_mapping.get(row["away_team"], row["away_team"])
    
    home_players = [row[f"home_{i}"] for i in range(5)]
    away_players = [row[f"away_{i}"] for i in range(5)]
    
    log_filename = LOGS_DIR / f"lineup_{index}.json"
    lineup_log = {"lineup_id": index, "home_team": home_team, "away_team": away_team, "starting_min": row["starting_min"], "tests": []}

    for removed_player in home_players:
        lineup_text = (
            f"NBA Game Matchup: {home_team} vs. {away_team}\n"
            f"Starting Minute: {row['starting_min']}\n\n"
            f"Current active players for {home_team}: {', '.join(p for p in home_players if p != removed_player)}\n\n"
            f"Players on the court for {away_team}: {', '.join(away_players)}\n\n"
            f"One player is missing from {home_team} lineup.\n\n"
            f"Based on the context of the game, who is the missing player?"
        )

        lineup_embedding = get_lineup_embedding(lineup_text)

        best_match = None
        best_similarity = -1
        similarity_scores = {}

        for candidate in home_players:
            emb_sim = cosine_similarity([lineup_embedding], [get_player_embedding(candidate)])[0][0]
            stat_sim = compute_stat_similarity(candidate, home_players, away_players, row["starting_min"])

            final_score = 0.8 * emb_sim + 0.2 * stat_sim
            # final_score = emb_sim
            similarity_scores[candidate] = final_score

            if final_score > best_similarity:
                best_similarity = final_score
                best_match = candidate

        lineup_log["tests"].append({"removed_player": removed_player, "similarity_scores": similarity_scores, "best_match": best_match, "correct": best_match == removed_player})
        if best_match == removed_player:
            correct_predictions += 1
        total_predictions += 1

    with log_filename.open("w") as log_file:
        json.dump(lineup_log, log_file, indent=4)

accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Model Accuracy: {accuracy:.2%}")
