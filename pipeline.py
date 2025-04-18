# pipeline.py
import os
from config import YEARS, OUTPUT_BASE_DIR, RANDOM_SEED
from utils.data_utils import load_matchup_data, save_dataframe, save_json, save_encoder
from utils.stats_utils import (
    compute_player_stats,
    build_team_rosters,
)
from models.encoder import expand_binary_data, encode_binary_data
from models.trainer import train_xgboost
from models.evaluator import evaluate_model


def main():
    for year in YEARS:
        print(f"Processing year: {year}")
        output_dir = os.path.join(OUTPUT_BASE_DIR, "data", str(year))
        os.makedirs(output_dir, exist_ok=True)

        # Load matchup data
        df = load_matchup_data(year)

        # Compute player statistics and save them
        player_stats = compute_player_stats(df)
        save_json(player_stats, output_dir, "players_stats")
        print(f"Player stats computed and saved for {year}.")

        # Build and save team rosters
        team_rosters = build_team_rosters(df)
        save_json(team_rosters, output_dir, "team_rosters")
        print(f"Team rosters saved for {year}.")

        # Expand the data into binary samples and save
        df_binary = expand_binary_data(df, player_stats, team_rosters, RANDOM_SEED)
        save_dataframe(df_binary, output_dir, "binary_data")
        print(f"Binary data expanded and saved for {year}.")

        # Encode the binary data and save
        encoded_binary_data, player_encoder, team_encoder = encode_binary_data(
            df_binary
        )
        save_dataframe(encoded_binary_data, output_dir, "encoded_binary_data")
        save_encoder(player_encoder, output_dir, "player_encoder")
        save_encoder(team_encoder, output_dir, "team_encoder")
        print(f"Binary data encoded and saved for {year}.")

        # Train XGBoost model
        train_xgboost(output_dir)
        print(f"XGBoost model trained and saved for {year}.")
    results = evaluate_model()
    save_json(results, OUTPUT_BASE_DIR, "evaluation_results")
    print(f"Evaluation results saved.")
    print("Pipeline completed.")
    return


if __name__ == "__main__":
    main()
