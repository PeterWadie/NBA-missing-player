# pipeline.py
import os
import json

from config import YEARS, OUTPUT_BASE_DIR, RANDOM_SEED
from utils.data_utils import load_matchup_data, save_dataframe
from utils.stats_utils import (
    compute_player_stats,
    save_player_stats,
    build_team_rosters,
)
from models.encoder import expand_binary_data, encode_binary_data
from models.trainer import train_all_models
from models.evaluator import evaluate_models


def main():
    for year in YEARS:
        print(f"Processing year: {year}")
        output_dir = os.path.join(OUTPUT_BASE_DIR, "data", str(year))
        os.makedirs(output_dir, exist_ok=True)

        # Load matchup data
        df = load_matchup_data(year)

        # Compute player statistics and save them
        player_stats = compute_player_stats(df)
        save_player_stats(player_stats, output_dir)
        print(f"Player stats computed and saved for {year}.")

        # Build and save team rosters
        team_rosters = build_team_rosters(df)
        with open(os.path.join(output_dir, "team_rosters.json"), "w") as f:
            json.dump(team_rosters, f)
        print(f"Team rosters saved for {year}.")

        # Expand the data into binary samples and save
        df_binary = expand_binary_data(df, player_stats, team_rosters, RANDOM_SEED)
        save_dataframe(df_binary, output_dir, "binary_data.csv")
        print(f"Binary data expanded and saved for {year}.")

        # Encode the binary data and save
        _, _, _ = encode_binary_data(df_binary, output_dir)
        print(f"Binary data encoded and saved for {year}.")

        # Train models (LightGBM, XGBoost, CatBoost)
        train_all_models(output_dir)

        # Evaluate models on the test scenario
        evaluate_models(year, output_dir)


if __name__ == "__main__":
    main()
