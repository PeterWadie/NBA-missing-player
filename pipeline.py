# pipeline.py
import os
from config import YEARS, TARGET_YEARS, OUTPUT_BASE_DIR, RANDOM_SEED
from utils.data_utils import load_matchup_data, save_dataframe, save_json, save_encoder
from utils.stats_utils import (
    compute_player_stats,
    build_team_rosters,
)
from models.encoder import expand_binary_data, encode_binary_data
from models.trainer import train_xgboost
from models.evaluator import evaluate_model


def main():
    # Process all training data years (2007-2015)
    for year in YEARS:
        print(f"Processing data for year: {year}")
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
    
    # Now use previous year's data to predict the next year
    # For example: use 2007 to predict 2008, use 2008 to predict 2009, and so on, finally use 2015 to predict 2016
    for target_year in TARGET_YEARS:
        source_year = target_year - 1  # Source year is always the previous year of the target year
        print(f"Training model using {source_year} data to predict {target_year}")
        
        # Create output directory for the target year
        target_dir = os.path.join(OUTPUT_BASE_DIR, "data", str(target_year))
        os.makedirs(target_dir, exist_ok=True)
        
        # Source data directory
        source_dir = os.path.join(OUTPUT_BASE_DIR, "data", str(source_year))
        
        # Train model using source year data and save to target year directory
        train_xgboost(source_dir, target_dir)
        print(f"XGBoost model trained using {source_year} data and saved for predicting {target_year}.")

    # Evaluate models for all target years (2008-2016)
    results = evaluate_model()
    save_json(results, OUTPUT_BASE_DIR, "evaluation_results")
    print(f"Evaluation results saved.")
    print("Pipeline completed.")
    return


if __name__ == "__main__":
    main()
