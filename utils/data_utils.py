# utils/data_utils.py
import os
import pandas as pd
from config import DATA_DIR


def load_matchup_data(year: int) -> pd.DataFrame:
    """
    Loads matchup data for a given year.
    """
    file_path = os.path.join(DATA_DIR, f"matchups-{year}.csv")
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """
    Saves the DataFrame as a CSV file to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)
