# utils/data_utils.py
import os
import pandas as pd
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR


def load_dataframe(input_dir: str, filename: str) -> pd.DataFrame:
    """
    Loads a CSV file as a DataFrame.
    """
    return pd.read_csv(os.path.join(input_dir, f"{filename}.csv"))


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """
    Saves the DataFrame as a CSV file to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)


def load_json(input_dir: str, filename: str) -> dict:
    """
    Loads a JSON file as a dictionary.
    """
    with open(os.path.join(input_dir, f"{filename}.json"), "r") as f:
        return json.load(f)


def save_json(dict: dict, output_dir: str, filename: str) -> None:
    """
    Saves the computed player statistics as a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
        json.dump(dict, f, indent=2)


def load_encoder(input_dir: str, filename: str) -> LabelEncoder:
    """
    Loads a label encoder from a pickle file.
    """
    with open(os.path.join(input_dir, f"{filename}.pkl"), "rb") as f:
        return pickle.load(f)


def save_encoder(encoder: LabelEncoder, output_dir: str, filename: str) -> None:
    """
    Saves the label encoder as a pickle file.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{filename}.pkl"), "wb") as f:
        pickle.dump(encoder, f)


def load_matchup_data(year: int) -> pd.DataFrame:
    """
    Loads matchup data for a given year.
    """
    return load_dataframe(DATA_DIR, f"matchups-{year}")
