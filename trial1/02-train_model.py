# 02-train_model.py
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split

with open("player_encoder.pkl", "rb") as f:
    player_encoder = pickle.load(f)

# Define feature columns: four home players, five away players, plus the extra features
feature_cols = [f"home_{i}" for i in range(4)] + [f"away_{i}" for i in range(5)] + ["home_team", "away_team", "starting_min"]

# Load encoded data
df_encoded = pd.read_csv("01-encoded_data.csv")
X = df_encoded[feature_cols]
y = df_encoded["removed_player"]

# Split the data into training and validation sets (80/20 split)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare LightGBM datasets
train_dataset = lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset)

# Set parameters for the multiclass LightGBM model
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": player_encoder.classes_.size,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": 10,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Train the model with early stopping
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=1000,
    valid_sets=[train_dataset, valid_dataset],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
)

# Save the trained model
model.save_model("lightgbm_model.txt")
print("LightGBM model trained and saved as lightgbm_model.txt")
