# 02-train_model.py
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

df_encoded = pd.read_csv("encoded_binary_data.csv")

feature_cols = (
    [f"home_{i}" for i in range(4)]
    + [f"away_{i}" for i in range(5)]
    + ["home_team", "away_team", "starting_min", "candidate_player"]
)
label_col = "label"

X = df_encoded[feature_cols]
y = df_encoded[label_col]

# Split into train/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",  # This is key for two-stage approach
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": 8,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

model = lgb.train(
    params,
    train_dataset,
    num_boost_round=1000,
    valid_sets=[train_dataset, valid_dataset],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
)

model.save_model("binary_lgb_model.txt")
print("Step 2 complete: Model trained and saved to binary_lgb_model.txt")
