# 02-train_model_lgb.py
import optuna
import lightgbm as lgb
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df_encoded = pd.read_csv("encoded_binary_data.csv")

# Define features and target variable
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

# Path to store best parameters
BEST_PARAMS_FILE = "best_lgb_params.json"

# Check if best parameters exist
if os.path.exists(BEST_PARAMS_FILE):
    print("Loading best hyperparameters from cache...")
    with open(BEST_PARAMS_FILE, "r") as f:
        best_params = json.load(f)
else:
    print("No cached hyperparameters found. Running Optuna tuning...")

    # Define Optuna objective function
    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset)

        model = lgb.train(
            param,
            train_dataset,
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        pred_labels = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_valid, pred_labels)

        return accuracy  # Maximizing accuracy

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Adjust as needed

    # Save best parameters
    best_params = study.best_params
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f)
    print("Best hyperparameters saved!")

# Train final model with best parameters
best_params.update({"objective": "binary", "metric": "binary_logloss"})
final_model = lgb.train(
    best_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)

# Save final model
final_model.save_model("best_lgb_model.txt")
print("Final model saved to best_lgb_model.txt")
