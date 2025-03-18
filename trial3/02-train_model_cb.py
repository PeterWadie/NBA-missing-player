# 02-train_model_cb.py
import optuna
import catboost as cb
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
BEST_PARAMS_FILE = "best_cb_params.json"

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
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 3, 12),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
            "bagging_temperature": trial.suggest_uniform(
                "bagging_temperature", 0.0, 1.0
            ),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_loguniform("random_strength", 1e-3, 10),
            "verbose": 0,
            "random_seed": 42,
        }

        model = cb.CatBoostClassifier(**param)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=50,
            verbose=False,
        )

        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, preds)

        return accuracy

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Save best parameters
    best_params = study.best_params
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f)
    print("Best hyperparameters saved!")

# Train final model with best parameters
final_model = cb.CatBoostClassifier(**best_params, random_seed=42)
final_model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=50,
    verbose=True,
)

# Save final model
final_model.save_model("best_cb_model.cbm")
print("Final CatBoost model saved to best_cb_model.cbm")
