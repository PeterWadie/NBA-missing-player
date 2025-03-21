# _02_train_model_cb.py
import optuna
import catboost as cb
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model_cb(year):
    output_dir = f"./{year}"
    # Load data
    df_encoded = pd.read_csv(f"{output_dir}/encoded_binary_data.csv")

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
    BEST_PARAMS_FILE = f"{output_dir}/best_cb_params.json"

    # Check if best parameters exist
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "r") as f:
            best_params = json.load(f)
    else:
        # Define Optuna objective function
        def objective(trial):
            param = {
                "loss_function": "Logloss",
                "eval_metric": "Accuracy",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "depth": trial.suggest_int("depth", 3, 12),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                ),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "verbose": 0,
                "random_seed": 42,
                "allow_writing_files": False,
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

            return accuracy  # Maximizing accuracy

        # Run Optuna optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)  # Adjust as needed

        # Save best parameters
        best_params = study.best_params
        with open(BEST_PARAMS_FILE, "w") as f:
            json.dump(best_params, f)

    # Train final model with best parameters
    best_params.update({"loss_function": "Logloss", "eval_metric": "Accuracy"})
    final_model = cb.CatBoostClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50,
        verbose=50,
    )

    # Save final model
    final_model.save_model(f"{output_dir}/best_cb_model.cbm")
    print(f"Step 2 complete for {year}: final model saved to best_cb_model.cbm")
