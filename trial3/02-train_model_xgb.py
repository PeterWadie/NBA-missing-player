# 02-train_model_xgb.py
import optuna
import xgboost as xgb
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
BEST_PARAMS_FILE = "best_xgb_params.json"

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
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "lambda": trial.suggest_float("lambda", 1e-5, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "n_jobs": -1,
            "seed": 42,
        }

        train_dataset = xgb.DMatrix(X_train, label=y_train)
        valid_dataset = xgb.DMatrix(X_valid, label=y_valid)

        model = xgb.train(
            param,
            train_dataset,
            evals=[(valid_dataset, "valid")],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        preds = model.predict(valid_dataset, iteration_range=(0, model.best_iteration))
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
best_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
final_model = xgb.train(
    best_params,
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=1000,
    evals=[(xgb.DMatrix(X_valid, label=y_valid), "valid")],
    early_stopping_rounds=50,
    verbose_eval=50,
)

# Save final model
final_model.save_model("best_xgb_model.json")
print("Final model saved to best_xgb_model.json")
