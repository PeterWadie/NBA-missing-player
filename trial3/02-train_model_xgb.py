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
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_loguniform("lambda", 1e-3, 10),
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "random_state": 42,
            "use_label_encoder": False,
        }

        model = xgb.XGBClassifier(**param)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
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
final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, random_state=42)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=50,
    verbose=True,
)

# Save final model
final_model.save_model("best_xgb_model.json")
print("Final XGBoost model saved to best_xgb_model.json")
