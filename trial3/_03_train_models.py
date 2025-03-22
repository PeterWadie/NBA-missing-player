# _03_train_models.py
import os
import json
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

def load_data(year):
    """
    Loads the encoded CSV data, defines feature and target columns, and returns the data along with output_dir.
    """
    output_dir = f"./{year}"
    df_encoded = pd.read_csv(os.path.join(output_dir, "encoded_binary_data.csv"))
    feature_cols = (
        [f"home_{i}" for i in range(4)]
        + [f"away_{i}" for i in range(5)]
        + [
            "home_team",
            "away_team",
            "starting_min",
            "candidate_player",
            "candidate_overall_win_rate",
            "candidate_home_win_rate",
            "candidate_away_win_rate",
            "candidate_avg_start_min",
            "candidate_avg_synergy_with_present",
            "candidate_avg_head2head",
            "present_4_avg_synergy",
        ]
    )
    label_col = "label"
    X = df_encoded[feature_cols]
    y = df_encoded[label_col]
    return output_dir, X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    """Splits data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_best_params(best_params_file, objective_func, n_trials=50):
    """
    Checks if a best parameters file exists. If so, it loads and returns it;
    otherwise, runs the provided objective function with Optuna to optimize hyperparameters,
    saves the result, and returns it.
    """
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            best_params = json.load(f)
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials)
        best_params = study.best_params
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)
    return best_params

def train_model_cb(year):
    output_dir, X, y = load_data(year)
    X_train, X_valid, y_train, y_valid = train_val_split(X, y)
    best_params_file = os.path.join(output_dir, "best_cb_params.json")
    
    def objective(trial):
        param = {
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
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
        return accuracy_score(y_valid, preds)
    
    best_params = get_best_params(best_params_file, objective)
    best_params.update({"loss_function": "Logloss", "eval_metric": "Accuracy"})
    
    final_model = cb.CatBoostClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50,
        verbose=50,
    )
    final_model.save_model(os.path.join(output_dir, "best_cb_model.cbm"))
    print(f"CatBoost model trained and saved at {os.path.join(output_dir, 'best_cb_model.cbm')}")

def train_model_lgb(year):
    output_dir, X, y = load_data(year)
    X_train, X_valid, y_train, y_valid = train_val_split(X, y)
    best_params_file = os.path.join(output_dir, "best_lgb_params.json")
    
    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
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
        return accuracy_score(y_valid, pred_labels)
    
    best_params = get_best_params(best_params_file, objective)
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
    final_model.save_model(os.path.join(output_dir, "best_lgb_model.txt"))
    print(f"LightGBM model trained and saved at {os.path.join(output_dir, 'best_lgb_model.txt')}")

def train_model_xgb(year):
    output_dir, X, y = load_data(year)
    X_train, X_valid, y_train, y_valid = train_val_split(X, y)
    best_params_file = os.path.join(output_dir, "best_xgb_params.json")
    
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
        return accuracy_score(y_valid, pred_labels)
    
    best_params = get_best_params(best_params_file, objective)
    best_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    
    final_model = xgb.train(
        best_params,
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=1000,
        evals=[(xgb.DMatrix(X_valid, label=y_valid), "valid")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    final_model.save_model(os.path.join(output_dir, "best_xgb_model.json"))
    print(f"XGBoost model trained and saved at {os.path.join(output_dir, 'best_xgb_model.json')}")
