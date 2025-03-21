import shutil
from _00_expand_data import expand_data
from _01_encode_data import encode_data
from _02_train_model_cb import train_model_cb
from _02_train_model_lgb import train_model_lgb
from _02_train_model_xgb import train_model_xgb
from _03_evaluate_models import evaluate_models

if __name__ == "__main__":
    for year in range(2007, 2016):
        expand_data(year)
        encode_data(year)
        train_model_lgb(year)
        train_model_xgb(year)
        train_model_cb(year)
        evaluate_models(year)
        shutil.rmtree("catboost_info", ignore_errors=True)
