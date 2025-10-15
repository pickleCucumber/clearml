import lightgbm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.constants import RANDOM_STATE
from src.data.cl_preprocess import foo_dtype, foo_sex, foo_sitename
from utils.custom_column_transformers import GenAge, GenDataFromDatetime


def get_lgbm_init_params(trial, mode):
    if mode == "fit":
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        learning_rate = trial.suggest_float("learning_rate", 0.05, 1.0, log=True)
        num_leaves = trial.suggest_int("num_leaves", 4, 64)
        min_child_samples = trial.suggest_int("min_child_samples", 20, 50)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 3)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 3)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.01, 1.0)
        min_split_gain = trial.suggest_float("min_split_gain", 0.01, 1.0)

    elif mode == "refit":
        n_estimators = trial.params["n_estimators"]
        max_depth = trial.params["max_depth"]
        learning_rate = trial.params["learning_rate"]
        num_leaves = trial.params["num_leaves"]
        min_child_samples = trial.params["min_child_samples"]
        reg_alpha = trial.params["reg_alpha"]
        reg_lambda = trial.params["reg_lambda"]
        colsample_bytree = trial.params["colsample_bytree"]
        min_split_gain = trial.params["min_split_gain"]

    lgbm_init_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "random_state": RANDOM_STATE,
        "min_child_samples": min_child_samples,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_split_gain": min_split_gain,
        "colsample_bytree": colsample_bytree,
        "is_unbalance": True,
    }

    return lgbm_init_params


def get_model(trial, mode):
    cat_cols = [
        "sex",
        "sitename",
        "citizenshipid",
        "martialid",
    ]

    level_1 = ColumnTransformer(
        [
            ("preprocess_sex", FunctionTransformer(foo_sex), "sex"),
            ("preprocess_sitename", FunctionTransformer(foo_sitename), "sitename"),
            ("gen_age", GenAge(), ["dtstart", "birthday"]),
            ("gen_datetime_data", GenDataFromDatetime(month=True), "dtstart"),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    level_2 = ColumnTransformer(
        [("preprocess_dtype", FunctionTransformer(foo_dtype), cat_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    preprocess_pipe = Pipeline([("step_1", level_1), ("step_2", level_2)])

    model = lightgbm.LGBMClassifier(**get_lgbm_init_params(trial, mode), n_jobs=5)

    model = Pipeline(
        [
            ("preprocess", preprocess_pipe),
            ("model", CalibratedClassifierCV(model)),
        ]
    )

    return model
