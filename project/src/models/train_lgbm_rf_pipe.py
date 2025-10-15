import os

import joblib
import lightgbm
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import src

from utils.classification import get_sample_weights

from utils.custom_column_transformers import AddDif
from utils.custom_column_transformers import AddRatio
from utils.custom_column_transformers import AsCategory
from utils.custom_column_transformers import GenAge
from utils.custom_column_transformers import GenDataFromDatetime
from utils.custom_column_transformers import GenInterest

DATA_PATH = os.path.join('../../../../../..', '..', 'data', 'prepared', 'co_prepared_dataset.csv')
MODEL_PATH = os.path.join('../../../../../..', '..', 'models', 'co_lgbm_rf_pipe.pkl')
NUM_CV_SPLITS = 10  # количество сплитов во время перекрёстной проверки
N_TRIALS = 500  # количество попыток при оптимизации гиперпараметров

data = pd.read_csv(DATA_PATH, index_col='app_id', parse_dates=['app_date'])
data = data.reindex(columns=[
    'app_date',
    'prev_request_amount',
    'request_amount',
    'credit_limit',
    'loan_amount',
    'loan_term',
    'income',
    'total_app_num',
    'reject_app_num',
    'birthdate',
    'gender',
    'marital_status',
    'child_num',
    'j_Applicationsnumber',
    'j_Loginsnumber',
    'j_Phonesnumber',
    'j_Lesstenordays',
    'j_Deviceageingwithuser',
    'j_TotalnumofShortTermcreditapplicationsin1day',
    'j_TotalnumofShortTermcreditapplicationsin7days',
    'j_TotalnumofShortTermcreditapplicationsin30days',
    'j_TotalnumofShortTermcreditapplicationsfromipin1day',
    'j_TotalnumofShortTermcreditapplicationsfromipin7days',
    'j_TotalnumofShortTermcreditapplicationsfromipin30days',
    'j_Browserhistorycount',
    'bki_numero_obligaciones',
    'bki_total_saldo',
    'bki_participacion_deuda',
    'bki_numero_obligaciones_dia',
    'bki_saldo_obligaciones_dia',
    'bki_cuota_obligaciones_dia',
    'bki_cantidad_obligaciones_mora',
    'bki_saldo_obligaciones_mora',
    'bki_cuota_obligaciones_mora',
    'bki_valor_mora',
    'default_status',

    'request_term',
    'interest_amount',
])
data = data.query('app_date < "2023-02-01"')

# Out-of-time
other_index = data.query('app_date < "2023-02-01"').index
# train/test
other_index_train, other_index_test = train_test_split(
    other_index,
    stratify=data.loc[other_index, 'default_status'],
    random_state=src.constants.RANDOM_STATE,
)


def get_X_y(df, index):
    X = df.loc[index].drop(columns=[
        'default_status',

        'request_term',

        'interest_amount',
    ])
    y = df.loc[index, 'default_status']

    return X, y


X_train, y_train = get_X_y(data, other_index_train)


def objective(trial: optuna.trial.Trial) -> float:
    sample_weights = get_sample_weights(
        data.loc[other_index_train, 'app_date'],
        trial.suggest_float('forget_coef', 1e-4, .1),
    )

    sss = StratifiedShuffleSplit(
        n_splits=NUM_CV_SPLITS, test_size=.25, random_state=src.constants.RANDOM_STATE)
    aucroc_scores = np.zeros(NUM_CV_SPLITS)
    for i, (t, v) in enumerate(sss.split(X_train, y_train)):
        model = get_model(trial, mode='fit')

        train_X = X_train.iloc[t]
        train_y = y_train.iloc[t]
        train_sw = sample_weights.iloc[t]

        val_X = X_train.iloc[v]
        val_y = y_train.iloc[v]
        val_sw = sample_weights.iloc[v]

        model.fit(
            train_X, train_y,
            LGBM__sample_weight=train_sw,
        )
        y_proba = model.predict_proba(val_X)[:, 1]
        aucroc_scores[i] = roc_auc_score(
            val_y, y_proba,
            sample_weight=val_sw,
        )

    return aucroc_scores.mean()


def get_model(trial: optuna.trial.Trial, mode: str) -> Pipeline:
    dict_with_categories = {
        'gender': ['female', 'male'],
        'marital_status': ['Soltero', 'Casado', 'Union libre', 'Divorciado', 'Viudo(a)'],
    }

    step0 = ColumnTransformer(
        transformers=[
            ('as_category', AsCategory(dict_with_categories), ['gender', 'marital_status']),
            ('gen_interes', GenInterest(), ['loan_amount', 'loan_term']),
            ('gen_age', GenAge(), ['app_date', 'birthdate']),
            ('gen_data_from_datetime', GenDataFromDatetime(), 'app_date'),
            ('add_request_limit_dif', AddDif(), ['request_amount', 'credit_limit']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    step1 = ColumnTransformer(
        [
            ('add_income_interest_ratio', AddRatio(), ['income', 'interest']),
            ('drop', 'drop', ['request_amount']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    preprocess = Pipeline([
        ('step0', step0),
        ('step1', step1),
    ])

    lgbm_model = lightgbm.LGBMClassifier(**get_lgbm_init_params(trial, mode))

    model = Pipeline([
        ('preprocess', preprocess),
        ('LGBM', lgbm_model),
    ])

    return model


def get_lgbm_init_params(trial, mode):
    if mode == 'fit':
        # Core Parameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        num_leaves = trial.suggest_int('num_leaves', 2, 32)

        # Learning Control Parameters
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_child_samples = trial.suggest_int('min_child_samples', 20, 50)

        min_split_gain = trial.suggest_float('min_split_gain', .01, 1.)
    elif mode == 'refit':
        # Core Parameters
        n_estimators = trial.params['n_estimators']
        num_leaves = trial.params['num_leaves']

        # Learning Control Parameters
        max_depth = trial.params['max_depth']
        min_child_samples = trial.params['min_child_samples']

        min_split_gain = trial.params['min_split_gain']
    else:
        assert False, 'ABOBA'

    lgbm_init_params = dict(
        # Core Parameters
        objective='binary',
        boosting_type='rf',
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        n_jobs=-1,
        random_state=src.constants.RANDOM_STATE,

        # Learning Control Parameters
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_split_gain=min_split_gain,
        subsample=.5,
        subsample_freq=1,

        # Objective Parameters
        is_unbalance=True,
    )

    return lgbm_init_params


if __name__ == '__main__':
    study = optuna.create_study(
        sampler=TPESampler(seed=src.constants.RANDOM_STATE),
        direction='maximize',
        study_name='lgbm_rf_study',
    )
    study.optimize(objective, N_TRIALS)

    model = get_model(study.best_trial, mode='refit')
    model.fit(X_train, y_train)

    print('Лучшие гиперпараметры:')
    for hyperparam, value in study.best_trial.params.items():
        print(f'* {hyperparam}: {value}')
    print(f'Лучший mean AUC: {study.best_trial.value}')

    joblib.dump(model, MODEL_PATH)
