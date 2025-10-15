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
from sklearn.decomposition import PCA

import src

from utils.classification import get_sample_weights

from src.constants import DATA_PATH_PREPARED

DATA_PATH = DATA_PATH_PREPARED
    # os.path.join('../..', 'data', 'prepared', 'prepared.csv')

MODEL_PATH = os.path.join('../..', 'models', 'cl_lgbm_gbdt_pipe.pkl')
NUM_CV_SPLITS = 10  # количество сплитов во время перекрёстной проверки
N_TRIALS = 30  # количество попыток при оптимизации гиперпараметров

data = pd.read_csv(
    DATA_PATH,
    index_col='AppId',
    parse_dates=['dtstart'],
    dtype={
        'sex': 'category',
        'domen': 'category',
        'nameto': 'category',
        'sitename': 'category',
        'fullname': 'category',
        'nation': 'category',
        'CategoryGoodsId': 'category',
        'martialid': 'category',
        'typeemployerid': 'category',
        'idchannel': 'category',
        'CompanyTypeId': 'category',
        'OccupationAreaId': 'category',
        'weekday': 'category',
        'CustomerReg': 'category',
        'DOC': 'category',
        'positionlevel': 'category',
        'EmploymentType': 'category',
        'pref_mobile': 'category',
    },
)

pos_cols = [f'position_{i}' for i in range(1, 313)]


def get_X_y(df, index):
    X = df.loc[index].drop(columns=[
        'mob_60',
        'interest_amount',
        'dtstart',
    ])
    y = df.loc[index, 'mob_60']

    return X, y


other_index = data.query('dtstart < "2022-10-01"').index
october_index = data.query('"2022-10-01" <= dtstart < "2022-11-01"').index
november_index = data.query('"2022-11-01" <= dtstart < "2022-12-01"').index
december_index = data.query('"2022-12-01" <= dtstart < "2023-01-01"').index
# train/test
other_index_train, other_index_test = train_test_split(
    other_index,
    stratify=data.loc[other_index, 'mob_60'],
    random_state=src.constants.RANDOM_STATE,
)

X, y = get_X_y(data, data.index)
X_train, y_train = get_X_y(data, other_index_train)
X_test, y_test = get_X_y(data, other_index_test)
X_october, y_october = get_X_y(data, october_index)
X_november, y_november = get_X_y(data, november_index)
X_december, y_december = get_X_y(data, december_index)


def objective(trial: optuna.trial.Trial) -> float:
    sample_weights = get_sample_weights(
        data.loc[other_index_train, 'dtstart'],
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
    if mode == 'fit':
        n_components = trial.suggest_int('n_components', 1, 30)
    elif mode == 'refit':
        n_components = trial.params['n_components']

    сt_pca = ColumnTransformer(
        [
            ('pca', PCA(n_components=n_components,
                        whiten=True,
                        random_state=src.constants.RANDOM_STATE),
             pos_cols
             ),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    lgbm_model = lightgbm.LGBMClassifier(**get_lgbm_init_params(trial, mode))

    model = Pipeline([
        ('pca_preprocess', сt_pca),
        ('LGBM', lgbm_model),
    ])

    return model


def get_lgbm_init_params(trial, mode):
    if mode == 'fit':
        # Core Parameters
        n_estimators = trial.suggest_int('n_estimators', 200, 400)
        num_leaves = trial.suggest_int('num_leaves', 2, 6)

        # Learning Control Parameters
        max_depth = trial.suggest_int('max_depth', 1, 8)
        min_child_samples = trial.suggest_int('min_child_samples', 20, 50)
        reg_alpha = trial.suggest_float('reg_alpha', .5, 2)
        reg_lambda = trial.suggest_float('reg_lambda', .5, 2)

        min_split_gain = trial.suggest_float('min_split_gain', .01, 1.)
    elif mode == 'refit':
        # Core Parameters
        n_estimators = trial.params['n_estimators']
        num_leaves = trial.params['num_leaves']

        # Learning Control Parameters
        max_depth = trial.params['max_depth']
        min_child_samples = trial.params['min_child_samples']
        reg_alpha = trial.params['reg_alpha']
        reg_lambda = trial.params['reg_lambda']

        min_split_gain = trial.params['min_split_gain']
    else:
        assert False, 'ABOBA'

    lgbm_init_params = dict(
        # Core Parameters
        objective='binary',
        boosting_type='gbdt',
        learning_rate=.05,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        num_leaves=num_leaves,
        n_jobs=-1,
        random_state=src.constants.RANDOM_STATE,

        # Learning Control Parameters
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_split_gain=min_split_gain,

        # Objective Parameters
        is_unbalance=True,
    )

    return lgbm_init_params


if __name__ == '__main__':
    study = optuna.create_study(
        sampler=TPESampler(seed=src.constants.RANDOM_STATE),
        direction='maximize',
        study_name='lgbm_gbdt_study',
    )
    study.optimize(objective, N_TRIALS)

    model = get_model(study.best_trial, mode='refit')
    model.fit(X_train, y_train)

    print('Лучшие гиперпараметры:')
    for hyperparam, value in study.best_trial.params.items():
        print(f'* {hyperparam}: {value}')
    print(f'Лучший mean AUC: {study.best_trial.value}')

    joblib.dump(model, MODEL_PATH)
