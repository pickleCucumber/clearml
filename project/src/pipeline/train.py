from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


## опишем этап обучения lgbm
@PipelineDecorator.component(
    return_values=["model"],
    cache=True,
    task_type=TaskTypes.training,
    output_uri=True,
    # helper_functions=[MultiColumnOrdinalEncoder, AsCategory]
)
def train_lgbm(X_train, y_train):
    import joblib
    import numpy as np
    import optuna
    from clearml import Task
    from optuna.samplers import TPESampler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    from src.constants import MODEL_DIR, RANDOM_STATE
    from src.train.lgbm import get_model

    task = Task.current_task()
    logger = task.get_logger()

    def objective(trial):
        tss = TimeSeriesSplit(n_splits=10)
        auc_scores = np.zeros(10)

        for i, (t, v) in enumerate(tss.split(X_train, y_train)):
            model = get_model(trial, mode="fit")

            train_X = X_train.iloc[t]
            train_y = y_train.iloc[t]

            val_X = X_train.iloc[v]
            val_y = y_train.iloc[v]

            model.fit(train_X, train_y)
            y_proba = model.predict_proba(val_X)[:, 1]

            auc_scores[i] = roc_auc_score(val_y, y_proba)

        logger.report_scalar(
            title="Optimization",
            series="Mean AUC",
            value=auc_scores.mean(),
            iteration=trial.number,
        )

        logger.report_scalar(
            title="Optimization",
            series="Max AUC",
            value=auc_scores.max(),
            iteration=trial.number,
        )

        logger.report_scalar(
            title="Optimization",
            series="Min AUC",
            value=auc_scores.min(),
            iteration=trial.number,
        )

        logger.report_scalar(
            title="Optimization",
            series="Std AUC",
            value=auc_scores.std(),
            iteration=trial.number,
        )

        return auc_scores.mean()

    study = optuna.create_study(
        direction="maximize",
        study_name="optuna",
        sampler=TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(objective, n_trials=300, show_progress_bar=True)

    best_trial = study.best_trial
    task.connect(best_trial.params, name="best_hyperparameters")

    logger.report_text(f"Лучший AUC: {best_trial.value}")
    logger.report_single_value("Лучший AUC", best_trial.value)

    model = get_model(best_trial, mode="refit")

    model.fit(X_train, y_train)

    model_path = f"{MODEL_DIR}/lgbm_model.pkl"
    joblib.dump(model, model_path)

    logger.report_text(f"Модель сохранена в {model_path}")

    return model


## опишем этап обучения random forest
@PipelineDecorator.component(
    return_values=["model"],
    cache=True,
    task_type=TaskTypes.training,
    output_uri=True,
    # helper_functions=[MultiColumnOrdinalEncoder, AsCategory]
)
def train_rf(X_train, y_train):
    import joblib
    import numpy as np
    from clearml import Task
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    from src.constants import MODEL_DIR
    from src.train.rf import get_model

    task = Task.current_task()
    logger = task.get_logger()

    rf_estimator = get_model()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=110, num=11)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(3, 23, num=11)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 30, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 6, 8, 10, 12, 20, 30, 50, 100]

    random_grid = {
        "model__estimator__n_estimators": n_estimators,
        "model__estimator__max_depth": max_depth,
        "model__estimator__min_samples_split": min_samples_split,
        "model__estimator__min_samples_leaf": min_samples_leaf,
        # 'model__method': ['isotonic', 'sigmoid'],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_estimator,
        param_distributions=random_grid,
        n_iter=300,
        scoring=["f1", "roc_auc"],
        n_jobs=5,
        cv=TimeSeriesSplit(n_splits=5),
        verbose=1,
        return_train_score=True,
        refit="roc_auc",
    )
    rf_search.fit(X_train, y_train)

    print(rf_search.cv_results_)
    keys = [
        "std_train_roc_auc",
        "mean_train_roc_auc",
        "std_test_roc_auc",
        "mean_test_roc_auc",
    ]

    for key in keys:
        for i, value in enumerate(rf_search.cv_results_[key]):
            logger.report_scalar(
                title="Optimization",
                series=key,
                value=value,
                iteration=i,
            )

    task.connect(rf_search.best_params_, name="best_hyperparameters")

    logger.report_text(f"Лучший AUC: {rf_search.best_score_}")
    logger.report_single_value("Лучший AUC", rf_search.best_score_)

    model = rf_search.best_estimator_

    model_path = f"{MODEL_DIR}/rf_model.pkl"
    joblib.dump(model, model_path)

    logger.report_text(f"Модель сохранена в {model_path}")

    return model
