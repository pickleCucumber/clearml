from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils.metrics.business import approval_rate_plot
from utils.metrics.classification import my_binary_classification_report


def calibrated_shap_values(
    model: BaseEstimator,
    X_shap: pd.DataFrame,
):
    shap_values_list = []
    for calibrated_classifier in model.calibrated_classifiers_:
        explainer = shap.TreeExplainer(calibrated_classifier.estimator)
        shap_values = explainer.shap_values(X_shap)
        # LGBM возвращает 2 класса
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        shap_values_list.append(shap_values)

    shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
    return shap_values


def compute_shap_values(
    model: BaseEstimator,
    X_shap: pd.DataFrame,
    title_suffix: str = "",
    calibrated: bool = False,
) -> None:
    if calibrated:
        shap_values = calibrated_shap_values(model, X_shap)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(
        shap_values,
        features=X_shap,
        class_names=[0, 1],
        max_display=15,
        show=False,
    )
    plt.title(f"SHAP Summary {title_suffix}")
    plt.show()
    plt.close()


def evaluate_classification_performance(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "",
) -> None:
    print(f"{title} Metrics")
    threshold = approval_rate_plot(model.predict_proba(X)[:, 1], title=title)
    my_binary_classification_report(model, X, y, threshold, title)


def preprocess_and_compute_shap_values(
    model: Pipeline,
    data: pd.DataFrame,
    preprocess: Pipeline,
    cols_dict: Dict[str, str],
    title_suffix: str,
    calibrated: bool = False,
) -> None:
    data_shap = preprocess.transform(data)
    data_shap.rename(columns=cols_dict, inplace=True)
    compute_shap_values(model, data_shap, title_suffix, calibrated)


def evaluate_monthly_data(
    model: Pipeline,
    X_monthly: Dict[str, pd.DataFrame],
    y_monthly: Dict[str, pd.Series],
    estimator: BaseEstimator,
    preprocess: Pipeline,
    cols_dict: Dict[str, str],
    calibrated: bool = False,
) -> None:
    for month, X in X_monthly.items():
        y = y_monthly[month]
        evaluate_classification_performance(model, X, y, month)

        preprocess_and_compute_shap_values(
            estimator,
            X,
            preprocess,
            cols_dict,
            f"{month} Data",
            calibrated,
        )


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import FunctionTransformer

    # Генерация данных
    X, y = make_classification(n_samples=1000)
    X = pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Обучение модели
    model = RandomForestClassifier().fit(X_train, y_train)
    estimator = RandomForestClassifier().fit(X_train, y_train)
    preprocess = FunctionTransformer(lambda x: x).set_output(transform="pandas")

    # Эмуляция ежемесячных данных
    X_monthly = {"Январь": X_train[:200], "Февраль": X_train[200:400]}
    y_monthly = {"Январь": y_train[:200], "Февраль": y_train[200:400]}

    # Оценка производительности и вычисление значений SHAP
    evaluate_classification_performance(model, X_test, y_test)
    evaluate_monthly_data(model, X_monthly, y_monthly, estimator, preprocess, {})
