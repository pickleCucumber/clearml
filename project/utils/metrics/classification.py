from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE = 1
NEGATIVE = 0


def my_binary_classification_report(
    classifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    threshold: Optional[float] = 0.5,
    title: str = "",
    *,
    classifier_name: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Предоставляет отчёт по бинарной классификации.

    Визуализирует матрицу ошибок, ROC- и Precision- и Recall- кривые, печатает
    `sklearn.classification_report` и индекс Gini.

    Args:
        classifier: классификатор.
        X: признаки.
        y_true: истинные метки.
        threshold: порог бинаризации.
        classifier_name: имя классификатора.
        figsize: (ширина, высота) рисунка в дюймах.
    """
    if figsize is None:
        figsize = (12.8, 10.6)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    y_proba = classifier.predict_proba(X)[:, 1]
    y_pred = np.where(y_proba >= threshold, POSITIVE, NEGATIVE)

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axes[0, 0], colorbar=False
    )
    axes[0, 0].set(title="Матрица ошибок")

    sns.histplot(data=y_proba, stat="density", kde=True, ax=axes[0, 1])
    axes[0, 1].set(xlabel="Probability")

    RocCurveDisplay.from_predictions(
        y_true, y_proba, name=classifier_name, ax=axes[1, 0], color="orange"
    )
    axes[1, 0].plot([0, 1], [0, 1], color="navy", linestyle="--")
    axes[1, 0].set(title="ROC-кривая", xlim=(-0.01, 1), ylim=(0, 1.01))

    # Графики зависимости Precision и Recall от порога бинаризации
    precision_recall_plot(y_true, y_proba, ax=axes[1, 1])
    axes[1, 1].set(title="Precision- и Recall-кривые")

    plt.title(f"Метрики {title}")
    plt.show()
    plt.close()

    print(classification_report(y_true, y_pred))

    # Gini_index: https://habr.com/ru/company/ods/blog/350440/
    print(f"Индекс Gini = {2 * roc_auc_score(y_true, y_proba) - 1}")


def precision_recall_plot(
    y_true: pd.Series,
    y_proba: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    nbin: int = 255,
) -> matplotlib.axes.Axes:
    """
    Рисует на заданном matplotlib.axes.Axes графики зависимости Precision и Recall от порога
    бинаризации.

    Args:
        y_true: истинные метки.
        y_proba: предсказанные вероятности отнесения к положительному классу.
        ax: matplotlib.axes.Axes, на котором следует отрисовать графики.
        nbin: количество бинов для равночастотного биннинга.

    Returns:
        ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    if ax is None:
        ax = plt.subplot()

    # равночастотный биннинг
    thresholds = np.interp(
        np.linspace(0, len(y_proba), nbin + 1),
        np.arange(len(y_proba)),
        np.sort(y_proba),
    )[1:-1]

    threshold_len = len(thresholds)
    precision_scores = np.empty(threshold_len, dtype=float)
    recall_scores = np.empty(threshold_len, dtype=float)

    for i, threshold in enumerate(thresholds):
        y_pred = np.where(y_proba >= threshold, POSITIVE, NEGATIVE)

        precision_scores[i] = precision_score(y_true, y_pred)
        recall_scores[i] = recall_score(y_true, y_pred)

    ax.plot(thresholds, precision_scores, color="red", label="precision")
    ax.plot(thresholds, recall_scores, color="blue", label="recall")
    ax.set(xlabel="величина порога")
    ax.legend()
    ax.grid()

    return ax


def psi_plot(
    x: np.ndarray,
    y: np.ndarray,
    xy1: tuple,
    xy2: tuple,
    title: Optional[str] = None,
    rotation: Optional[int] = None,
    x_label: Optional[int] = None,
    y_label: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Строит график PSI
    """
    if figsize is None:
        figsize = (12, 6)
    plt.figure(figsize=figsize)

    plt.plot(x, y, marker="o")
    plt.axline(xy1, xy2, color="red", linestyle="--")
    plt.xticks(rotation=rotation)
    plt.title(title)
    (plt.xlabel(x_label, fontsize=10),)
    plt.ylabel(y_label, fontsize=10)
