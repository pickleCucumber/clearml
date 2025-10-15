from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT = 1
NOT_DEFAULT = 0


def threshold_money_plot(
    data: pd.DataFrame,
    y_proba: np.ndarray,
    *,
    target_colname: str,
    loan_colname: str,
    interest_colname: str,
    nbin: int = 255,
) -> None:
    """
    Визуализирует зависимость коммерческой значимости от величины порога бинаризации.

    Args:
        data: DataFrame, содержащий колонки с истинными метками, телом кредита и телом процентов.
        y_proba: предсказанные вероятности дефолта.
        target_colname: имя колонки с истинными метками.
        loan_colname: имя колонки с телом кредита.
        interest_colname: имя колонки с телом процентов.
        nbin: количество бинов для равночастотного биннинга.
    """
    # равночастотный биннинг
    thresholds = np.interp(
        np.linspace(0, len(y_proba), nbin + 1),
        np.arange(len(y_proba)),
        np.sort(y_proba),
    )[1:-1]
    threshold_len = len(thresholds)

    saves = np.empty(threshold_len, dtype=float)
    loses = np.empty(threshold_len, dtype=float)
    deltas = np.empty(threshold_len, dtype=float)
    approval_rate = np.empty(threshold_len, dtype=float)

    for i, threshold in enumerate(thresholds):
        y_pred = np.where(y_proba >= threshold, DEFAULT, NOT_DEFAULT)

        save, lose, delta = commercial_impact(
            data,
            y_pred,
            target_colname=target_colname,
            loan_colname=loan_colname,
            interest_colname=interest_colname,
        )

        saves[i] = save
        loses[i] = lose
        deltas[i] = delta

        approval_rate[i] = 1 - y_pred.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(thresholds, saves, label="сохранённые потери")
    axes[0].plot(thresholds, loses, label="потерянная прибыль")
    axes[0].plot(thresholds, deltas, label="delta")
    axes[0].set(xlabel="величина порога", ylabel="денежки")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(thresholds, approval_rate, color="red", label="approval rate")
    axes[1].set(xlabel="величина порога", ylabel="approval rate")
    axes[1].legend()
    axes[1].grid()

    plt.show()


def commercial_impact(
    data: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    target_colname: str,
    loan_colname: str,
    interest_colname: str,
    report: Optional[bool] = False,
) -> Tuple:
    """
    Возвращает/печатает суммы сохранённых потерь, упущенной прибыли и коммерческой значимости
    модели.

    Args:
        data: DataFrame, содержащий колонки с истинными метками, телом кредита и телом процентов.
        y_pred: list-like с предсказанными метками.
        target_colname: имя колонки с истинными метками.
        loan_colname: имя колонки с телом кредита.
        interest_colname: имя колонки с телом процентов.
        report: True - печатает, False - возвращает.

    Returns:
        Кортеж `(save, lose, delta)`
          save: сохранённые потери
          lose: упущенная прибыль
          delta: коммерческая значимость
    """
    data = data[[target_colname, loan_colname, interest_colname]].copy()
    data["pred"] = y_pred

    TP_mask = (data[target_colname] == DEFAULT) & (data["pred"] == DEFAULT)
    FP_mask = (data[target_colname] == NOT_DEFAULT) & (data["pred"] == DEFAULT)

    save = int(data.loc[TP_mask, loan_colname].sum())
    lose = int(data.loc[FP_mask, interest_colname].sum())
    delta = save - lose

    if report:
        print(
            f"Сохранили потерь {_insert_(save)}\n"
            f"Упустили прибыли {_insert_(lose)}\n"
            f"Коммерческая значимость {_insert_(delta)}"
        )

    return save, lose, delta


def _insert_(num: int) -> str:
    result = str(num)[::-1]
    result = "_".join(result[i : i + 3] for i in range(0, len(result), 3))
    result = result[::-1]

    return result


def clossest_approval_rate(y_proba, best_ar=0.9):
    """
    Возвращает порог, при котором approval rate ближайший к best_ar

    Args:
        y_proba: list-like с предсказанными метками.
        best_ar: approval rate по которому считать порог

    Returns:
        float: порог
    """
    thresholds = np.sort(y_proba)
    app_rate_all_thresholds = 1 - (thresholds > thresholds.reshape(-1, 1)).mean(axis=1)
    return thresholds[abs(app_rate_all_thresholds - best_ar).argmin()]


def approval_rate_plot(
    y_proba: np.ndarray,
    *,
    nbin: int = 1000,
    lower_bound: Optional[float] = 0,
    upper_bound: Optional[float] = 1,
    best_ar: Optional[float] = 0.9,
    title: str = "",
) -> float:
    """
    Визуализирует зависимость approval rate от величины порога бинаризации, а также печатает оптимальный порог для 1 - AP = 90%

    Args:
        data: DataFrame, содержащий колонки с истинными метками, телом кредита и телом процентов.
        y_proba: list-like с предсказанными метками.
        target_colname: имя колонки с истинными метками.
        nbin: количество биннов для равночастотного биннинга.
        lower_bound: нижний порог бинаризации.
        upper_bound: верхний порог бинаризации.
        best_ar: approval rate по которому считать порог бинаризации
    """

    # равночастотный биннинг
    mask = (y_proba > lower_bound) & (y_proba < upper_bound)
    thresholds = np.interp(
        np.linspace(0, len(y_proba[mask]), nbin + 1),
        np.arange(len(y_proba[mask])),
        np.sort(y_proba[mask]),
    )[1:-1]
    threshold_len = len(thresholds)

    approval_rate = np.empty(threshold_len, dtype=float)

    for i, threshold in enumerate(thresholds):
        y_pred = np.where(y_proba >= threshold, 1, 0)
        approval_rate[i] = 1 - y_pred.mean()

    fig, axes = plt.subplots(figsize=(12, 4))

    axes.plot(thresholds, approval_rate, color="red", label="approval rate")
    axes.set(xlabel="величина порога", ylabel="approval rate")
    axes.legend()
    axes.grid()
    axes.set_title(f"{title} approval rate")

    fig.show()
    plt.close()

    best_proba = clossest_approval_rate(y_proba, best_ar)
    print(f"лучший порог: {best_proba}")
    return best_proba


if __name__ == "__main__":
    print("This module is not for direct call!")
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "target": np.random.randint(0, 2, size=1000),
        }
    )
    approve_rate = 0.9
    # Generate some random probabilities
    y_proba = np.random.rand(1000)

    # Call the function
    threshold = approval_rate_plot(
        y_proba=y_proba, nbin=1000, lower_bound=0, upper_bound=1, best_ar=approve_rate
    )
    print(f"Функция возвращает {1 - (y_proba > threshold).mean()} +- = {approve_rate}")
