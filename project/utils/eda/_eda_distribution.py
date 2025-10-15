from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def cat_feature_report(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    target_colname: str,
    figsize: Optional[Tuple[float, float]] = None,
    x_rot: Optional[int | float] = 0,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Визуализирует разницу в распределениях категориального признака для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        figsize: (ширина, высота) рисунка в дюймах.
        x_rot: угол наклона xticklabels.

    Returns:
        Кортеж (fig, ax).
          fig: matplotlib.Figure, содержащий все графики.
          axes: matplotlib.axes.Axes, содержащие отрисованные график.
    """
    data = data[[feature_colname, target_colname]].copy()

    # если есть пропуски
    if data[feature_colname].isna().sum():
        if figsize is None:
            figsize = (19.2, 4.8)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # график, где пропуски рассматриваются как доп. категория
        tmp_data = data.copy()
        if tmp_data[feature_colname].dtype.name == "category":
            tmp_data[feature_colname] = tmp_data[feature_colname].cat.add_categories(
                ["пропуск"]
            )
        tmp_data[feature_colname] = tmp_data[feature_colname].fillna("пропуск")
        _bar_plot(
            tmp_data,
            feature_colname=feature_colname,
            target_colname=target_colname,
            x_rot=x_rot,
            ax=axes[1],
        )
        del tmp_data

        # график, где исследуется распределение значение/пропуск
        na_bar_plot(
            data,
            feature_colname=feature_colname,
            target_colname=target_colname,
            x_rot=x_rot,
            ax=axes[2],
        )
    else:
        if figsize is None:
            figsize = (6.4, 4.8)

        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]  # заглушка

    # график, где рассматривается распределение только значений, т.е. без пропусков
    data = data.dropna(subset=feature_colname)
    _bar_plot(
        data,
        feature_colname=feature_colname,
        target_colname=target_colname,
        x_rot=x_rot,
        ax=axes[0],
    )
    axes[0].set(xlabel="Целевые классы", ylabel="Доли категорий")
    fig.suptitle(f"cat_feature_report для признака {feature_colname}")

    return fig, axes


def num_feature_report(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    target_colname: str,
    value_range: Tuple[float | None, float | None] = (None, None),
    figsize: Optional[Tuple[float, float]] = None,
    x_rot: Optional[int | float] = 0,
    histplot_args: Optional[Dict[Any, Any]] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Визуализирует разницу в распределениях численного непрерывного признака для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        value_range: задаваемый диапазон рассматриваемых значений.
        figsize: (ширина, высота) рисунка в дюймах.
        x_rot: угол наклона xticklabels.
        histplot_args: аргументы для seaborn.histplot().

    Returns:
        Кортеж (fig, axes).
          fig: matplotlib.Figure, содержащий все графики.
          axes: matplotlib.axes.Axes, содержащие отрисованные график.
    """
    # подготовка данных
    data = data[[feature_colname, target_colname]].copy()
    data = _slice_by_value_range(data, feature_colname, value_range)

    if histplot_args is None:
        histplot_args = {}

    # если есть пропуски
    if data[feature_colname].isna().sum():
        if figsize is None:
            figsize = (19.2, 4.8)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        na_bar_plot(
            data,
            feature_colname=feature_colname,
            target_colname=target_colname,
            x_rot=x_rot,
            ax=axes[2],
        )
    else:
        if figsize is None:
            figsize = (12.8, 4.8)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Violinplot
    sns.violinplot(
        data=data, x=feature_colname, y=target_colname, orient="h", ax=axes[0]
    )
    axes[0].set(title="Violinplot")

    # Density Histogram
    sns.histplot(
        data=data,
        x=feature_colname,
        hue=target_colname,
        stat="density",
        common_norm=False,
        ax=axes[1],
        **histplot_args,
    )
    axes[1].set(title="Density Histogram")

    fig.suptitle(f"num_feature_report для признака {feature_colname}")

    return fig, axes


def _slice_by_value_range(
    data: pd.DataFrame,
    feature_colname: str,
    value_range: Tuple[float | None, float | None] = (None, None),
) -> pd.DataFrame:
    """
    Возвращает срез pd.DataFrame по задаваемому признаку и диапазону значений.

    Args:
        data: pd.DataFrame, который необходимо обрезать.
        feature_colname: название столбца с признаком, по которому необходимо сделать срез.
        value_range: задаваемый диапазон рассматриваемых значений.

    Returns:
        data: срезанный pd.DataFrame.
    """
    data = data.copy()

    min_value = value_range[0]
    max_value = value_range[1]
    if min_value:
        data = data[data[feature_colname] >= min_value]
    if max_value:
        data = data[data[feature_colname] <= max_value]

    return data


def na_bar_plot(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    target_colname: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    x_rot: Optional[int | float] = 0,
) -> matplotlib.axes.Axes:
    """
    Визуализирует разницу в доле пропусков для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        ax: matplotlib.axes.Axes, на котором следует отрисовать график.
        x_rot: угол наклона xticklabels.

    Returns:
        ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    data = data[[feature_colname, target_colname]].copy()

    if ax is None:
        ax = plt.subplot()

    data["has_na"] = (
        data[feature_colname].isna().replace({True: "пропуск", False: "значение"})
    )
    del data[feature_colname]

    _bar_plot(
        data,
        feature_colname="has_na",
        target_colname=target_colname,
        x_rot=x_rot,
        ax=ax,
    )
    ax.set(ylabel="Доли пропусков")

    return ax


def _bar_plot(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    target_colname: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    x_rot: Optional[int | float] = 0,
) -> matplotlib.axes.Axes:
    """
    Визуализирует распределение значений признака в виде столбчатой диаграммы для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        ax: ранее созданный ax.
        x_rot: угол наклона xticklabels.

    Returns:
        ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    data = data[[feature_colname, target_colname]].copy()

    labels = sorted(data[target_colname].unique())
    categories = list(data[feature_colname].unique())
    if "пропуск" in categories:
        categories.remove("пропуск")
        categories = sorted(categories) + ["пропуск"]
    else:
        categories = sorted(categories)

    if ax is None:
        ax = plt.subplot()

    shape = len(categories), len(labels) + 1
    a = np.empty(shape, dtype=float)
    for i, category in enumerate(categories):
        category_ratios = np.empty(shape[1], dtype=float)
        for j, label in enumerate(labels):
            class_df = data[data[target_colname] == label]
            all_count = class_df.shape[0]
            category_count = class_df[class_df[feature_colname] == category].shape[0]
            ratio = category_count / all_count
            category_ratios[j] = ratio

        all_count = data.shape[0]
        category_count = data[data[feature_colname] == category].shape[0]
        ratio = category_count / all_count
        category_ratios[-1] = ratio

        a[i] = category_ratios

    labels = [str(label) for label in labels] + ["весь датасет"]

    bottom = np.zeros(len(labels))
    for i, category in enumerate(categories):
        ax.bar(labels, a[i], bottom=bottom, label=category)
        bottom += a[i]

    # add annotations
    for c in ax.containers:
        # customize the label to account for cases when there might not be a bar section
        labels = [f"{h:2.3%}" if (h := v.get_height()) > 0.05 else "" for v in c]

        # set the bar label
        ax.bar_label(c, labels=labels, label_type="center", fontsize=8)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1, 1.05), title="Категории")
    ax.set(xlabel="Целевые классы", ylabel="Доли категорий")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=x_rot, ha="right")

    return ax
