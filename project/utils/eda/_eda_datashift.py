from typing import Optional, Tuple

import calplot
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ._eda_distribution import _slice_by_value_range


def ridge_plot(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    date_colname: str,
    value_range: Tuple[float | None, float | None] = (None, None),
    freq: str,
) -> None:
    """
    Визуализирует распределения численного непрерывного признака для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и поле с датами.
        feature_colname: название столбца с исследуемым признаком.
        date_colname: название столбца с датами.
        value_range: задаваемый диапазон рассматриваемых значений.
        freq: частота группировки по дате.
          Доступны 'D' для группировки по дням и 'M' - по месяцам.

    https://seaborn.pydata.org/examples/kde_ridgeplot.html
    https://www.python-graph-gallery.com/ridgeline-graph-seaborn
    """
    # подготовка данных
    data = data[[feature_colname, date_colname]].copy()
    data = _slice_by_value_range(data, feature_colname, value_range)
    length = {
        "M": 7,  # 2023-02
        "D": 10,  # 2023-02-10
    }
    data["gropby_index"] = (
        data[date_colname].astype(str).apply(lambda x: x[: length[freq]])
    )
    data = data.sort_values("gropby_index")

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be
    # represented by colors with 'palette'
    g = sns.FacetGrid(
        data,
        row="gropby_index",
        hue="gropby_index",
        aspect=13,
        height=0.75,
        palette=sns.cubehelix_palette(10, rot=-0.25, light=0.7),
    )
    # then we add the densities KDEplots for each month
    g.map(
        sns.kdeplot,
        feature_colname,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1,
    )
    # here we add a white line that represents the contour of each KDEplot
    g.map(
        sns.kdeplot,
        feature_colname,
        clip_on=False,
        color="w",
        linewidth=2,
    )
    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, feature_colname)

    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.75)

    # eventually we remove axes titles, yticks and spines
    g.set(title="", yticks=[], ylabel="")
    g.despine(bottom=True, left=True)


def area_plot(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    date_colname: str,
    freq: str,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """
    Визуализирует распределения численного дискретного или категориального признака для целевых
    классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и поле с датами.
        feature_colname: название столбца с исследуемым признаком.
        date_colname: название столбца с датами.
        freq: частота группировки по дате.
        ax: ранее созданный ax.

    Returns:
        ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    if not ax:
        ax = plt.subplot()

    grouped = (
        data.groupby([pd.Grouper(key=date_colname, freq=freq), feature_colname])[
            date_colname
        ]
        .count()
        .unstack()
    )
    normalized = (grouped.T / grouped.sum(axis=1)).T

    normalized.plot(kind="area", stacked=True, title=feature_colname, ax=ax)
    ax.legend(loc="lower left", facecolor="white")

    return ax


def na_datashift(
    data: pd.DataFrame,
    *,
    feature_colname: str,
    target_colname: str,
    date_colname: str,
    figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
):
    """
    Визуализирует доли пропусков в исследуемом признаке по дням в виде вафельной диаграммы.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак, целевую переменную и поле с датами.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        date_colname: название столбца с датами.
        figsize: (ширина, высота) рисунка в дюймах.

    Returns:
        Кортеж `(fig, ax)`.
          fig: matplotlib.Figure, содержащий все графики.
          axes: matplotlib.axes.Axes, содержащие отрисованные график.
    """
    # подготовка данных
    data = data[[feature_colname, target_colname, date_colname]].copy()
    data["has_na"] = data[feature_colname].isna()

    grouped = data.groupby(pd.Grouper(key=date_colname, freq="D"))

    # маска с днями, где есть хоть один выданный кредит
    mask_d = grouped[target_colname].count().loc[lambda x: x != 0].index
    # маска с днями, где есть хоть один пропуск в признаке
    mask_f = grouped["has_na"].mean().loc[lambda x: x != 0].index
    # финальная маска
    mask = mask_d.intersection(mask_f)

    fig, ax = calplot.calplot(
        grouped["has_na"].mean().loc[mask],
        how=None,
        dropzero=True,
        figsize=figsize,
        fillcolor="gray",
        cmap="YlGn",
        yearlabel_kws={"fontname": "sans-serif"},
        suptitle=f"Распределение доли пропусков в `{feature_colname}` по дням",
        suptitle_kws={"fontsize": 16},
    )

    return fig, ax
