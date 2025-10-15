import pandas as pd


def get_forget_weights(date_col: pd.Series, coefficient: float) -> pd.Series:
    """
    Возвращает для примеров веса, обеспечивающие "забывание".

    Разбивает данные по месяцам и чем старее месяц, тем меньше коэффициент.

    Args:
        date_col: pd.Series с датами.
        coefficient: коэффициент "забывания".

    Returns:
        forget_weights: веса забывания примеров.
    """
    months = date_col.apply(lambda x: str(x)[:7])  # 2023-02
    months = months.replace(
        {month: i for i, month in enumerate(sorted(months.unique(), reverse=True))}
    )
    forget_weights = months.apply(lambda x: (1 - coefficient) ** x)

    return forget_weights
