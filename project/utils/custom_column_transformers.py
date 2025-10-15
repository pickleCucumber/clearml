from functools import wraps

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AddDif(BaseEstimator, TransformerMixin):
    """Добавляет разность заданных признаков к остальным."""

    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        dif_feature_name = f"{X.iloc[:, 0].name}_{X.iloc[:, 1].name}_dif"
        self.columns = np.array(X.columns.to_list() + [dif_feature_name])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[f"{X.iloc[:, 0].name}_{X.iloc[:, 1].name}_dif"] = X.iloc[:, 0] - X.iloc[:, 1]

        return X

    def get_feature_names_out(self, *args, **params):
        return self.columns


class AddRatio(BaseEstimator, TransformerMixin):
    """Добавляет отношение заданных признаков к остальным."""

    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        ratio_feature_name = f"{X.iloc[:, 0].name}_{X.iloc[:, 1].name}_ratio"
        self.columns = np.array(X.columns.to_list() + [ratio_feature_name])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numerator = X.iloc[:, 0]
        denominator = X.iloc[:, 1]
        ratio = numerator / denominator

        X[f"{numerator.name}_{denominator.name}_ratio"] = ratio

        return X

    def get_feature_names_out(self, *args, **params):
        return self.columns


class AsCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = np.array(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.columns:
            X.loc[:, feature] = pd.Categorical(X[feature])

        return X

    def get_feature_names_out(self, *args, **params):
        return self.columns


class GenAge(BaseEstimator, TransformerMixin):
    """Генерирует и добавляет возраст к остальным признакам."""

    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        self.columns = np.array(["age"])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        app_date = pd.to_datetime(X.iloc[:, 0])
        birthdate = pd.to_datetime(X.iloc[:, 1])
        X["age"] = (app_date - birthdate) / np.timedelta64(1, "Y")

        return pd.DataFrame(X["age"])

    def get_feature_names_out(self, *args, **params):
        return self.columns


# для старой версии seconds, для новой - month, hour
class GenDataFromDatetime(BaseEstimator, TransformerMixin):
    """Генерирует и добавляет признаки из даты к остальным."""

    def __init__(
        self,
        day=True,
        weekday=True,
        hour=True,
        month=False,
    ):
        if not any((day, weekday, month, hour)):  # seconds
            raise ValueError(
                "что-то из day, weekday, hour, month должно быть True"
            )  # seconds

        self.columns = None
        self.day = day
        self.weekday = weekday
        self.month = month
        self.hour = hour
        # self.seconds = seconds

    def fit(self, X, y=None):
        cols = []
        if self.day:
            cols.append("day")
        if self.weekday:
            cols.append("weekday")
        if self.month:
            cols.append("month")
        if self.hour:
            cols.append("hour")
        # if self.seconds:
        #     cols.append('seconds')
        self.columns = np.array(cols)

        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        app_date = pd.to_datetime(X)

        result = pd.DataFrame(index=app_date.index)
        if self.day:
            result["day"] = pd.Categorical(
                app_date.dt.day, categories=list(range(1, 32))
            )
        if self.weekday:
            result["weekday"] = pd.Categorical(
                app_date.dt.weekday, categories=list(range(1, 8))
            )
        if self.month:
            result["month"] = pd.Categorical(
                app_date.dt.month, categories=list(range(1, 13))
            )
        if self.hour:
            result["hour"] = pd.Categorical(
                app_date.dt.hour, categories=list(range(24))
            )
        # if self.seconds:
        #     result['seconds'] = pd.to_timedelta(app_date.dt.time.astype(str)).dt.total_seconds()

        return result

    def get_feature_names_out(self, *args, **params):
        return self.columns


class GenInterest(BaseEstimator, TransformerMixin):
    """Генерирует и добавляет проценты к остальным признакам."""

    def __init__(self, interest_rate: float):
        self.columns = None
        self.interest_rate = interest_rate

    def fit(self, X, y=None):
        self.columns = np.array(X.columns.to_list() + ["interest"])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["interest"] = (X.iloc[:, 0] * self.interest_rate * X.iloc[:, 1]).round(2)

        return X

    def get_feature_names_out(self, *args, **params):
        return self.columns


def function_transformer_available(function):
    """Оборачивает функцию, работающую с содержимым ячейки pd.Series, в работающую с самим
    pd.Series."""

    @wraps(function)
    def wrapper(t: pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(t, pd.Series):
            t = pd.DataFrame(t)

        for col in t.columns:
            t[col] = t[col].apply(function)

        return t

    return wrapper
