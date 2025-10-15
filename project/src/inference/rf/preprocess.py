from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd

TYPES = {
    "dtstart": "datetime64[ns]",
    "sex": "float64",
    "birthday": "datetime64[ns]",
    "citizenshipid": "int64",
    "martialid": "float64",
    "dependents": "int64",
    "sitename": "object",
    "DOC": "float64",
    "averagemonthlyincome": "int64",
    "Days_since_last_credit": "float64",
    "Max_overdue": "float64",
    "Nb_delays_90plus_ever_eq": "float64",
    "CH_length_eq": "float64",
    "S_hare_active_credit": "float64",
    "Score": "float64",
    "MatchingLevel": "float64",
    "INTEGRALSCOREValueId": "float64",
    "LIFETIMEBINValueId": "float64",
    "requested_amount": "float64",
}


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        self._model_name = 'RF'
        self._model = None
        self.threshold = 0.31148091786263643  # получаем на этапе eval
        self.dtypes = TYPES

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        self._model = joblib.load(local_file_name)

    def preprocess(
        self, body: dict, state: dict, collect_custom_statistics_fn=None
    ) -> Any:
        print('Model:', self._model_name)
        print(body)
        # row_json = body["row"]

        # row = pd.read_json(row_json, typ="series")
        row = body["row"]
        df = pd.DataFrame([row])

        errors = []

        for col, dtype in self.dtypes.items():
            try:
                df[col] = df[col].astype(dtype)
            except ValueError:
                # В случае, если преобразование невозможно из-за несовместимости значений,
                # вы можете выбрать, что делать: игнорировать ошибку, выводить предупреждение или применять запасной план.
                error_message = f"Внимание: Не удалось преобразовать столбец '{col}' к типу {dtype}."
                print(error_message)
                errors.append(error_message)

            except TypeError:
                # Этот тип исключения может возникнуть, если попытка приведения типа вообще не имеет смысла.
                # Например, попытка привести строку к целочисленному типу, когда в строке содержится текст.
                error_message = (
                    f"Ошибка: Приведение типа для '{col}' не поддерживается."
                )
                print(error_message)
                errors.append(error_message)

            except KeyError:
                # Возникает, если указанный ключ (в нашем случае имя столбца) отсутствует в DataFrame.
                error_message = f"Ошибка: Столбец '{col}' не найден в DataFrame."
                print(error_message)
                errors.append(error_message)

            except pd.errors.IntCastingNaNError:
                # Это специфичная для pandas ошибка, возникающая при попытке привести столбец с NaN значениями к целочисленному типу.
                error_message = f"Внимание: Невозможно преобразовать столбец '{col}' в целочисленный тип из-за наличия NaN значений."
                print(error_message)
                errors.append(error_message)

            except Exception as e:
                # Общий перехватчик для всех неучтенных типов исключений.
                error_message = f"Необработанное исключение при преобразовании столбца '{col}': {str(e)}"
                print(error_message)
                errors.append(error_message)

        state["errors"] = errors

        return df

    def process(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:
        try:
            transform_data = self._model[0].transform(data)
            if collect_custom_statistics_fn:
                collect_custom_statistics_fn(transform_data.iloc[0].to_dict())
            prediction = self._model[1].predict_proba(transform_data)[:, 1]
            print(prediction)
            return prediction

        except Exception as e:
            state["errors"].append(f"Processing error: {str(e)}")
            return None  # or appropriate default

    def postprocess(
        self, data: Any, state: dict, collect_custom_statistics_fn=None
    ) -> dict:
        output = {"Probability": None, "Result": None, "Threshold": self.threshold}
        if data is not None:
            output["Probability"] = data[0]
            output["Result"] = int(data[0] > self.threshold)

        # Add errors to the output
        if "errors" in state and state["errors"]:
            output["Errors"] = state["errors"]
        else:
            output["Errors"] = []

        print(output)
        return output
