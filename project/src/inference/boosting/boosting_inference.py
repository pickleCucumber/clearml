from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
import xmltodict


THRESHOLD = 0.62
DEFAULT = 1
NOT_DEFAULT = 0

ALL_FEATURES = [
    'dtstart',
    'birthday',
    'sex',
    'citizenshipid',
    'martialid',
    'dependents',
    'sitename',
    'DOC',
    'averagemonthlyincome',
    'requested_amount',

    # nbki
    'Days_since_last_credit',
    'Max_overdue',

    #equifax
    'Nb_delays_90plus_ever_eq',
    'CH_length_eq',
    'S_hare_active_credit',

    # megafon/mail
    'Score',
    'MatchingLevel',
    'LIFETIMEBINValueId',
    'INTEGRALSCOREValueId',
]

NOT_NA_FEATURES = [
    'dtstart',
    'birthday',
    'dependents',
    'sitename',
    'averagemonthlyincome',
    'requested_amount',
]
FLOAT_FEATURES = [
    'sex',
    'martialid',
    'DOC',
    'Days_since_last_credit',
    'Max_overdue',
    'Nb_delays_90plus_ever_eq',
    'CH_length_eq',
    'S_hare_active_credit',
    'Score',
    'MatchingLevel',
    'INTEGRALSCOREValueId',
    'LIFETIMEBINValueId',
    'requested_amount'
]

CAT_FEATURES = [
    'dtstart',
    'birthday',
    'sitename',
]

INT_FEATURES = [
    'citizenshipid',
    'dependents',
    'averagemonthlyincome',
]

def _xml_to_df(xml_input: str) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Конвертирует XML-строку в pd.DataFrame.

    Args:
        xml_input: XML-строка, содержащая входной вектор.

    Returns:
        Кортеж `(df, messages)`.
          df: pd.DataFrame, содержащий входной вектор.
          messages: сообщение об ошибках в данных.
          codes: коды сообщений об ошибках в данных.
    """
    messages = []
    codes = []

    dict_ = xmltodict.parse(xml_input)
    dict_ = dict_['REQUEST']

    del dict_['InfoType']
    del dict_['@xmlns:xsi']

    for i in ['RequestId', 'SolutionId', 'OrderId']:
        del dict_[i]

    extra_features = []
    for feature in dict_:
        if feature not in ALL_FEATURES:
            extra_features.append(feature)
            messages.append(f'в данных присутствует лишний признак `{feature}`')
            codes.append(0)
            
    for feature in extra_features:
        del dict_[feature]

    missing_feature = []
    for feature in ALL_FEATURES:
        if feature not in dict_.keys():
            missing_feature.append(feature)
            messages.append(f'в данных отсутствует необходимый признак `{feature}`')
            codes.append(0)

    # for old_name, new_name in NAME_MAP.items():
    #     try:
    #         dict_[new_name] = dict_.pop(old_name)
    #     except KeyError:
    #         messages.append(f'в данных отсутствует необходимый признак `{old_name}`(`{new_name}`)')
    #         codes.append(1)
    #         dict_[new_name] = None

    # замена XML-nil на None

    for feature, value in dict_.items():
        if value == {'@xsi:nil': 'true'}:
            dict_[feature] = None

    for not_na_feature in NOT_NA_FEATURES:
        try:
            if dict_[not_na_feature] is None:  # TODO
                messages.append(
                    f'признак `{not_na_feature}` должен иметь значение (отличное от "nil") (NOT_NA_FEATURES)')
                codes.append(2)
        except KeyError:
             messages.append(
                 f'отстутствует необходимый признак {not_na_feature} (NOT_NA_FEATURES)'
             )
             codes.append(10)

    for float_feature in FLOAT_FEATURES:
        try:
            if dict_[float_feature] is None:
                dict_[float_feature] = float('nan')
            else:
                dict_[float_feature] = float(dict_[float_feature])
        except ValueError:
            messages.append(
                f'невозможно преобразовать к float значение `{dict_[float_feature]}` '
                f'признака `{float_feature}`'
            )
            codes.append(3)
        except KeyError:
             messages.append(
                 f'отстутствует необходимый признак {float_feature} (FLOAT_FEATURES)'
             )
             codes.append(10)

    for int_feature in INT_FEATURES:
        try:
            dict_[int_feature] = int(dict_[int_feature])
        except ValueError:
            messages.append(
                f'невозможно преобразовать к float значение `{dict_[int_feature]}` '
                f'признака `{int_feature}`'
            )
            codes.append(3)
        except KeyError:
             messages.append(
                 f'отстутствует необходимый признак {int_feature} (FLOAT_FEATURES)'
             )
             codes.append(10)


    # for datetime_feature in DATETIME_FEATURES:
    #     try:
    #       if dict_[datetime_feature] is not None:
    #         pd.to_datetime(dict_[datetime_feature])
    #     except ValueError:
    #        messages.append(
    #            f'Значение `{dict_[datetime_feature]}` признака `{datetime_feature}` невозможно '
    #            'привести к datetime'
    #        )
    #        codes.append(5)
    #     except KeyError:
    #        messages.append(
    #            f'отстутствует необходимый признак {datetime_feature} (DATETIME_FEATURES)'
    #        )
    #        codes.append(10)

    # оборачиваем значения словаря в списки для дальнейшей конвертации в pd.DataFrame
    for feature in dict_:
        dict_[feature] = [dict_[feature]]

    df = pd.DataFrame.from_dict(dict_)

    return df, messages, codes


def _gen_xml_output(
        pred: int,
        proba: float,
        threshold: float,
        messages: List[str],
        codes: List[int],
) -> str:
    """
    Формирует ответ в виде XML-строки.

    Args:
        pred: предсказание модели (1 - дефолт, 0 - не дефолт).
        proba: "вероятность" дефолта (значение от 0 до 1).
        threshold: порог бинаризации.
        messages: список сообщений с ошибками.
        codes: коды сообщений об ошибках в данных.

    Returns:
        root: XML-строка, содержащая предсказание модели, "вероятность" дефолта, порог бинаризации и
          сообщения об ошибках (если имеются).
    """
    body = f"""<Result>{pred}</Result>
    <Probability>{proba}</Probability>
    <Threshold>{threshold}</Threshold>"""

    for message, code in zip(messages, codes):
        body += f"""
    <Error>
        <Message>{message}</Message>
        <Code>{code}</Code>
    </Error>"""

    root = f"""<RESPONSE>
    {body}
</RESPONSE>"""

    return root


def exec_ml(xml_input: str, model: Pipeline) -> str:
    """
    Принимает XML-строку, содержащую входной вектор, и возвращает предсказание и вероятность
    дефолта, а также сообщение об ошибках в данных.

    Args:
        xml_input: XML-строка, содержащая входной вектор.
        model: модель классификации.

    Returns:
        Кортеж `(default_prediction, default_probability, message)`
          default_prediction: предсказание модели (1 - дефолт, 0 - не дефолт).
          default_probability: "вероятность" дефолта (значение от 0 до 1).
          message: сообщение об ошибках в данных.
    """
    data_point, messages, codes = _xml_to_df(xml_input)

    try:
        default_probability = model.predict_proba(data_point)[0, 1]
        default_prediction = DEFAULT if default_probability >= THRESHOLD else NOT_DEFAULT
    except:
        messages.append('Падение на уровне модели')
        codes.append(6)
        default_prediction = -1
        default_probability = -1

    xml_output = _gen_xml_output(
        default_prediction, default_probability, THRESHOLD, messages, codes)

    return xml_output
