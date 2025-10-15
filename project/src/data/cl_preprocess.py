import numpy as np
import pandas as pd
import re

# import torch
# from transformers import AutoTokenizer, AutoModel

from sklearn.base import BaseEstimator, TransformerMixin
from utils.custom_column_transformers import function_transformer_available

# import nltk
# nltk.download("stopwords")
# stopwords = nltk.corpus.stopwords.words("russian")


# @function_transformer_available
def foo_dtype(x):
    return x.astype("category")


@function_transformer_available
def foo_sex(x):
    return x if x in [1, 2] else None


# @function_transformer_available
# def foo_phone_prefix(x):
#     x = str(x)
#     x = re.sub("\W+", "", x)
#     x = x[1:] if x.startswith("7") else x

#     if x in ["nan", " null", "none", ""]:
#         x = None

#     return x


POPULAR_DOMAINS = [
    "gmail",
    "mail",
    "bk",
    "maximus",
    "yandex",
    "interc",
    "gk-prime",
    "list",
    "riteiler",
    "icloud",
    "net",
    "inbox",
    "ya",
    "interantenna",
    "mob-reshenie",
    "unionshops",
    "petro-park",
    "ukplus",
    "scat-mobile",
    "rambler",
    "megafon",
    "vk",
    "t2retail",
    "diler-centr",
    "t2distrib",
    "ural-retail",
    "westtelecom",
    "mega",
    "mob-mir",
    "cor-all",
    "retail",
    "vse-svoi",
    "monobrend",
    "tele2",
    "organicreligion",
    "telebox",
    "mf-partner",
    "none",
    "nan",
    "null",
    "",
]


@function_transformer_available
def foo_mail_domain(x):
    x = str(x).strip().lower()

    if x in ["none", "nan", "null", ""]:
        x = None
    else:
        x = x.split(".")[0]

    return x


# class GenMailError(BaseEstimator, TransformerMixin):
#     """TODO."""
#     def __init__(self):
#         self.columns = None

#     def fit(self, x, y=None):
#         """TODO."""
#         self.columns = np.array(["mail_error"])

#         return self

#     def transform(self, x):
#         """TODO."""
#         result = pd.DataFrame()
#         result["mail_error"] = x.apply(foo_mail_errors)

#         return result

#     def get_feature_names_out(self, *args, **params):
#         """TODO."""
#         return self.columns


# def foo_mail_errors(x):
#     if x:
#         if x in ["nan", "null", "none", ""]:
#             x = None
#         elif x in POPULAR_DOMAINS:
#             x = 0
#         else:
#             x = 1

#     return x


@function_transformer_available
def foo_group_domain(x):
    domain_dict = {
        "0gmail": "mail",
        "508gmail": "gmail",
        "616gmail": "gmail",
        "658gmail": "gmail",
        "6gmail": "gmail",
        "9gmail": "gmail",
        "cmail": "gmail",
        "dmail": "gmail",
        "gail": "gmail",
        "gameil": "gmail",
        "gami": "gmail",
        "gamil": "gmail",
        "gemail": "gmail",
        "gimail": "gmail",
        "gimal": "gmail",
        "gma": "gmail",
        "gmael": "gmail",
        "gmai": "gmail",
        "gmaii": "gmail",
        "gmaiil": "gmail",
        "gmail": "gmail",
        "gmailc": "gmail",
        "gmaill": "gmail",
        "gmal": "gmail",
        "gmali": "gmail",
        "gmaul": "gmail",
        "gmeil": "gmail",
        "gmil": "gmail",
        "gmill": "gmail",
        "gmsil": "gmail",
        "gnail": "gmail",
        "iuogmail": "gmail",
        "jmail": "gmail",
        "jmali": "gmail",
        "qmail": "gmail",
        "123mail": "mail",
        "29mail": "mail",
        "88mail": "mail",
        "email": "mail",
        "emil": "mail",
        "imail": "mail",
        "ma": "mail",
        "mai": "mail",
        "maik": "mail",
        "mail": "mail",
        "maile": "mail",
        "mailru": "mail",
        "mal": "mail",
        "mali": "mail",
        "maul": "mail",
        "mil": "mail",
        "nail": "mail",
        "17bk": "bk",
        "95bk": "bk",
        "maxsimus": "maximus",
        "naximus": "maximus",
        "ayndex": "yandex",
        "yanex": "yandex",
        "pk-prime": "gk-prime",
        "rileiler": "riteiler",
        "riteile": "riteiler",
        "icloub": "icloud",
        "iclout": "icloud",
        "iclud": "icloud",
        "icoud": "icloud",
        "neet": "net",
        "netu": "net",
        "inbo": "inbox",
        "mobreshenie": "mob-reshenie",
        "unionashops": "unionshops",
        "petropark": "petro-park",
        "ya": "ya",
        "ukplus": "ukplus",
        "scat-mobile": "scat-mobile",
        "rambler": "rambler",
        "megafon": "megafon",
        "vk": "vk",
        "t2retail": "t2retail",
        "t2distrib": "t2distrib",
        "ural-retail": "ural-retail",
        "westtelecom": "westtelecom",
        "diler-centr": "diler-centr",
        "mega": "mega",
        "mob-mir": "mob-mir",
        "cor-all": "cor-all",
        "retail": "retail",
        "vse-svoi": "vse-svoi",
        "monobrend": "monobrend",
        "tele2": "tele2",
        "organicreligion": "organicreligion",
        "telebox": "telebox",
        "mf-partner": "mf-partner",
    }

    if x:
        # исправляет все найденные ошибки
        if x in domain_dict.keys():
            x = domain_dict[x]

        # остальные почты группирует в 'undefined' (NaN остаётся NaN)
        if x not in POPULAR_DOMAINS:
            x = "undefined"

    return x


@function_transformer_available
def foo_nameto(x):
    x = x.lower()

    if x.startswith("ооо"):
        x = x.replace("ооо", "")
    elif x.startswith("ао"):
        x = x.replace("ао", "")
    elif x.startswith("зао"):
        x = x.replace("зао", "")
    elif x.startswith("оао"):
        x = x.replace("оао", "")
    elif x.startswith("пао"):
        x = x.replace("пао", "")
    elif x.startswith("ип"):
        x = "ип"

    x = re.sub("\W+", " ", x)
    x = x.strip()

    if x.startswith("новая связь "):
        x = "новая связь"
    elif x.startswith("заря "):
        x = "заря"
    elif x.startswith("тренд телеком"):
        x = "тренд телеком"
    elif x.startswith("домотехника"):
        x = "домотехника"
    elif x.startswith("комфорт телеком"):
        x = "комфорт телеком"
    elif x.startswith("выдача займа"):
        x = "займ"
    elif x.startswith("дилер "):
        x = "дилер"
    elif x.startswith("мегафон"):
        x = "мегафон"

    return x


@function_transformer_available
def foo_sitename(x):

    group_sitename = [
        'tele2',
        'maximus',
        'beeline',
        'mts',
        'интерком интерком',
        'xiaomi',
        'реал связь',
        'broker',
        'ноу хау',
        'ип',
        'хорошая связь',
        'cpoint',
        'скат',
        'strela',
    ]

    x = str(x).lower()

    if x.startswith("ooo"):
        x = x.replace("ooo", "")
    elif x.startswith("ооо"):
        x = x.replace("ооо", "")
    elif x.startswith("гк"):
        x = x.replace("гк", "")
    elif x.startswith("ип"):
        x = "ип"
    elif x.startswith("ip"):
        x = "ип"
    elif x.startswith("билайн"):
        x = "ип"
    elif "cpoint" in x:
        x = "cpoint"
    elif x.startswith("мегафон"):
        x = "megafon"

    x = x.replace("www.", "")
    x = x.replace(".ru", "")
    x = x.replace(".com", "")
    x = re.sub("\W+", " ", x)
    x = x.strip()

    if x in ["nan", " null", "none", ""]:
        x = None

    if x not in group_sitename:
        x = 'small_group'

    return x


class GenIsSite(BaseEstimator, TransformerMixin):
    """TODO."""
    def __init__(self):
        self.columns = None

    def fit(self, x, y=None):
        """TODO."""
        self.columns = np.array([f"{x.name}", "is_site"])
        # self.columns = np.array(["is_site"])
        return self

    def transform(self, x):
        """TODO."""
        sites = [
            "www.maximus.ru",
            "www.beeline.ru",
            "www.mts.ru",
            "www.megafon.ru",
            "www.domotekhnika.ru/",
            "tomatomobile.ru",
            "www.cifrocity.com",
            "Cash.paylate.ru",
            "www.eldorado.ru",
            "www.egrad.ru",
            "msphone.ru",
            "cash.paylate.ru",
            "www.plus7.ru/",
        ]

        result = pd.DataFrame()
        result[f"{x.name}"] = x
        result["is_site"] = x.apply(lambda i: 1 if i in sites else 0)

        return result

    def get_feature_names_out(self, *args, **params):
        """TODO."""
        return self.columns


# @function_transformer_available
# def foo_fullname(x):
#     # TODO: x = x.fillna('пропуск')
#     x = str(x).strip().lower()

#     x = re.sub("\W+", " ", x)  # убирает спец символы
#     x = re.sub("\d+", "", x)  # убирает цифры

#     x_splitted = [word for word in x.split() if word not in stopwords]
#     for word in x_splitted:
#         if word in ["су", "пик", "констракшн", "прайдекс", "эталон"]:
#             x_splitted = ["стройка"]
#             break
#         elif word in ["школа"]:
#             x_splitted = ["школа"]
#             break
#         elif word in ["университет"]:
#             x_splitted = ["университет"]
#             break
#         elif word in ["спар"]:
#             x_splitted = ["супермаркет"]
#             break
#         elif word in ["чайхона", "чайхана", "урюк"]:
#             x_splitted = ["чайхана"]
#             break
#     x = " ".join(x_splitted)

#     if x.startswith("ооо"):
#         x = x.replace("ооо", "")
#     elif x.startswith("ао"):
#         x = x.replace("ао", "")
#     elif x.startswith("зао"):
#         x = x.replace("зао", "")
#     elif x.startswith("оао"):
#         x = x.replace("оао", "")
#     elif x.startswith("пао"):
#         x = x.replace("пао", "")
#     elif x.startswith("ип"):
#         x = "ип"

#     if "жилищник" in x or "благоустройство" in x:
#         x = "жилищник"
#     elif (
#         "строй" in x
#         or "фодд" in x
#         or "крост д" in x
#         or "ант япы" in x
#         or "ибт" in x
#         or "анттек" in x
#     ):
#         x = "стройка"
#     elif (
#         "дикси" in x
#         or "вкусвилл" in x
#         or "ашан" in x
#         or "пятерочка" in x
#         or "магнит" in x
#         or "билла" in x
#         or "перекресток" in x
#         or "лента" in x
#         or "евроспар" in x
#     ):
#         x = "супермаркет"
#     elif "такси" in x or "taxi" in x or "сити мобил" in x:
#         x = "такси"
#     elif (
#         "самокат" in x
#         or "деливери клаб" in x
#         or "яндекс еда" in x
#         or "яндекс доставка" in x
#         or "яндекс лавка" in x
#         or "локалкитчен" in x
#         or "деливери" in x
#         or "кухня районе" in x
#     ):
#         x = "доставка"
#     elif "яндекс" in x:
#         x = "яндекс"
#     elif "альфа м" in x:
#         x = "красное белое"
#     elif "петрович" in x:
#         x = "петрович"
#     elif "бургер кинг" in x:
#         x = "burger king"

#     x = x.strip()

#     if x in ["nan", " null", "none", ""]:
#         x = None

#     return x


@function_transformer_available
def foo_sigdoc(x):
    return 1 if x else 0


# class GenBert(BaseEstimator, TransformerMixin):
#     """TODO."""
#     def __init__(self):
#         self.columns = None

#     def fit(self, x: pd.Series, y=None):
#         """TODO."""
#         self.columns = np.array([f"position_{i}" for i in range(1, 313)])

#         return self

#     def transform(self, x: pd.Series):
#         """."""
#         result = pd.DataFrame(index=x.index)
#         result["position"] = x

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
#         model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").to(device)

#         np_arr = []
#         for word in result["position"].to_list():
#             t = tokenizer(word, return_tensors="pt", add_special_tokens=True)
#             with torch.no_grad():
#                 model_output = model(
#                     **{k: torch.tensor(v).to(device) for k, v in t.items()}
#                 )
#             embeddings = model_output.last_hidden_state[:, 0, :]
#             embeddings = torch.nn.functional.normalize(embeddings)
#             np_arr.append(embeddings[0].cpu().numpy())
#         return pd.DataFrame(np_arr, index=x.index)

#     def get_feature_names_out(self, *args, **params):
#         """TODO."""
#         return self.columns


# @function_transformer_available
# def foo_position(x):
#     x = str(x).strip().lower()

#     x = re.sub("\W+", " ", x)
#     x = re.sub("\d+", "", x)

#     if 'рабочий' in x or 'робоч' in x:
#         x = 'рабочий'
#     elif 'такси' in x:
#         x = 'таксист'
#     elif 'кассир' in x:
#         x = 'кассир'
#     elif 'строитель' in x:
#         x = 'строитель'
#     elif 'курьер' in x or 'доставщик' in x:
#         x = 'курьер'
#     elif 'маляр' in x or 'молер' in x or 'моляр' in x:
#         x = 'маляр'
#     elif 'продавец' in x:
#         x = 'продавец'
#     elif 'водитель' in x  or 'водит' in x:
#         x = 'водитель'
#     elif 'повар' in x or 'пекарь' in x:
#         x = 'повар'
#     elif 'уборщик' in x or 'уборщица' in x or 'уборк' in x:
#         x = 'уборщик'
#     elif 'грузчик' in x or 'груз' in x:
#         x = 'грузчик'
#     elif 'монтажник' in x:
#         x = 'монтажник'
#     elif 'администратор' in x:
#         x = 'администратор'
#     elif 'менеджер' in x:
#         x = 'менеджер'
#     elif 'бригадир' in x:
#         x = 'бригадир'
#     elif 'оператор' in x:
#         x = 'оператор'
#     elif 'мастер' in x:
#         x = 'мастер'
#     elif 'директ' in x:
#         x = 'директор'
#     elif 'слесарь' in x:
#         x = 'слесарь'
    
#     if x in ["none", "null", "nan", "", " "]:
#         x = "пропуск"

#     return x
