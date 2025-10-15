import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from src.data.cl_preprocess import (
    foo_sex,
    foo_sigdoc,
    foo_phone_prefix,
    foo_mail_domain,
    foo_nameto,
    foo_sitename,
    foo_fullname,
    foo_group_domain,
    foo_position,
    foo_dtype,
    GenMailError,
    GenIsSite,
    GenBert,
)

from utils.custom_column_transformers import GenAge, GenDataFromDatetime


RAW_DATA_PATH = os.path.join("../..", "data", "raw", "raw.csv")
NEW_RAW_PATH = os.path.join("../..", "data", "raw", "new_raw.csv")
PREPARED_DATA_PATH = os.path.join("..", "..", "data", "prepared", "prepared.csv")
NEW_PREPARED_PATH = os.path.join("..", "..", "data", "prepared", "new_prepared.csv")
MOB_30_2_DATA_PATH = os.path.join("..", "..", "data", "prepared", "30_2_mob.csv")

data = pd.read_csv(RAW_DATA_PATH, parse_dates=["dtstart", "birthday"])

data = data.drop(
    columns=[
        # targets
        "30_2mob",
        "30_3mob",
        # '60_6mob',
        "90_6mob",
        "90_12mob",
        "mob_60",

        # data leak
        # "sigdoc", # поле из будущего
        "verification",
        "EmployeeId",

        # поля-разделители
        "Сервис3",
        "risk",
        "Comment",
        "clientid",
        "ds_ver",

        # origin
        "citizenshipid", # полностью повторияет `nation`
        # "nation",  # полностью повторияет `citizenshipid`
        "spouseincome",  # константный признак
        "amountoriginal",  # полностью повторияет `Loan amount`
        "client_type",  # константный признак
        "educationid",  # константный признак
        "bankrupt",  # константный признак
        "monthlycreditpayment",  # константный признак
        # "AmountPurchaseOriginal", # повторяет Loan amount
        # "idchannel", # неинформативное
        # "CustomerReg", # нет информации о том, что это за поле

        # NBKI
        "Timestamp",  # Есть старые запросы до года, однако их очень мало. В основном запросы не старше суток
        "Recent_account_date",  # Всего 743 non-null
        "Oldest_account_date",  # Всего 743 non-null
        "Total_disputed",  # константный признак
        "Total_letigation",  # константный признак
        "Total_bankruptcy",  # константный признак
        "RCC_overdue_amount",  # константный признак
        "Fico_score",  # не можем получать
        # "IS_Fico", # перестали получать
        "Reg_region",  # неинформативный признак
        "Liv_region",  # неинформативный признак

        # Equifax
        "Timestamp_eq",  # Есть старые запросы до года, однако их очень мало. В основном запросы не старше суток
        "Recent_account_date_eq",  # Всего 2060 non-null
        "Oldest_account_date_eq",  # Всего 2060 non-null
        "Payment_cards",  # константный признак
        "Total_bankruptcy_eq",  # константный признак
        "Total_letigation_eq",  # константный признак
        "RCC_credit_limit_eq",  # константный признак
        "RCC_overdue_amount_eq",  # константный признакы
        # "current_120DPD_3K_flag",
        # "Nb_active_consumer_credit_eq",
        # "Nb_active_rcc_eq",
        # "Nb_active_mortgages_eq",
    ]
)

data = data.query('dtstart < "2023-09-01"')

data["60_6mob"] = data["60_6mob"].apply(
    lambda x: 0 if str(x) == "0.0" else 1
)  # Предобработка таргета

X = data.drop(
    columns=[
        "60_6mob",
        "interest_amount",
    ]
)

y = data["60_6mob"]


def preprocessing_foo(x: pd.DataFrame) -> pd.DataFrame:
    change_dtype_cols = [
        "pref_mobile",
        "sex",
        "domen",
        "mail_error",
        "nameto",
        "sitename",
        "is_site",
        "fullname",
        "citizenshipid",
        "CategoryGoodsId",
        "martialid",
        "typeemployerid",
        "idchannel",
        "CompanyTypeId",
        "OccupationAreaId",
        "EmploymentType",
        "ALLCLCValueId",
        "BLOCKCNTValueId",
        "BLOCKDURValueId",
        "INTEGRALSCOREValueId",
        "LIFETIMEBINValueId",
        "PAYMAXValueId",
        "CH_length",
    ]

    level1 = ColumnTransformer(
        [
            ("preprocess_sex", FunctionTransformer(foo_sex), "sex"),
            ("preprocess_sigdoc", FunctionTransformer(foo_sigdoc), "sigdoc"),
            ("preprocess_phone_prefix", FunctionTransformer(foo_phone_prefix), "pref_mobile"),
            ("preprocess_domen", FunctionTransformer(foo_mail_domain), "domen"),
            ("preprocess_nameto", FunctionTransformer(foo_nameto), "nameto"),
            ("gen_is_site", GenIsSite(), "sitename"),
            ("preprocess_sitename", FunctionTransformer(foo_sitename), "sitename"),
            ("preprocess_fullname", FunctionTransformer(foo_fullname), "fullname"),
            ("preprocess_position", FunctionTransformer(foo_position), "position"),
            ("gen_age", GenAge(), ["dtstart", "birthday"]),
            ("gen_datetime_data", GenDataFromDatetime(), "dtstart"),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    level2 = ColumnTransformer(
        [
            ("gen_mail_errors", GenMailError(), "domen"),
            ("group_domen", FunctionTransformer(foo_group_domain), "domen"),
            ("gen_Bert", GenBert(), "position"),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    level3 = ColumnTransformer(
        [
            ("preprocess_dtype", FunctionTransformer(foo_dtype), change_dtype_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    preprocess = Pipeline(
        [
            ("step1", level1),
            ("step2", level2),
            ("step3", level3),
        ]
    )

    x = preprocess.fit_transform(x)

    return x


prepared_data = preprocessing_foo(X)
prepared_data["interest_amount"] = data["interest_amount"]
prepared_data["dtstart"] = data["dtstart"]
prepared_data["60_6mob"] = y

prepared_data.to_csv(PREPARED_DATA_PATH, index=False)
