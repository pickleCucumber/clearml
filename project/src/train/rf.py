from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.constants import RANDOM_STATE
from utils.custom_column_transformers import AsCategory, GenAge, GenDataFromDatetime


def get_model():
    int_cols = [
        "age",
        "day",
        "dependents",
        "averagemonthlyincome",
        "Days_since_last_credit",
        "Max_overdue",
        "Nb_delays_90plus_ever_eq",
        "CH_length_eq",
        "LIFETIMEBINValueId",
    ]

    cat_cols = [
        "sex",
        "weekday",
        "hour",
        "citizenshipid",
        "martialid",
        "sitename",
        "DOC",
        "MatchingLevel",
    ]

    float_cols = [
        "INTEGRALSCOREValueId",
        "Score",
        "requested_amount",
        "S_hare_active_credit",
    ]

    cat_pipe = Pipeline(
        [
            ("categorizer", AsCategory()),
            (
                "encode",
                OrdinalEncoder(
                    encoded_missing_value=-1,
                    handle_unknown="use_encoded_value",
                    unknown_value=-2,
                ),
            ),
        ]
    )

    int_pipe = Pipeline([("imputer", SimpleImputer())])

    pretransform_1 = ColumnTransformer(
        [
            ("gen_age", GenAge(), ["dtstart", "birthday"]),
            ("gen_datetime_data", GenDataFromDatetime(), "dtstart"),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    pretransform_2 = ColumnTransformer(
        [
            ("cat_feature", cat_pipe, cat_cols),
            ("int_features", int_pipe, int_cols),
            ("float_features", int_pipe, float_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    rf_feautre_step = Pipeline(
        [
            ("step_1", pretransform_1),
            ("step_2", pretransform_2),
            ("selection", SelectKBest(k=15)),
        ],
    ).set_output(transform="pandas")

    rf_estimator = Pipeline(
        [
            ("feature", rf_feautre_step),
            (
                "model",
                CalibratedClassifierCV(
                    RandomForestClassifier(random_state=RANDOM_STATE)
                ),
            ),
        ]
    )
    return rf_estimator
