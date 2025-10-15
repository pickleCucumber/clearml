from typing import Dict, List, Optional, Tuple

import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer


def emb_size_rule(cat_num: int) -> int:
    # https://forums.fast.ai/t/size-of-embedding-for-categorical-variables/42608/2
    return min(600, round(1.6 * cat_num**0.56))


def make_emb_projections(
    data: pd.DataFrame,
    cat_features: List[str],
) -> Dict[str, Tuple[int, int]]:
    emb_projections = {}
    for cat_feature in cat_features:
        cat_num = data[cat_feature].nunique()
        emb_projections[cat_feature] = (cat_num, emb_size_rule(cat_num))

    return emb_projections


def make_rnn_model(
    cat_feature_names: List[str],
    embedding_projections: Dict[str, Tuple[int, int]],
    rnn_units: Optional[int] = 32,
    classifier_units: Optional[int] = 16,
    optimizer: Optional[Optimizer] = None,
):
    """
    Строит рекуррентную модель.

    Args:
        cat_feature_names:
        embedding_projections:
        rnn_units:
        classifier_units:
        optimizer:

    Returns:
        model: скомпилированная модель.
    """
    if not optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    inputs = []
    cat_embeds = []
    for feature_name in cat_feature_names:
        inp = L.Input(shape=(None,), dtype="uint32", name=f"input_{feature_name}")
        inputs.append(inp)
        source_size, projection = embedding_projections[feature_name]
        emb = L.Embedding(
            source_size + 1,
            projection,
            trainable=True,
            name=f"embedding_{feature_name}",
        )(inp)
        cat_embeds.append(emb)

    concated_cat_embeds = L.concatenate(cat_embeds)

    last_state = L.GRU(units=rnn_units)(concated_cat_embeds)

    dense_intermediate = L.Dense(classifier_units, activation="relu")(last_state)
    proba = L.Dense(1, activation="sigmoid")(dense_intermediate)

    model = Model(inputs=inputs, outputs=proba)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model
