from typing import Dict

import pandas as pd
from lightgbm import LGBMClassifier, Dataset
from lightgbm.callback import early_stopping, log_evaluation


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict,
    verbose: bool = False,
)->LGBMClassifier:
    """
    Trains a LightGBM classification model with early stopping and evaluation logging.

    Parameters:
        X_train: pandas DataFrame, training features.
        y_train: pandas Series, training target variable.
        X_test: pandas DataFrame, test features.
        y_test: pandas Series, test target variable.
        params: dict, LightGBM model parameters.
        verbose: bool, controls log verbosity (default: False).

    Returns:
        LGBMClassifier: the trained LightGBM model.
    """

    # Entrenar el modelo
    callbacks = [early_stopping(stopping_rounds=100), log_evaluation()]
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)

    return model
