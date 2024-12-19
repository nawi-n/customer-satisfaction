import logging
import mlflow
import pandas as pd
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "lightgbm",
    fine_tuning: bool = False,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        model_name: str
        fine_tuning: bool
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None

        if model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e
