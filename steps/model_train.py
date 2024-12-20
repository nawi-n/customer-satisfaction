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
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

experiment_tracker = Client().active_stack.experiment_tracker

# Set MLflow tracking URI
mlflow.set_tracking_uri(get_tracking_uri())

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
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
        mlflow.set_experiment("customer_satisfaction_experiment")
        
        with mlflow.start_run(nested=True) as run:
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

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("fine_tuning", fine_tuning)
            
            # Log the model to MLflow
            mlflow.sklearn.log_model(trained_model, "model")
            
            return trained_model
    except Exception as e:
        logging.error(f"Error in train_model step: {str(e)}")
        raise e
    finally:
        # Ensure any active MLflow run is ended
        mlflow.end_run()
