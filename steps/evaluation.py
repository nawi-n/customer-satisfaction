import logging
import mlflow
import numpy as np
import pandas as pd
from model.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing import Tuple
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float]:
    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        # Set up MLflow experiment
        mlflow.set_experiment("customer_satisfaction_experiment")
        
        with mlflow.start_run(nested=True) as run:
            prediction = model.predict(x_test)

            # Using the MSE class for mean squared error calculation
            mse_class = MSE()
            mse = mse_class.calculate_score(y_test, prediction)
            mlflow.log_metric("mse", mse)

            # Using the R2Score class for R2 score calculation
            r2_class = R2Score()
            r2_score = r2_class.calculate_score(y_test, prediction)
            mlflow.log_metric("r2_score", r2_score)

            # Using the RMSE class for root mean squared error calculation
            rmse_class = RMSE()
            rmse = rmse_class.calculate_score(y_test, prediction)
            mlflow.log_metric("rmse", rmse)
            
            return r2_score, rmse
    except Exception as e:
        logging.error(f"Error in evaluation step: {str(e)}")
        raise e
    finally:
        # Ensure any active MLflow run is ended
        mlflow.end_run()
