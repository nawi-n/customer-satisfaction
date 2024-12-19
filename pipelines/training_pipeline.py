import mlflow
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """Training pipeline for the customer satisfaction model"""
    
    # Initialize MLflow
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    mlflow.set_experiment("customer_satisfaction_experiment")
    
    with mlflow.start_run() as run:
        # Get the data
        df = ingest_data()
        
        # Clean the data
        x_train, x_test, y_train, y_test = clean_data(df)
        
        # Train the model
        model = train_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        
        # Evaluate the model
        mse, rmse = evaluation(model=model, x_test=x_test, y_test=y_test)
        
        return mse, rmse