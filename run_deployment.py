from typing import cast
import click
from rich import print
import time
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

from typing import cast

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0,
    help="Minimum accuracy required to deploy the model",
)
def main(config: str, min_accuracy: float):
    """Run the MLflow example pipeline."""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        # Initialize a continuous deployment pipeline run
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )
        print(
            "Deployment pipeline completed. Waiting for model to be deployed..."
        )
        time.sleep(10)  # Give some time for the deployment to complete

    if predict:
        # Get the active deployment if any
        existing_services = mlflow_model_deployer_component.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model",
        )

        if not existing_services:
            print(
                "No MLflow prediction service is currently running. The deployment "
                "pipeline must run first to train and deploy a model. Execute "
                "the same command with the `--config deploy` argument to deploy a model."
            )
            return

        # Initialize an inference pipeline run
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        f"\nMLflow UI: [italic green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]"
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"MLflow prediction server is running at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service: "
                f"[italic green]zenml model-deployer models delete {str(service.uuid)}[/italic green]"
            )
        elif service.is_failed:
            print(
                f"MLflow prediction server failed:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )

if __name__ == "__main__":
    main()