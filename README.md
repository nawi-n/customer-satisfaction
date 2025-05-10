# Predicting Customer Satisfaction Before Purchase

**Problem Overview**: The task is to predict the review score a customer would give for a product they have not yet purchased, based on their previous order history. This prediction uses data from the Brazilian E-Commerce Public Dataset by Olist, which contains details of 100,000 orders from 2016 to 2018 across multiple Brazilian marketplaces. The dataset provides a wide range of information, such as order status, price, payment methods, shipping performance, customer locations, product attributes, and reviews. The goal is to forecast the satisfaction score for an upcoming purchase based on factors like order status, price, and payment information. To make this solution scalable and production-ready, we utilize ZenML to build a reliable machine learning pipeline that continuously predicts customer satisfaction.

The aim of this repository is to showcase how ZenML can streamline the process of building and deploying machine learning pipelines with ease. Key features include:

- Providing a solid framework and template for developing your own pipelines.
- Seamlessly integrating with deployment tools like MLflow for efficient tracking and deployment.
- Enabling quick and easy machine learning pipeline deployment.

## :snake: Required Python Packages

First, clone the project repository and install the necessary dependencies:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
pip install -r requirements.txt
```

ZenML 0.20.0 and later comes with a React-based dashboard to observe stack components and pipeline DAGs. To use this feature, install the optional dependencies for ZenML server:

```bash
pip install zenml["server"]
zenml up
```

For deploying models using MLflow, install the ZenML integration:

```bash
zenml integration install mlflow -y
```

A ZenML stack is required to have an MLflow experiment tracker and model deployer. To configure the stack, use the following commands:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## :thumbsup: Solution Overview

To predict customer satisfaction in a real-world context, we need more than just a one-time trained model. We are constructing a comprehensive pipeline that continuously predicts and deploys the machine learning model. This process also integrates a data application that consumes the model for business decisions.

The pipeline is scalable and designed for deployment in the cloud. It tracks the flow of data, features, model parameters, and predictions, ensuring end-to-end observability. ZenML makes building this workflow simple and efficient.

The solution also leverages ZenML’s integration with MLflow for tracking metrics and deploying models. Additionally, a Streamlit app is used to demonstrate how the model would function in a live environment.

### Training Pipeline

The training pipeline follows these steps:

- `ingest_data`: Collects data and organizes it into a DataFrame.
- `clean_data`: Cleans the data by removing irrelevant columns.
- `train_model`: Trains the model and logs it using MLflow autologging.
- `evaluation`: Assesses the model and logs evaluation metrics with MLflow autologging.

### Deployment Pipeline

The `deployment_pipeline.py` builds on the training pipeline and adds continuous deployment functionality. It processes input data, trains a model, and deploys it if the evaluation criteria are met. The additional steps include:

- `deployment_trigger`: Verifies if the newly trained model meets deployment criteria.
- `model_deployer`: Deploys the model using MLflow if the model meets accuracy thresholds.

ZenML’s integration with MLflow ensures that the pipeline tracks hyperparameters and model evaluations, while MLflow serves the model as a service once it meets the criteria. The deployed model is continuously updated when new models meet the required accuracy.

Finally, the Streamlit app consumes the deployed model and predicts customer satisfaction based on new inputs.

### Example Pipeline Execution

Run the following commands to execute the pipelines:

- Training pipeline:

```bash
python run_pipeline.py
```

- Continuous deployment pipeline:

```bash
python run_deployment.py
```

## :question: FAQ

**Q1**: I encounter an error `No Step found for the name mlflow_deployer` when running the continuous deployment pipeline.

**Solution**: This occurs because the artifact store has been overwritten. You need to delete the artifact store and rerun the pipeline. To find the artifact store location, use:

```bash
zenml artifact-store describe
```

To delete the artifact store, run:

```bash
rm -rf PATH
```

**Q2**: I see the error `No Environment component with name mlflow is currently registered.`

**Solution**: This indicates that the MLflow integration was not installed. Install the integration with:

```bash
zenml integration install mlflow -y
```
