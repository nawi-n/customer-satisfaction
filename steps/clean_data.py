import logging
from typing import Tuple
import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from zenml import step
from zenml.steps import StepContext
from typing_extensions import Annotated
from pydantic import BaseModel, Field

class DataFrameInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    df: Annotated[pd.DataFrame, Field(...)]

@step
def clean_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Clean and preprocess the input data."""
    try:
        # Convert StepArtifact to DataFrame if needed
        if isinstance(df, StepContext):
            df = df.read(pd.DataFrame)
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        logging.info(f"Starting data cleaning with input shape: {df.shape}")
        
        # Preprocess data
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()
        
        if preprocessed_data is None or preprocessed_data.empty:
            raise RuntimeError("Preprocessing resulted in empty or None DataFrame")
        
        # Split data
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info(f"Split data - X_train shape: {x_train.shape}, X_test shape: {x_test.shape}")
        return x_train, x_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error in clean_data step: {str(e)}")
        raise e
