import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated

from zenml import step


@step
def clean_data(
    df: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        df: pd.DataFrame
    Returns:
        Tuple containing (x_train, x_test, y_train, y_test)
    """
    try:
        if df is None:
            raise ValueError("Input DataFrame cannot be None")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame but got {type(df)}")
            
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        logging.info(f"Cleaning data with shape: {df.shape}")
        
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info(f"Successfully cleaned and split data. Train shape: {x_train.shape}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in clean_data step: {str(e)}")
        raise e
