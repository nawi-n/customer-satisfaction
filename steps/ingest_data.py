import logging
import pandas as pd
from zenml import step
import os

class IngestData:
    def __init__(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame:
        try:
            data_path = "./data/olist_customers_dataset.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("The loaded DataFrame is empty")
                
            logging.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise e

@step
def ingest_data() -> pd.DataFrame:
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        if df is None:
            raise ValueError("IngestData.get_data() returned None")
        return df
    except Exception as e:
        logging.error(f"Error in ingest_data step: {str(e)}")
        raise e
