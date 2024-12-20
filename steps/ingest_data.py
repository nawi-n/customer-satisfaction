import logging
import os
import pandas as pd
from zenml import step

@step
def ingest_data() -> pd.DataFrame:
    """Ingest data from CSV file."""
    try:
        data_path = "./data/olist_customers_dataset.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("The loaded DataFrame is empty")
            
        logging.info(f"Successfully ingested data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logging.error(f"Error in ingest_data step: {str(e)}")
        raise e
