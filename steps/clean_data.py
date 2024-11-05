from src.data_wrangling import (DataPreproStrategy, DataSplitStrategy,
                                DataFeatureEngineering)
from ingest_data import ingest_data_step
import pandas as pd
from typing import Dict

def clean_data_step(features:pd.DataFrame,
                    sales:pd.DataFrame,
                    stores:pd.DataFrame) -> tuple:
    """
    This function is used to clean, preprocessing and split the data.
    Returns:
        train, validation, test datasets
    """
    # Data Preprocessing
    preprocessor = DataPreproStrategy()
    data = preprocessor.handle_data(features, sales, stores)
    return data

if __name__ == "__main__":
    features, sales, stores = ingest_data_step(features_path='data/raw/features_data_set.csv',
                                               sales_path='data/raw/sales_data_set.csv',
                                               stores_path='data/raw/stores_data_set.csv')
    data = clean_data_step(features,
                           sales,
                           stores)


 


    