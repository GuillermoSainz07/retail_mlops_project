from src.data_wrangling import (DataPreproStrategy, DataSplitStrategy,
                                DataFeatureEngineering)
from .ingest_data import features, sales, stores

from typing import Dict
from typing_extensions import Annotated

def clean_data_step() -> Dict[str,tuple]:
    """
    This function is used to clean, preprocessing and split the data.
    Returns:
        train, validation, test datasets
    """
    # Data Preprocessing
    preprocessor = DataPreproStrategy()
    df = preprocessor.fit(features, sales, stores)

    # Feature Engineering
    feature_engineering = DataFeatureEngineering()
    df = feature_engineering.handle_data(df)

    # Split Data
    splitter = DataSplitStrategy()
    dataset = splitter.handle_data(df)

    return dataset

if __name__ == "__main__":
    dataset = clean_data_step()





    