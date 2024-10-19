from src.data_wrangling import (DataPreproStrategy, DataSplitStrategy,
                                DataFeatureEngineering)
from .ingest_data import features, sales, stores

from typing import Dict
from typing_extensions import Annotated

def clean_data_step() -> Dict[str,Annotated[tuple,'train_y, test_y'],
                              str,Annotated[tuple,'train_future_cov, test_future_cov'],
                              str,Annotated[tuple,'train_past_cov, test_past_cov']]:
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





    