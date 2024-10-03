import pandas as pd
from darts.utils.model_selection import train_test_split
from darts.timeseries import TimeSeries

from abc import ABC, abstractmethod
import logging
from typing import Union, Tuple
from typing_extensions import Annotated


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

class DataWrangling(ABC):
    """
    Abstract Class for full process data wrangling
    """

    @abstractmethod
    def handle_data(self,
                    data:pd.DataFrame) -> pd.DataFrame:
        pass


class DataPreproStrategy(DataWrangling):
    """
    Class for data pre-processing strategy
    """
    def handle_data(self, data: pd.DataFrame) -> TimeSeries:
        """
        This handle function we allows encode the 
        categorical variables
        """
        # Convert the data to TimeSeries
        try:
            ts = TimeSeries.from_dataframe(data)
            logging.info("Dataframe to darts time series transformation successful")
            return ts
        except Exception as e:
            logging.error(f"Error transforming dataframe to darts time series: {e}")
            raise e

class DataTransformation(DataWrangling):
    """
    Class for data transformation strategy
    """
    def handle_data(self, ts: TimeSeries) -> TimeSeries:
        pass

class DataSplitStrategy(DataWrangling):
    """
    Class for data splitting strategy
    """
    def handle_data(self, ts: TimeSeries) -> Tuple[Annotated[TimeSeries,'Training_Dataset'],
                                                   Annotated[TimeSeries,'Validation_Dataset'],
                                                   Annotated[TimeSeries,'Testing_Dataset']]:
        """
        Method to split data into train, val, test data
        """
        try:
            train, test = train_test_split(ts)
            train, val = train_test_split(train)
            logging.info("Data splitting successful")
            return train,val,test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise e
