import pandas as pd
from darts.timeseries import TimeSeries

from abc import ABC, abstractmethod
import logging
from typing import Union, Tuple, Dict
from typing_extensions import Annotated

from .utils import train_test_timeseries


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
    def handle_data(self, features: pd.DataFrame,
                          sales: pd.DataFrame,
                          stores: pd.DataFrame) -> pd.DataFrame:
        """
        This handle function we allows preprocces we own data
        """
        # Convert the data to TimeSeries
        try:
            sales['Date'] = pd.to_datetime(sales.Date, dayfirst=True)
            sales = sales.sort_values(by=['Date'], ascending=True)

            features['Date'] = pd.to_datetime(features.Date, dayfirst=True)
            features = features.sort_values(by=['Date'], ascending=True)

            sales['Date'] = pd.to_datetime(sales.Date, dayfirst=True)
            sales = sales.sort_values(by=['Date'], ascending=True)

            type_store = {row.Store:row.Type for _,row in stores.iterrows()}

            sales_store_grouped = (sales[['Store','Weekly_Sales','Date']].groupby(['Store','Date']).agg({'Weekly_Sales':'sum'})
                       .reset_index())
            
            new_dataset = pd.merge(sales_store_grouped, features, on=['Store','Date'], how='inner')

            new_dataset = new_dataset.drop(columns=['MarkDown1','MarkDown2',
                                        'MarkDown3','MarkDown4','MarkDown5'])

            new_dataset['IsHoliday'] = new_dataset['IsHoliday']*1

            new_dataset['Type'] = new_dataset.Store.map(type_store)

            new_dataset['Type'] = new_dataset.Type.map({'A':0,
                                                        'B':1,
                                                        'C':2})
            
            new_dataset.to_csv('data/clean/clean_data.csv')

            logging.info("Preproccessing done")

            return new_dataset
        
        except Exception as e:
            logging.error(f"Error in preproccesing: {e}")
            raise e

class DataFeatureEngineering(DataWrangling):
    """
    Class for data transformation strategy
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['Date'] = pd.to_datetime(data.Date)
            data['ma1_sales'] = data.Weekly_Sales.rolling(window=2).mean() 
            data['ma2_sales'] = data.Weekly_Sales.rolling(window=3).mean()
            data['ma5_sales'] = data.Weekly_Sales.rolling(window=5).mean()
            data['std_sales'] = data.Weekly_Sales.rolling(window=2).std()

            data['month'] = data.Date.dt.month
            data['day'] = data.Date.dt.day
            data['year'] = data.Date.dt.year

            data = data.dropna()
            logging.info('Feature Engineering Done')

            return data
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise e
        
class DataSplitStrategy(DataWrangling):
    """
    Class for data splitting strategy
    """
    def handle_data(self, data: pd.DataFrame) -> Dict[str,tuple[TimeSeries]]:
        """
        Method to split data into train, val, test data
        Args:
            Data: Dataset to train the model
        Return:
            Dictionary with all splited dataset (taget and features)
        """
        try:
            y_ts = TimeSeries.from_group_dataframe(data,
                                       time_col='Date',
                                       value_cols=['Weekly_Sales'],
                                       group_cols=['Store','Type'])

            future_cov_ts = TimeSeries.from_group_dataframe(data,
                                                            time_col='Date',
                                                            value_cols=['month','day','year','IsHoliday'],
                                                            group_cols=['Store','Type'])

            past_cov_ts = TimeSeries.from_group_dataframe(data,
                                                        time_col='Date',
                                                        value_cols=['ma1_sales','ma2_sales','ma5_sales','std_sales',
                                                                    'Temperature','Fuel_Price','CPI','Unemployment'],
                                                        group_cols=['Store','Type']) 
            
            logging.info("DataFrame transformed into timeseries object")

            dataset_timeseries = train_test_timeseries(y_ts,
                                                       future_cov_ts,
                                                       past_cov_ts)
            logging.info('Data Split Done')

            return dataset_timeseries
        
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise e
