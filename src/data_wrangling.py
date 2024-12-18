import pandas as pd
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.window_transformer import WindowTransformer

from abc import ABC, abstractmethod
import logging
from typing import  Dict
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
    def handle_data(self, features: pd.DataFrame,
                          sales: pd.DataFrame,
                          stores: pd.DataFrame,
                          save_data:bool=True) -> pd.DataFrame:
        """
        This handle function we allows preprocces we own data
        Args:
            features (pd.DataFrame): Features data
            sales (pd.DataFrame): Sales data
            stores (pd.DataFrame): Stores data
        Returns:
            DataFrame processed
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
            
            if save_data:
                new_dataset.to_csv('data/clean/clean_data.csv')
            else:
                pass

            logging.info("Preproccessing done")

            return new_dataset
        
        except Exception as e:
            logging.error(f"Error in preproccesing: {e}")
            raise e

class DataFeatureEngineering(DataWrangling):
    """
    Class for data transformation strategy
    """
    def handle_data(self, data: pd.DataFrame) -> tuple[Annotated[TimeSeries,'y_series'],
                                                       Annotated[TimeSeries,'past_covariates_series'],
                                                       Annotated[TimeSeries,'future_covariates_series']]:
        '''
        This handle function we allows to make a feature engineering 
        to train machine learning time series model
        Args:
            data: Data to make feature engineering
        Returns:
            post feature engineering data
        '''
        try:
            logging.info('Feature engineering ...')
            data['Date'] = pd.to_datetime(data.Date)
            data['month'] = data.Date.dt.month
            data['day'] = data.Date.dt.day
            data['year'] = data.Date.dt.year

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
                                                        value_cols=['Temperature','Fuel_Price','CPI','Unemployment',
                                                                    'Weekly_Sales'],
                                                        group_cols=['Store','Type']) 
            
            logging.info("DataFrame transformed into timeseries object")

        except Exception as e:
            logging.error(f"Error in transforming data: {e}")
            raise e
        
        try:
            logging.info('Add agregation metrics')
            window_transformer = WindowTransformer([{'function':'mean',
                                                    'mode':'rolling',
                                                    'components':'Weekly_Sales',
                                                    'window':2,
                                                    'closed':'left'},
                                                    {'function':'mean',
                                                    'mode':'rolling',
                                                    'components':'Weekly_Sales',
                                                    'window':5,
                                                    'closed':'left'},
                                                    {'function':'std',
                                                    'mode':'rolling',
                                                    'components':'Weekly_Sales',
                                                    'window':2,
                                                    'closed':'left'},
                                                    {'function':'std',
                                                    'mode':'rolling',
                                                    'components':'Weekly_Sales',
                                                    'window':5,
                                                    'closed':'left'}],
                                                    keep_non_transformed=True)
            
            past_cov_ts = window_transformer.transform(past_cov_ts)
            past_cov_ts = [past_cov_ts[i].drop_columns('Weekly_Sales') for i in range(len(past_cov_ts))]

            logging.info('Feature Engineering Done')     

            return y_ts, past_cov_ts, future_cov_ts
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise e
        
class DataSplitStrategy(DataWrangling):
    """
    Class for data splitting strategy
    """
    def handle_data(self, y_ts: TimeSeries,
                          past_cov_ts:TimeSeries,
                          future_cov_ts: TimeSeries) -> Dict[str,tuple[TimeSeries]]:
        """
        Method to split data into train, val, test data
        Args:
            Data: Dataset to train the model
        Return:
            Dictionary with all splited dataset (taget and features)
            with the following form:
            {'y_timeseries':(train_y, test_y),
            'future_cov':(train_future_cov,test_future_cov),
            'past_cov':(train_past_cov, test_past_cov)}
        """
        try:
            from .utils import train_test_timeseries
        
            dataset_timeseries = train_test_timeseries(y_ts,
                                                       future_cov_ts,
                                                       past_cov_ts)
            logging.info('Data Split Done')

            return dataset_timeseries
        
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise e
