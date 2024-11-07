import unittest
import pandas as pd
import logging

from steps.ingest_data import ingest_data_step
from steps.model_train import feature_engineering_step, split_step
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

class DataTest(unittest.TestCase):
    def test_format_date(self):
         """
         Test to ensure that dates follow the following format
         """
         features, sales, _ = ingest_data_step(features_path='data/raw/features_data_set.csv',
                                               sales_path='data/raw/sales_data_set.csv',
                                               stores_path='data/raw/stores_data_set.csv')
         
         date_format = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'

         self.assertTrue(np.all(features.Date.str.match(date_format)))
         logging.info('Features Date Format is correct')

         self.assertTrue(np.all(sales.Date.str.match(date_format)))
         logging.info('Sales Date Format is correct')


    def test_split_data(self):
        """
        Test to ensure that there is no data leakage
        """
        data = pd.read_csv('data/clean/clean_data.csv')

        y_ts,past_cov,future_cov  = feature_engineering_step(data)
        dataset = split_step(y_ts=y_ts,
                              past_cov=past_cov,
                              future_cov=future_cov)
        
        y_train, y_test = dataset['y_timeseries']
        futcov_train, futcov_test = dataset['future_cov']
        pastcov_train, pastcov_test = dataset['past_cov']
        
        self.assertEqual(len(y_train), len(y_test))
        self.assertEqual(len(futcov_train), len(futcov_test))
        self.assertEqual(len(pastcov_train), len(pastcov_test))

        for ytrain,ytest in zip(y_train,y_test):
             self.assertGreater(ytest[0].time_index ,ytrain[-1].time_index)
        logging.info('Y Series has not data leak')

        for fut_train,fut_test in zip(futcov_train,futcov_test):
             self.assertGreater(fut_test[0].time_index ,fut_train[-1].time_index)
        logging.info('Future Covariates has not data leak')

        for past_train,past_test in zip(pastcov_train,pastcov_test):
             self.assertGreater(past_test[0].time_index ,past_train[-1].time_index)
        logging.info('Past Covariates has not data leak')
    
if __name__ == '__main__':
     unittest.main()