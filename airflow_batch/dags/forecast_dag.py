from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow import DAG
from airflow.utils.dates import days_ago
import logging
import pandas as pd

import sys 
sys.path.append('/opt/airflow/src')
from data_wrangling import DataPreproStrategy, DataFeatureEngineering

from darts.models.forecasting.xgboost import XGBModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

default_args = {'owner':'Guillermo Sainz',
                'schedule_interval':'@once',
                'start_date':days_ago(1),
                'description':'Pipeline mensual de predicción de ML'}

with DAG(dag_id='monthly_prediction_pipeline',
         default_args=default_args) as dag:
    
    def pred():
        logging.info('Pulling data from AWS S3')
        model = XGBModel(lags=[-2,-5],
                 lags_future_covariates=[0],
                 lags_past_covariates=[-1,-2,-5]).load('/opt/airflow/models/xgb_model.pkl')
        logging.info('Model inference DONE')
        
    start_task = EmptyOperator(task_id='start_task')
    model_forecast_task = PythonOperator(task_id='model_forecast_task',
                                         python_callable=pred)
    
    end_task = EmptyOperator(task_id='end_task')

    start_task >> model_forecast_task >> end_task


