from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

from airflow.hooks.S3_hook import S3Hook

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


def download_from_s3(key:str, bucket_name:str, local_path:str):
    hook = S3Hook('aws_connection_s3')
    file_name = hook.download_file(key=key,
                                   bucket_name=bucket_name,
                                   local_path=local_path)
    return file_name


with DAG(dag_id='monthly_prediction_pipeline',
         default_args=default_args) as dag:
    
    def pred():
        logging.info('Pulling data from AWS S3')
        model = XGBModel(lags=[-2,-5],
                 lags_future_covariates=[0],
                 lags_past_covariates=[-1,-2,-5]).load('/opt/airflow/models/xgb_model.pkl')
        logging.info('Model inference DONE')
        
    start_task = EmptyOperator(task_id='start_task')

    check_features = S3KeySensor(task_id='Check-features-exist',
                                bucket_name='weekly-data-bucket',
                                bucket_key='features.csv',
                                aws_conn_id='aws_connection_s3')
    
    check_sales = S3KeySensor(task_id='Check-sales-exist',
                                bucket_name='weekly-data-bucket',
                                bucket_key='sales.csv',
                                aws_conn_id='aws_connection_s3')
    
    check_stores = S3KeySensor(task_id='Check-store-exist',
                                bucket_name='weekly-data-bucket',
                                bucket_key='stores.csv',
                                aws_conn_id='aws_connection_s3')
    

    download_features = PythonOperator(task_id='Download_features',
                                       python_callable=download_from_s3,
                                       op_kwargs={'key':'features.csv',
                                                  'bucket_name':'weekly-data-bucket',
                                                  'local_path':'/opt/airflow'})
    
    download_sales = PythonOperator(task_id='Download_sales',
                                    python_callable=download_from_s3,
                                    op_kwargs={'key':'sales.csv',
                                                  'bucket_name':'weekly-data-bucket',
                                                  'local_path':'/opt/airflow'})
    
    download_stores = PythonOperator(task_id='Download_stores',
                                     python_callable=download_from_s3,
                                     op_kwargs={'key':'stores.csv',
                                                  'bucket_name':'weekly-data-bucket',
                                                  'local_path':'/opt/airflow'})


    model_forecast_task = PythonOperator(task_id='model_forecast_task',
                                         python_callable=pred)
    
    end_task = EmptyOperator(task_id='end_task')

    start_task >> (check_features, check_sales, check_stores) >> download_features >> download_sales >> download_stores >> model_forecast_task >> end_task
