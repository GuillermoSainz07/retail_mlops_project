from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

from airflow.hooks.S3_hook import S3Hook

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.window_transformer import WindowTransformer
from darts.utils.missing_values import fill_missing_values
from darts.models.forecasting.xgboost import XGBModel

import logging
import pandas as pd
import numpy as np
import json

import io
import sys 

sys.path.append('/opt/airflow/src')
from data_wrangling import DataPreproStrategy, DataFeatureEngineering
from model_dev import XGBForecaster

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Configuración del DAG
default_args = {'owner':'Guillermo Sainz',
                'schedule_interval':'@once',
                'start_date':days_ago(1),
                'description':'Pipeline mensual de predicción de ML'}

def batch_prediction_task():

    with open('/opt/airflow/config.json','r') as config_file:
        config = json.load(config_file)
        max_lag = min([l for l in config['MODEL_LAGS']]) - 1

    s3 = S3Hook('aws_connection_s3')
    features_s3 = s3.get_key('features.csv', 'weekly-data-bucket').get()['Body'].read()
    sales_s3 = s3.get_key('sales.csv', 'weekly-data-bucket').get()['Body'].read()
    stores_s3 = s3.get_key('stores.csv', 'weekly-data-bucket').get()['Body'].read()
    to_predict_s3 = s3.get_key('to_predict.csv', 'weekly-data-bucket').get()['Body'].read()

    features = pd.read_csv(io.BytesIO(features_s3))
    sales = pd.read_csv(io.BytesIO(sales_s3))
    stores = pd.read_csv(io.BytesIO(stores_s3))
    to_predict = pd.read_csv(io.BytesIO(to_predict_s3))
    
    assert all(col in to_predict.columns for col in ['Store', 'Date', 'IsHoliday']), 'Verifica las columnas de los datos'

    date_format = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'

    assert np.all(features.Date.str.match(date_format)), 'Proporciona el formato correto para las fechas'

    to_predict['Date'] = pd.to_datetime(to_predict['Date'], dayfirst=True)

    prepro = DataPreproStrategy()
    full_data = prepro.handle_data(features,
                                   sales,
                                   stores,
                                   save_data=False)
    
    data_stack = pd.concat([full_data,to_predict])

    data_stack['Type'] = data_stack.Store.map({row.Store:row.Type for _,row in stores.iterrows()})
    data_stack['Type'] = data_stack.Type.map({'A':0,
                                              'B':1,
                                              'C':2})
    engineering = DataFeatureEngineering()
    y_ts,past_cov_ts,future_cov_ts = engineering.handle_data(data_stack)

    past_cov_ts = [fill_missing_values(past_cov_ts[i]) for i in range(len(past_cov_ts))]
    
    y_for_prediction = [y_ts[i][:-1] for i in range(len(y_ts))]
    fut_for_prediction = [future_cov_ts[i][-1] for i in range(len(future_cov_ts))]
    past_for_prediction = [past_cov_ts[i][max_lag:] for i in range(len(past_cov_ts))]

    logging.info('Loanding Model')    
    model = XGBForecaster(create_experiment=False).model_instance.load('/opt/airflow/models/xgb_model.pkl')
    logging.info('Model Loaded')
    
    logging.info('Making prediction  ...')
    total_sales_prediction = sum(model.predict(n=1,
                                series=y_for_prediction,
                                past_covariates=past_for_prediction,
                                future_covariates=fut_for_prediction)).pd_dataframe().reset_index()
    logging.info('Prediction Done')
    logging.info(total_sales_prediction)
    
    logging.info('Saving predictions')
    csv_buffer = io.StringIO()
    total_sales_prediction.to_csv(csv_buffer, index=False)
    s3.get_conn().put_object(Bucket='weekly-predictions-bucket', Key='predictions.csv', Body=csv_buffer.getvalue())
    
# =================DAG==============================
with DAG(dag_id='monthly_prediction_pipeline',
         default_args=default_args) as dag:
    
    start_task = EmptyOperator(task_id='start_task')

    check_features = S3KeySensor(
        task_id='Check-features-exist',
        bucket_name='weekly-data-bucket',
        bucket_key='features.csv',
        aws_conn_id='aws_connection_s3'
    )
    
    check_sales = S3KeySensor(
        task_id='Check-sales-exist',
        bucket_name='weekly-data-bucket',
        bucket_key='sales.csv',
        aws_conn_id='aws_connection_s3'
    )
    
    check_stores = S3KeySensor(
        task_id='Check-store-exist',
        bucket_name='weekly-data-bucket',
        bucket_key='stores.csv',
        aws_conn_id='aws_connection_s3'
    )
    
    check_pred = S3KeySensor(
        task_id='Check-preds-exist',
        bucket_name='weekly-data-bucket',
        bucket_key='to_predict.csv',
        aws_conn_id='aws_connection_s3'
    )

    batch_prediction = PythonOperator(
        task_id='batch_prediction',
        python_callable=batch_prediction_task
    )
    end_task = EmptyOperator(task_id='end_task')

    start_task >> [check_features, check_sales, check_stores, check_pred] >> batch_prediction >> end_task
