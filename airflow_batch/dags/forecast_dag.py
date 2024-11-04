from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow import DAG
from airflow.utils.dates import days_ago

from steps.model_train import feature_engineering_step, split_step
from steps.model_forecast import model_forecasting
from darts.models.forecasting.xgboost import XGBModel
import pandas as pd
import json

default_args = {'owner':'Guillermo Sainz',
                'schedule_interval':'@once',
                'start_date':days_ago(1),
                'description':'Pipeline mensual de predicción de ML'}

with DAG(dag_id='monthly_prediction_pipeline',
         default_args=default_args) as dag:
    
    def pred():
        data = pd.read_csv('data/clean/clean_data.csv')
        data = feature_engineering_step(data)
        dataset = split_step(data)
        
        with open('config.json','r') as config:
            config = json.load(config)

        model = XGBModel(lags=[-2,-5],
                    lags_future_covariates=[0],
                    lags_past_covariates=[-1,-2,-5]).load('models/xgb_model.pkl')
    
    
        _, y_test = dataset['y_timeseries']
        past_cov_train, past_cov_test = dataset['past_cov']
        _, fut_cov_test = dataset['future_cov']
        n_stepts = len(y_test)
        max_lag = min([l for l in config['MODEL_LAGS']])

        past_cov_test = [past_cov_train[idx][max_lag:].append(pc_test) for idx,pc_test in enumerate(past_cov_test)]

        preds = sum(model.predict(n=10,
                               series=y_test,
                               future_covariates=fut_cov_test,
                               past_covariates=past_cov_test)).pd_dataframe()
        print(preds)


    start_task = EmptyOperator(task_id='start_task')
    model_forecast_task = PythonOperator(task_id='model_forecast_task',
                                         python_callable=pred)
    
    end_task = EmptyOperator(task_id='end_task')

    start_task >> model_forecast_task >> end_task


