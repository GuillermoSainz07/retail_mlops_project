from model_train import feature_engineering_step, split_step
from darts.models.forecasting.xgboost import XGBModel
from src.model_evaluation import ModelEvaluation
import pandas as pd
import json
import mlflow
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def evaluation_metrics() -> None:
    '''
    Function to make and track evaluation metrics
    '''
    data = pd.read_csv('data/clean/clean_data.csv')
    data = feature_engineering_step(data)
    dataset = split_step(data)

    with open('config.json','r') as config:
        config = json.load(config)

    model = XGBModel(lags=[-2,-5],
                 lags_future_covariates=[0],
                 lags_past_covariates=[-1,-2,-5]).load('models/xgb_model.pkl')
    
    y_train, y_test = dataset['y_timeseries']
    past_cov_train, past_cov_test = dataset['past_cov']
    fut_cov_train, fut_cov_test = dataset['future_cov']
    n_stepts = len(y_test)
    max_lag = min([l for l in config['MODEL_LAGS']])

    past_cov_ts = [past_cov_train[idx].append(pc_test) for idx, pc_test in enumerate(past_cov_test)]
    fut_cov_ts = [fut_cov_train[idx].append(fc_test) for idx, fc_test in enumerate(fut_cov_test)]

    past_cov_test = [past_cov_train[idx][max_lag:].append(pc_test) for idx,pc_test in enumerate(past_cov_test)]

    y_all = [y_train[idx].append(yt) for idx,yt in enumerate(y_test)]

    evaluation_instance = ModelEvaluation(model,
                                          y_train,
                                          y_test,
                                          past_cov_test,
                                          fut_cov_test,
                                          y_all)
    
    metrics = evaluation_instance.write_metrics()

    evaluation_instance.make_backtest_plot(past_cov_ts=past_cov_ts,
                                           fut_cov_ts=fut_cov_ts)
    evaluation_instance.make_error_distribution_plot()

    with open('config.json', 'r') as config:
        config_dict = json.load(config)
    try:
        import dagshub
        dagshub.init(repo_owner='GuillermoSainz07', repo_name='retail_mlops_project', mlflow=True)
        
        with mlflow.start_run(run_id=config_dict["RUN_ID"]):
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
                
        logging.info('Log Metrics Done')
        
    except Exception as e:
        logging.error(f'Error logging metrics: {e}')

if __name__ == '__main__':
    evaluation_metrics()

