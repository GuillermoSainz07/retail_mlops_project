import json
from darts.timeseries import TimeSeries
from typing_extensions import Annotated
from typing import Dict
from darts.models.forecasting.xgboost import XGBModel
from darts.metrics.metrics import mse,rmse,mape,coefficient_of_variation


def make_report(metrics:dict) -> None:
    with open('report_metrics.json', 'w') as report:
        json.dump(metrics, report)

def write_metrics(model:XGBModel,
                  y_train:TimeSeries,
                  y_test:TimeSeries,
                  past_cov_test:TimeSeries,
                  fut_cov_test:TimeSeries) -> dict:
    """
    This function should return a dictionary with metric
    """
    predictions = model.predict(series=y_train,
                                past_covariables=past_cov_test,
                                future_covariables=fut_cov_test)
    
    total_sales_prediction = sum(predictions['Weekly_Sales'])
    total_sales_real = sum(y_test['Weekly_Sales'])

    mse_metric = mse(actual_series=total_sales_real,
                     pred_series=total_sales_prediction)
    rmse_metric = rmse(actual_series=total_sales_real,
                     pred_series=total_sales_prediction)
    mape_metric = mape(actual_series=total_sales_real,
                     pred_series=total_sales_prediction)
    cv_metric = coefficient_of_variation(actual_series=total_sales_real,
                     pred_series=total_sales_prediction)
    
    metrics = {'mse_metric':mse_metric,
               'rmse_metric':rmse_metric,
               'mape_metric':mape_metric,
               'cv_metric':cv_metric}
    
    return metrics

def make_plots() -> None:
    pass

def train_test_timeseries(y:TimeSeries,
                          future_cov:TimeSeries=None,
                          past_cov:TimeSeries=None,
                          train_size:int=0.80)-> Dict[str,Annotated[tuple,'train_y, test_y'],
                                                      str,Annotated[tuple,'train_future_cov, test_future_cov'],
                                                      str,Annotated[tuple,'train_past_cov, test_past_cov']]:
    """
    This function help us to split data into train, test, and validation set
    Args:
        y: Principal TimeSeries, value to predict
        future_cov: Future covariables that actually known in present
        past_cov: Covariables that we have to use their past values because we dont known their future value
    Return:
        Dicctionary with data
    """
    
    train_y, test_y = [],[]
    train_future_cov, test_future_cov = [],[]
    train_past_cov, test_past_cov = [],[]
    
    for y_single in y:
        train_y_single, test_y_single = y_single.split_before(train_size)
        train_y.append(train_y_single)
        test_y.append(test_y_single)
    for future_cov_single in future_cov:
        train_future_cov_single, test_future_cov_single = future_cov_single.split_before(train_size)
        train_future_cov.append(train_future_cov_single)
        test_future_cov.append(test_future_cov_single)
    for past_cov_single in past_cov:
        train_past_cov_single, test_past_cov_single = past_cov_single.split_before(train_size)
        train_past_cov.append(train_past_cov_single)
        test_past_cov.append(test_past_cov_single)

    return {'y_timeseries':(train_y, test_y),
            'future_cov':(train_future_cov,test_future_cov),
            'past_cov':(train_past_cov, test_past_cov)}