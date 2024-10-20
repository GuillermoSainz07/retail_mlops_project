from .clean_data import dataset
from darts.metrics.metrics import (coefficient_of_variation,
                                   mse,
                                   rmse,
                                   mape)
from darts.models.forecasting.xgboost import XGBModel
from src.model_evaluation import ModelEvaluation

from ..config import MODEL_LAGS

def evaluation_metrics() -> None:
    model = XGBModel().load('models/xgb_model.pkl')
    y_train, y_test = dataset['y_timeseries']
    past_cov_train, past_cov_test = dataset['past_cov']
    fut_cov_train, fut_cov_test = dataset['future_cov']
    n_stepts = len(y_test)
    max_lag = min(*[l for l in MODEL_LAGS])

    past_cov_ts = [past_cov_train[idx].append(pc_test) for idx, pc_test in past_cov_test]
    fut_cov_ts = [fut_cov_train[idx].append(fc_test) for idx, fc_test in fut_cov_test]

    past_cov_test = [past_cov_train[idx][max_lag:].append(pc_test) for idx,pc_test in past_cov_test]

    y_all = [y_train[idx].append(yt) for idx,yt in y_test]

    evaluation_instance = ModelEvaluation(model,
                                          y_train,
                                          y_test,
                                          past_cov_test,
                                          fut_cov_test,
                                          y_all)
    
    evaluation_instance.write_metrics()
    evaluation_instance.make_backtest_plot(pat_cov_ts=past_cov_ts,
                                           fut_cov_ts=fut_cov_ts)
    evaluation_instance.make_error_distribution_plot()

if __name__ == '__main__':
    evaluation_metrics()

