from darts.timeseries import TimeSeries
from darts.models.forecasting.xgboost import XGBModel
from darts.metrics.metrics import mse,rmse,mape,coefficient_of_variation
import matplotlib.pyplot as plt
import numpy as np
from .utils import make_report


class ModelEvaluation:
    def __init__(self,
                model:XGBModel,
                y_train:TimeSeries,
                y_test:TimeSeries,
                past_cov_test:TimeSeries,
                fut_cov_test:TimeSeries,
                y_all:TimeSeries):
        
        self.model = model
        self.y_train = y_train
        self.y_test = y_test
        self.past_cov_test = past_cov_test
        self.fut_cov_test = fut_cov_test
        self.y_all = y_all

    def write_metrics(self) -> None:
        predictions = self.model.predict(series=self.y_train,
                                past_covariables=self.past_cov_test,
                                future_covariables=self.fut_cov_test)
    
        total_sales_prediction = sum(predictions['Weekly_Sales'])
        total_sales_real = sum(self.y_test['Weekly_Sales'])

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
        
        
        make_report(metrics)

    def make_backtest_plot(self) -> None:
        self.historical_forecast = self.model(series=self.y_all,
                                              future_covariates=self.future_cov_ts,
                                              past_covariates=self.past_cov_ts,
                                              start=0.5,
                                              forecast_horizon=10, 
                                              retrain=True,
                                              last_points_only=False)
        
        sum(*[self.historical_forecast]).plot()
        plt.savefig("backtest_plot.png")
        plt.close()

    def make_error_distribution_plot(self) -> None:
        backtest_xgb = self.model.backtest(series=self.y_all,
                                   historical_forecasts=self.historical_forecast,
                                   metric=rmse,
                                   last_points_only=False,
                                   reduction=None)
        
        means_of_metrics = [np.mean(metric) for metric in backtest_xgb]
        plt.hist(means_of_metrics)
        plt.savefig("error_dist_plot.png")
        plt.close()