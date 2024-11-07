from darts.timeseries import TimeSeries
from darts.models.forecasting.xgboost import XGBModel
from darts.metrics.metrics import mse,rmse,mape,coefficient_of_variation
import matplotlib.pyplot as plt
import numpy as np
from .utils import make_report
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

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
        '''
        Function to make and create metrics
        '''
        try:
            predictions = self.model.predict(
                                    n=len(self.y_test[0]),
                                    series=self.y_train,
                                    past_covariates=self.past_cov_test,
                                    future_covariates=self.fut_cov_test)
        
            self.total_sales_prediction = sum(predictions)
            total_sales_real = sum(self.y_test)

            mse_metric = mse(actual_series=total_sales_real,
                            pred_series=self.total_sales_prediction)
            rmse_metric = rmse(actual_series=total_sales_real,
                            pred_series=self.total_sales_prediction)
            mape_metric = mape(actual_series=total_sales_real,
                            pred_series=self.total_sales_prediction)
            cv_metric = coefficient_of_variation(actual_series=total_sales_real,
                            pred_series=self.total_sales_prediction)
            
            metrics = {'mse_metric':mse_metric,
                    'rmse_metric':rmse_metric,
                    'mape_metric':mape_metric,
                    'cv_metric':cv_metric}
            logging.info(f'Metrics Calculated: {metrics}')
            make_report(metrics)
            return metrics
        except Exception as e:
            logging.error(f'Error in calculating metrics: {e}')
            raise e
        
    def make_test_horizon_plot(self):
        
        sum(self.y_train).plot(label='training')
        sum(self.y_test).plot(label='testing')
        #sum(predictions_lgbm).plot(label='prediction lgbm')
        self.total_sales_prediction.plot(label='predictions')

        plt.title('Test Forecasting')
        plt.savefig("testing_plot.png")
        plt.close()


    def make_backtest_plot(self,
                           past_cov_ts:TimeSeries,
                           fut_cov_ts:TimeSeries) -> None:
        '''
        Funtion to create back-test plots
        '''

        self.historical_forecast = self.model.historical_forecasts(series=self.y_all,
                                              future_covariates=fut_cov_ts,
                                              past_covariates=past_cov_ts,
                                              start=0.5,
                                              forecast_horizon=10, 
                                              retrain=False,
                                              last_points_only=False)
        forecaste_plots = []

        for j in range(len(self.historical_forecast[0])):
            for i in range(len(self.historical_forecast)):
                if i == 0:
                    plot = self.historical_forecast[i][j]
                else:
                    plot += self.historical_forecast[i][j]

            forecaste_plots.append(plot)

        sum(self.y_all).plot('Real Values')
        for p in forecaste_plots:
            p.plot(default_formatting=False)
            legend = plt.legend()  # Crea la leyenda
            legend.set_visible(False)

        plt.title('Historical forecast / Backtest')
        plt.savefig("backtest_plot.png")
        plt.close()

    def make_error_distribution_plot(self) -> None:
        '''
        Function to create distribution error plot
        '''
        backtest_xgb = self.model.backtest(series=self.y_all,
                                   historical_forecasts=self.historical_forecast,
                                   metric=rmse,
                                   last_points_only=False,
                                   reduction=None)
        
        means_of_metrics = [np.mean(metric) for metric in backtest_xgb]
        plt.hist(means_of_metrics)
        plt.title('Rmse Distribution')
        plt.savefig("error_dist_plot.png")
        plt.close()