import pandas as pd
from darts.timeseries import TimeSeries
from steps.clean_data import clean_data_step
from steps.model_train import feature_engineering_step
from darts.models.forecasting.xgboost import XGBModel


def model_forecasting(features:pd.DataFrame,
                      sales:pd.DataFrame,
                      stores:pd.DataFrame,
                      n:int):
    
    model = XGBModel(lags=[-2,-5],
                 lags_future_covariates=[0],
                 lags_past_covariates=[-1,-2,-5]).load('models/xgb_model.pkl')
    
    data = clean_data_step(features,
                           sales,
                           stores)
    
    data = feature_engineering_step(data)

    y_ts = TimeSeries.from_group_dataframe(data,
                                       time_col='Date',
                                       value_cols=['Weekly_Sales'],
                                       group_cols=['Store','Type'])

    future_cov_ts = TimeSeries.from_group_dataframe(data,
                                                    time_col='Date',
                                                    value_cols=['month','day','year','IsHoliday'],
                                                    group_cols=['Store','Type'])

    past_cov_ts = TimeSeries.from_group_dataframe(data,
                                                  time_col='Date',
                                                  value_cols=['ma1_sales','ma2_sales','ma5_sales','std_sales',
                                                                    'Temperature','Fuel_Price','CPI','Unemployment'],
                                                  group_cols=['Store','Type'])
    
    prediction = model.predict(n=n,
                               series=y_ts,
                               future_covariates=future_cov_ts,
                               past_covariates=past_cov_ts)
    