import pandas as pd
import numpy as np

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.window_transformer import WindowTransformer
from darts.utils.missing_values import fill_missing_values

from src.data_wrangling import DataPreproStrategy
from darts.models.forecasting.xgboost import XGBModel
import json


def model_forecasting(input_data:pd.DataFrame,
                      features:pd.DataFrame,
                      sales:pd.DataFrame,
                      stores:pd.DataFrame):
    '''
    Forecast total sales for next week
    '''
    assert all(col in input_data.columns for col in ['Store', 'Date', 'IsHoliday']), 'Verifica las columnas de los datos'

    date_format = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'

    assert np.all(features.Date.str.match(date_format)), 'Proporciona el formato correto para las fechas'

    input_data['Date'] = pd.to_datetime(input_data['Date'], dayfirst=True)

    prepro = DataPreproStrategy()
    full_data = prepro.handle_data(features,
                                   sales,
                                   stores,
                                   save_data=False)
    
    data_stack = pd.concat([full_data,input_data])

    data_stack['Type'] = data_stack.Store.map({row.Store:row.Type for _,row in stores.iterrows()})
    data_stack['Type'] = data_stack.Type.map({'A':0,
                                                'B':1,
                                                'C':2})

    data_stack['year'] = data_stack.Date.dt.year
    data_stack['day'] =  data_stack.Date.dt.day
    data_stack['month'] =  data_stack.Date.dt.month


    window_transformer = WindowTransformer([{'function':'mean',
                                         'mode':'rolling',
                                         'components':'Weekly_Sales',
                                         'window':2,
                                         'closed':'left'},
                        
                                         {'function':'mean',
                                         'mode':'rolling',
                                         'components':'Weekly_Sales',
                                         'window':5,
                                         'closed':'left'},
                                         
                                         {'function':'std',
                                         'mode':'rolling',
                                         'components':'Weekly_Sales',
                                         'window':2,
                                         'closed':'left'},
                                         
                                         {'function':'std',
                                         'mode':'rolling',
                                         'components':'Weekly_Sales',
                                         'window':5,
                                         'closed':'left'}],
                                         keep_non_transformed=True)
        
    y_ts = TimeSeries.from_group_dataframe(data_stack,
                                            time_col='Date',
                                            value_cols=['Weekly_Sales'],
                                            group_cols=['Store','Type'])

    future_cov_ts = TimeSeries.from_group_dataframe(data_stack,
                                                    time_col='Date',
                                                    value_cols=['month','day','year','IsHoliday'],
                                                    group_cols=['Store','Type'])

    past_cov_ts = TimeSeries.from_group_dataframe(data_stack,
                                                    time_col='Date',
                                                    value_cols=['Temperature','Fuel_Price','CPI','Unemployment','Weekly_Sales'],
                                                    group_cols=['Store','Type']) 
    
    past_cov_ts = window_transformer.transform(past_cov_ts)
    past_cov_ts = [past_cov_ts[i].drop_columns('Weekly_Sales') for i in range(len(past_cov_ts))]


    past_cov_ts = [fill_missing_values(past_cov_ts[i]) for i in range(len(past_cov_ts))]

    with open('config.json','r') as config_file:
        config = json.load(config_file)
        max_lag = min([l for l in config['MODEL_LAGS']]) - 1
    
    y_for_prediction = [y_ts[i][:-1] for i in range(len(y_ts))]
    fut_for_prediction = [future_cov_ts[i][-1] for i in range(len(future_cov_ts))]
    past_for_prediction = [past_cov_ts[i][max_lag:] for i in range(len(past_cov_ts))]

    model = XGBModel(lags=[-2,-5],
                    lags_future_covariates=[0],
                    lags_past_covariates=[-1,-2,-5]).load('models/xgb_model.pkl')
    
    total_sales_prediction = sum(model.predict(n=1,
                                series=y_for_prediction,
                                past_covariates=past_for_prediction,
                                future_covariates=fut_for_prediction)).pd_dataframe()
    
    return total_sales_prediction

'''
if __name__ == '__main__':
    features = pd.read_csv('data/raw/features_data_set.csv')
    sales = pd.read_csv('data/raw/sales_data_set.csv')
    stores = pd.read_csv('data/raw/stores_data_set.csv')
    to_predict = pd.read_csv("C:/Users/PC/Downloads/to_s3_data/to_predict.csv")

    prediction = model_forecasting(input_data=to_predict,
                                   features=features,
                                   sales=sales,
                                   stores=stores)
    print(prediction)
'''