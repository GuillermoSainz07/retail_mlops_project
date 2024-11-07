import pandas as pd
import numpy as np

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.window_transformer import WindowTransformer
from darts.utils.missing_values import fill_missing_values

from src.data_wrangling import DataPreproStrategy, DataFeatureEngineering
from darts.models.forecasting.xgboost import XGBModel
import json

from src.model_dev import XGBForecaster


def model_forecasting(input_data:pd.DataFrame,
                      features:pd.DataFrame,
                      sales:pd.DataFrame,
                      stores:pd.DataFrame):
    '''
    Forecast total sales for next week
    Args:
        - input_data: A dataframe with the following columns: [Stores, Date, IsHoliday].
                      Where Date column corresponds to the date to be predicted
        - features: A historical data about features
        - sales: A historical data about sales
        - stores: A historical data about stores
    '''
    with open('config.json','r') as config_file:
        config = json.load(config_file)
        max_lag = min([l for l in config['MODEL_LAGS']]) - 1
        
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
    engineering = DataFeatureEngineering()
    y_ts,past_cov_ts,future_cov_ts = engineering.handle_data(data_stack)

    past_cov_ts = [fill_missing_values(past_cov_ts[i]) for i in range(len(past_cov_ts))]
    
    y_for_prediction = [y_ts[i][:-1] for i in range(len(y_ts))]
    fut_for_prediction = [future_cov_ts[i][-1] for i in range(len(future_cov_ts))]
    past_for_prediction = [past_cov_ts[i][max_lag:] for i in range(len(past_cov_ts))]
    
    model = XGBForecaster(create_experiment=False).model_instance.load('models/xgb_model.pkl')
    
    total_sales_prediction = sum(model.predict(n=1,
                                series=y_for_prediction,
                                past_covariates=past_for_prediction,
                                future_covariates=fut_for_prediction)).pd_dataframe()
    
    return total_sales_prediction

if __name__ == '__main__':
    features = pd.read_csv('data/raw/features_data_set.csv')
    sales = pd.read_csv('data/raw/sales_data_set.csv')
    stores = pd.read_csv('data/raw/stores_data_set.csv')
    to_predict = pd.read_csv("C:/Users/PC/Downloads/to_s3_data/to_predict.csv")

    prediction = model_forecasting(input_data=to_predict,
                                   features=features,
                                   sales=sales,
                                   stores=stores)
    print(prediction.reset_index())
    prediction.reset_index().to_csv("C:/Users/PC/Downloads/to_s3_data/predictions.csv")