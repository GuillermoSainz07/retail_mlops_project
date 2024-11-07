from typing import Dict
from typing_extensions import Annotated
from darts.timeseries import TimeSeries
from src.model_dev import XGBForecaster
from src.data_wrangling import DataFeatureEngineering, DataSplitStrategy
import pandas as pd

### La longitud del vector de covariables pasadas que deberias utilizar
### para predecir n puntos en el futuro esta dada por n + | min (lags) |
### donde lags es el vector de lags que se utilizan en el modelo, es decir
### n + la cantidad maxima de lags, de tal manera que si quieres predecir
### 1 punto en el futuro, y tu lag maximo es de 3, deberias pasar un vector
### de longitud 1 + 3 = 4

def feature_engineering_step(df:pd.DataFrame)-> tuple[Annotated[TimeSeries,'y_series'],
                                                       Annotated[TimeSeries,'past_covariates_series'],
                                                       Annotated[TimeSeries,'future_covariates_series']]:
    fe_object = DataFeatureEngineering()
    y_ts, past_cov, future_cov = fe_object.handle_data(df)

    return  y_ts, past_cov, future_cov

def split_step(y_ts:TimeSeries,
               past_cov:TimeSeries,
               future_cov:TimeSeries) -> Dict:
    
    splitter = DataSplitStrategy()
    data = splitter.handle_data(y_ts=y_ts,
                                past_cov_ts=past_cov,
                                future_cov_ts=future_cov)

    return data

def model_training(dataset:Dict[str,tuple]) -> None:
    y_train, y_test = dataset['y_timeseries']
    past_cov_train, past_cov_test = dataset['past_cov']
    fut_cov_train, fut_cov_test = dataset['future_cov']

    model = XGBForecaster(model_name='XGB Retail',
                          name_experiment_intance='XGB Retail Experiment')
    model = model.train(y_train=y_train,
                    past_cov_train=past_cov_train,
                    fut_cov_train=fut_cov_train)

if __name__ == '__main__':
    data = pd.read_csv('data/clean/clean_data.csv')
    y_ts,past_cov,future_cov  = feature_engineering_step(data)
    dataset = split_step(y_ts=y_ts,
                         past_cov=past_cov,
                         future_cov=future_cov)
    model_training(dataset)



