import mlflow 
from .clean_data import dataset
from darts.models.forecasting.xgboost import XGBModel
from src.model_dev import XGBForecaster

### La longitud del vector de covariables pasadas que deberias utilizar
### para predecir n puntos en el futuro esta dada por n + | min (lags) |
### donde lags es el vector de lags que se utilizan en el modelo, es decir
### n + la cantidad maxima de lags, de tal manera que si quieres predecir
### 1 punto en el futuro, y tu lag maximo es de 3, deberias pasar un vector
### de longitud 1 + 3 = 4


def model_training():
    y_train, y_test = dataset['y_timeseries']
    past_cov_train, past_cov_test = dataset['future_cov']
    fut_cov_train, fut_cov_test = dataset['past_cov']
    
    model = XGBForecaster()
    model.train(y_train,
                past_cov_train,
                fut_cov_train)

    return model

if __name__ == '__main__':
    model = model_training()



