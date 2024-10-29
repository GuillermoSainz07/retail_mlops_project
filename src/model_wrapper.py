import mlflow
from darts.timeseries import TimeSeries


class DartsModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context,
                series:TimeSeries,
                past_covariates:TimeSeries,
                future_covariates:TimeSeries,
                n:int):
        # Asegúrate de que model_input esté en el formato que Darts requiere
        return self.model.predict(n=n,
                                  series=series,
                                  past_covariates=past_covariates,
                                  future_covariates=future_covariates)