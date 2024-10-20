import mlflow
from abc import ABC, abstractmethod

from darts.models.forecasting.baselines import NaiveDrift
from darts.models.forecasting.xgboost import XGBModel
from darts.timeseries import TimeSeries
class Model(ABC):

    def __init__(self,
                 model_name:str,
                 run_id:str,
                 name_experiment_intance:str):
        
        """
        Construct Method
        Args:
            model_name: Name to register model
            run_id: Runs ID of experiment
            name_experiment_intance: Models Name of experiment instance
        """
        self.model_name = model_name
        self.client = mlflow.MlflowClient()
        self.model_uri = f"runs:/{run_id}/{name_experiment_intance}"
        self.name_experiment_intance = name_experiment_intance

    @abstractmethod
    def track_model(self):
        pass

    @abstractmethod
    def train(self, X, y):
        pass


class BaseLineModel(Model):
    def __init__(self,
                 model_name:str,
                 run_id:str,
                 name_experiment_intance:str):
        super().__init__(model_name=model_name,
                         run_id=run_id,
                         name_experiment_intance=name_experiment_intance)
        
    def track_model(self):
        mlflow.register_model(self.model_uri)
        latest_mv = self.client.get_lastest_versions(self.model_name)[0]
        self.client.set_registered_model_alias(self.model_name, "staging", latest_mv.version)
        self.client.set_registered_model_tag(self.model_name,"Developer","Guillermo")
        self.client.set_model_version_tag(self.model_name, latest_mv.version,"validation_status")

        self.client.update_model_version(name=self.model_name,
                                         version=latest_mv.version,
                                         description="Description")
        
        self.client.update_registered_model(name=self.model_name,
                                            description="Description")
    def train(self,
              train,
              val,
              test):
        #mlflow.set_experiment('')

        with mlflow.start_run(run_name=f'Retrain {self.model_name}') as run:
            model = NaiveDrift()

            mlflow.darts.autolog()

            mlflow.darts.log_model(model, self.name_experiment_intance)
            model_uri = f"runs:/{run.info.run_id}/{self.name_experiment_intance}"

            model.register_model(model_uri, self.model_name)


class XGBForecaster(Model):
        
    def track_model(self):
        pass
    def train(y_train:TimeSeries,
              past_cov_train:TimeSeries,
              fut_cov_train:TimeSeries):
        
        model = XGBModel(lags=[-2,-5],
                 lags_future_covariates=[0],
                 lags_past_covariates=[-1,-2,-5])
        
        model.fit(series=y_train,
                  past_covariates=past_cov_train,
                  future_covariates=fut_cov_train)
        
        model.save("../models/xgb_model.pkl")

        return model

        
        


