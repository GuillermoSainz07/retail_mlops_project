import mlflow
from abc import ABC, abstractmethod
from typing import List, Dict
import logging

from darts.models.forecasting.xgboost import XGBModel
from darts.timeseries import TimeSeries
import json
from .model_wrapper import DartsModelWrapper


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

class Model(ABC):

    def __init__(self,
                 model_name:str,
                 name_experiment_intance:str):
        
        """
        Construct Method
        Args:
            model_name: Name to register model
            name_experiment_intance: Models Name of experiment instance
        """
        import dagshub

        dagshub.init(repo_owner='GuillermoSainz07', repo_name='retail_mlops_project', mlflow=True)

        experiment_name = "Retail Forecasting"
        try:
            mlflow.create_experiment(experiment_name, artifact_location="s3://ml-experiments-artifacts")
        except:
            pass

        mlflow.set_experiment(experiment_name)

        self.model_name = model_name
        self.client = mlflow.MlflowClient()
        self.name_experiment_intance = name_experiment_intance

    @abstractmethod
    def train(self, X, y):
        pass


class XGBForecaster(Model):
    def __init__(self,
                 model_name:str,
                 name_experiment_intance:str):
        
        super().__init__(model_name=model_name,
                         name_experiment_intance=name_experiment_intance)
         
    def train(self,
              y_train:List[TimeSeries],
              past_cov_train:List[TimeSeries],
              fut_cov_train:List[TimeSeries]):
        
        with mlflow.start_run(run_name=f'Train {self.model_name}') as run:
        
            model = XGBModel(lags=[-1,-2,-5],
                    lags_future_covariates=[0],
                    lags_past_covariates=[-1,-2,-5])
            try:
                model.fit(series=y_train,
                        past_covariates=past_cov_train,
                        future_covariates=fut_cov_train)
                
                logging.info('Model Trained')

                model.save("models/xgb_model.pkl")

            except Exception as e:
                logging.error(f"Error training model: {e}")
                raise e
            
            try:                
                mlflow.pyfunc.log_model(self.name_experiment_intance,
                                        python_model=DartsModelWrapper(model=model))
                

                run_id = run.info.run_id
                model_uri = f"runs:/{run.info.run_id}/{self.name_experiment_intance}"

                with open('config.json', 'r') as config:
                    config_dict = json.load(config)

                config_dict["RUN_ID"] = f"{run_id}"

                with open('config.json', 'w') as file:
                    json.dump(config_dict, file, indent=4)


                mlflow.register_model(model_uri, self.model_name)

                latest_mv = self.client.get_latest_versions(self.model_name)[0]
                self.client.set_registered_model_alias(self.model_name, "staging", latest_mv.version)
                self.client.set_registered_model_tag(self.model_name,"Developer","Guillermo")
                self.client.set_model_version_tag(self.model_name, latest_mv.version,"validation_status")

                self.client.update_model_version(name=self.model_name,
                                                version=latest_mv.version,
                                                description="Modelo Desarrollado con XGBoost")
                
                self.client.update_registered_model(name=self.model_name,
                                                    description="Modelo con Machine Learning")
                
                logging.info('Tracking experiment and model in mlflow')

            except Exception as e:
                logging.error(f"Error tracking model: {e}")
                pass

        return model

        
        


