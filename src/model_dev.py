import mlflow
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass


class ExampleModel(Model):
    def __init__(self,
                 register_id:str=None,
                 model_name:str=None):
        """
        Construct Method
        Args:
            register_id: Register ID Models in MLFlow
        """
        self.register_id = register_id

    def train(self):
        with mlflow.start_run() as run:

            client = mlflow.MlflowClient()
            client.transition_model_version_stage(stage='production')


