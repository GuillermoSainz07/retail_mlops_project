# retail_mlops_project

### Complete MLOps project for retail sales forecasting.

* Complete data science process using darts, sklearn, pandas, notebooks, etc.
    * Compare statistical models vs machine learning models in order to select the best model tu predict total sales
* Creation with batch prediction approach with airflow.
    * Search historical and to predict data in bucket s3 (montly)
    * Pull historical and to predict data (monthly)
    * Create one step prediction (next month)
    * Push prediction to s3 bucket
* Experiment tracking and model registration with MLFlow.
    * Track the model at the time of training
    * Track model performance metrics
    * Versioning of the model
* Data versioning and pipeline creation with DVC
    * Initilize dvc repository in order to versioning data and pipeline outputs such that metrics, plots and models
* AWS tools (S3, EC2)
    * Use S3 to store data, models, artifacts and pipelines outputs
    * Use EC2 to deploy airflow pipeline
* Containerisation using Docker (Dockerfile, Docker Compose)
    * Use of dockerfile and docker compose in this project in order to make it reportable and easily deployable.
* Creation of CI/CD pipeline with github actions
    * Use of github actions in order to automate the deployment of the project in a cloud environment

You will be able to monitor the monthly forecast DAG from the web interface provided by airflow from the public IPv4 on the port configured for airflow in docker compose 8080, the instance will be closed to avoid generating costs.