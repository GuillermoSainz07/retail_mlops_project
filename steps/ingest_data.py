import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def ingest_data_step(features_path:str,
                     sales_path:str,
                     stores_path:str) ->  tuple:
    '''
    Funtion to load the data
    Args:
        feature_path: Path of the features
        sales_path: Path of the sales data
        stores_path: Path of the stores data
    Returns:
        A tuple with features, sales and store data
    '''
    try:
        features = pd.read_csv(features_path)
        sales = pd.read_csv(sales_path)
        stores = pd.read_csv(stores_path)

        logging.info(f"Ingesting data from {features_path}, {sales_path}, {stores_path}")
        return features, sales, stores
    except Exception as e:
        logging.error(f"Error ingesting data from {features_path}, {sales_path}, {stores_path}: {str(e)}")
        raise e
    