import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def ingest_data_step(features_path:str,
                     sales_path:str,
                     stores_path:str) ->  tuple:
    try:
        features = pd.read_csv(features_path)
        sales = pd.read_csv(sales_path)
        stores = pd.read_csv(stores_path)

        logging.info(f"Ingesting data from {features_path}, {sales_path}, {stores_path}")
        return features, sales, stores
    except Exception as e:
        logging.error(f"Error ingesting data from {features_path}, {sales_path}, {stores_path}: {str(e)}")
        raise e
    
if __name__ == '__main__':
    features, sales, stores = ingest_data_step(features_path='../data/raw/features_data_set.csv',
                                               sales_path='../data/raw/sales_data_set.csv',
                                               stores_path='../data/raw/stores_data_set.csv')
    