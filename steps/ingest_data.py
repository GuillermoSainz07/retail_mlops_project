import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def ingest_data_step(data_path:str) ->  pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Ingesting data from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error ingesting data from {data_path}: {str(e)}")
        raise e
    
if __name__ == '__main__':
    features, sales, stores = ingest_data_step()
    