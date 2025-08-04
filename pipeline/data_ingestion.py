import os
import pandas as pd
import logging


# ensure the logging directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)



def load_data(file_path):    
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except KeyError as e:
        logger.error("Check the path to the file and ensure it exists.")
        raise 
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise e
    
def main():
    try:
        file_path = 'data/concrete_data.csv'
        data = load_data(file_path)
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise e
    

if __name__ == "__main__":
    main()


