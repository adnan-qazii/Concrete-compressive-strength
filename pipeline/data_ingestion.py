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

# This code is part of the data ingestion module for the concrete compressive strength prediction project.
# It includes logging functionality to track the data loading process and handle errors.
# The load_data function reads a CSV file and logs the success or failure of the operation.
# The main function serves as an entry point for testing the data loading functionality.
# Ensure that the logging directory exists and is created if it doesn't.
# The log file will be created in the 'logs' directory with the name 'data_ingestion.log'.
# The logging level is set to DEBUG to capture detailed information about the data loading process.
# The code is designed to be run as a standalone script, and it will log any errors encountered during execution.
# The file path for the CSV data is set to 'data/concrete_data.csv', which should be adjusted based on your project structure.
# The code is structured to be modular, allowing for easy integration into a larger data pipeline.
# The logging messages will help in debugging and understanding the flow of data ingestion.
# The load_data function can be reused in other parts of the project where data loading is required.
# The code is designed to be robust, with error handling to catch and log exceptions that may occur during data loading.
