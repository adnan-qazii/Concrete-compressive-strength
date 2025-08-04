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
    

def preprocess_data(df):
    """
    Preprocess the DataFrame by handling missing values, duplicates, 
    outliers, and feature engineering for concrete strength prediction.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    logger.info("Data preprocessing started.")
    
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # 1. Data Quality Check
    logger.info(f"Initial dataset shape: {processed_df.shape}")
    logger.info(f"Initial null values: {processed_df.isnull().sum().sum()}")
    logger.info(f"Initial duplicate rows: {processed_df.duplicated().sum()}")
    
    # 2. Handle Missing Values
    if processed_df.isnull().any().any():
        logger.warning("Missing values detected. Handling missing data...")
        
        # For numerical columns, use median imputation
        numerical_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if processed_df[col].isnull().any():
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_val}")
    else:
        logger.info("No missing values found.")
    
    # 3. Remove Duplicate Rows
    initial_rows = len(processed_df)
    processed_df = processed_df.drop_duplicates()
    duplicates_removed = initial_rows - len(processed_df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
    
    # 4. Handle Outliers using IQR method
    logger.info("Detecting and handling outliers...")
    numerical_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    outliers_count = 0
    
    for col in numerical_cols:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before handling
        outliers_before = len(processed_df[(processed_df[col] < lower_bound) | 
                                         (processed_df[col] > upper_bound)])
        if outliers_before > 0:
            outliers_count += outliers_before
            # Cap outliers instead of removing them to preserve data
            processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Capped {outliers_before} outliers in {col} column.")
    
    if outliers_count == 0:
        logger.info("No outliers detected.")
    
    # 5. Data Type Optimization
    logger.info("Optimizing data types...")
    for col in processed_df.columns:
        if processed_df[col].dtype == 'float64':
            # Check if values can be converted to int
            if processed_df[col].apply(lambda x: x.is_integer()).all():
                processed_df[col] = processed_df[col].astype('int32')
            else:
                processed_df[col] = processed_df[col].astype('float32')
        elif processed_df[col].dtype == 'int64':
            processed_df[col] = processed_df[col].astype('int32')
    
    # 6. Statistical Summary
    logger.info("Final preprocessing statistics:")
    logger.info(f"Final dataset shape: {processed_df.shape}")
    logger.info(f"Memory usage: {processed_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 7. Data Validation
    if processed_df.isnull().any().any():
        logger.error("Warning: Null values still present after preprocessing!")
    else:
        logger.info("Data preprocessing completed successfully - no null values remaining.")
    
    return processed_df


    
def main():
    try:
        file_path = 'data/raw/concrete_data.csv'
        
        # Load raw data
        logger.info("Starting data ingestion pipeline...")
        raw_data = load_data(file_path)
        
        # Preprocess data
        processed_data = preprocess_data(raw_data)
        
        # Save processed data (optional)
        output_path = 'data/1_pre_process/processed_concrete_data.csv'
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise e
    

if __name__ == "__main__":
    main()



