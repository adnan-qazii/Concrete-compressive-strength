import os 
import pandas as pd
import numpy as np
import logging
import yaml
from sklearn.model_selection import train_test_split
from data_ingestion import load_data

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_processing')
logger.setLevel(logging.DEBUG)

# Create handlers if not already exists
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(log_dir, 'data_processing.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_config(config_path=None):
    """
    Load configuration from YAML file in main folder.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the parent directory (main project folder) from current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, 'data_processing_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config['data_processing']
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        logger.error("Please ensure 'data_processing_config.yaml' exists in the main project folder")
        raise FileNotFoundError(f"Configuration file '{config_path}' not found. Please create it in the main folder.")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise e


def load_normalized_data(data_path):
    """
    Load normalized data from feature selection step.
    
    Args:
        data_path (str): Path to normalized data file
        
    Returns:
        pd.DataFrame: Normalized dataset
    """
    try:
        logger.info(f"Loading normalized data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        return data
    except Exception as e:
        logger.error(f"Error loading normalized data: {e}")
        raise e


def split_data(data, config):
    """
    Split data into train and test sets.
    
    Args:
        data (pd.DataFrame): Input dataset
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing split datasets
    """
    logger.info("Starting data splitting...")
    
    # Extract configuration
    target_col = config['target_column']
    test_size = config['train_test_split']['test_size']
    random_state = config['train_test_split']['random_state']
    shuffle = config['train_test_split']['shuffle']
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    
    logger.info(f"Train set size: {len(X_train)} ({(1-test_size)*100:.1f}%)")
    logger.info(f"Test set size: {len(X_test)} ({test_size*100:.1f}%)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test
    }


def save_split_data(split_data, config):
    """
    Save split datasets to specified directories.
    
    Args:
        split_data (dict): Dictionary containing split datasets
        config (dict): Configuration parameters
    """
    logger.info("Saving split datasets...")
    
    output_dir = config['output_dir']
    save_format = config['save_format']
    
    # Create directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save training and test data
    if save_format == 'csv':
        split_data['X_train'].to_csv(os.path.join(train_dir, 'X_train.csv'), index=False)
        split_data['y_train'].to_csv(os.path.join(train_dir, 'y_train.csv'), index=False)
        logger.info(f"Training data saved to: {train_dir}")
        
        split_data['X_test'].to_csv(os.path.join(test_dir, 'X_test.csv'), index=False)
        split_data['y_test'].to_csv(os.path.join(test_dir, 'y_test.csv'), index=False)
        logger.info(f"Test data saved to: {test_dir}")
    
    elif save_format == 'pickle':
        import pickle
        
        # Save as pickle files
        with open(os.path.join(train_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump({'X_train': split_data['X_train'], 'y_train': split_data['y_train']}, f)
        
        with open(os.path.join(test_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump({'X_test': split_data['X_test'], 'y_test': split_data['y_test']}, f)
        
        logger.info(f"Data saved as pickle files in: {output_dir}")


def create_split_summary(split_data, config):
    """
    Create a summary report of the data splitting process.
    
    Args:
        split_data (dict): Dictionary containing split datasets
        config (dict): Configuration parameters
    """
    output_dir = config['output_dir']
    summary_path = os.path.join(output_dir, 'split_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("DATA SPLITTING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CONFIGURATION PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Test Size: {config['train_test_split']['test_size']}\n")
        f.write(f"Random State: {config['train_test_split']['random_state']}\n")
        f.write(f"Shuffle: {config['train_test_split']['shuffle']}\n\n")
        
        f.write("DATASET SIZES:\n")
        f.write("-" * 15 + "\n")
        
        total_samples = len(split_data['X_train']) + len(split_data['X_test'])
        
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Training Samples: {len(split_data['X_train'])} ({len(split_data['X_train'])/total_samples*100:.1f}%)\n")
        f.write(f"Test Samples: {len(split_data['X_test'])} ({len(split_data['X_test'])/total_samples*100:.1f}%)\n")
        
        f.write(f"\nFeatures: {split_data['X_train'].shape[1]}\n")
        f.write(f"Feature Names: {list(split_data['X_train'].columns)}\n")
        
        f.write(f"\nTarget Statistics:\n")
        f.write(f"Train - Mean: {split_data['y_train'].mean():.2f}, Std: {split_data['y_train'].std():.2f}\n")
        f.write(f"Test - Mean: {split_data['y_test'].mean():.2f}, Std: {split_data['y_test'].std():.2f}\n")
    
    logger.info(f"Split summary saved to: {summary_path}")


def main():
    """
    Main function to execute data processing pipeline.
    """
    try:
        logger.info("Starting data processing pipeline...")
        
        # Load configuration
        config = load_config()
        
        # Load normalized data from feature selection
        normalized_data = load_normalized_data(config['input_data_path'])
        
        # Split data into train/validation/test
        split_datasets = split_data(normalized_data, config)
        
        # Save split data
        save_split_data(split_datasets, config)
        
        # Create summary report
        create_split_summary(split_datasets, config)
        
        logger.info("Data processing pipeline completed successfully!")
        logger.info(f"Data saved to: {config['output_dir']}")
        
        return split_datasets
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}")
        raise e


if __name__ == "__main__":
    # Check if config file exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_path = os.path.join(parent_dir, 'data_processing_config.yaml')
    
    if not os.path.exists(config_path):
        print("ERROR: Configuration file 'data_processing_config.yaml' not found in main folder.")
        print(f"Expected path: {config_path}")
        print("Please ensure the YAML config file exists before running this script.")
        exit(1)
    
    main()
