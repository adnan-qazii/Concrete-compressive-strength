import os
import pandas as pd
import numpy as np
import logging
import yaml
import datetime
import warnings
warnings.filterwarnings('ignore')


import xgboost as xgb
# Machine Learning libraries
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           explained_variance_score)

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

# Create handlers if not already exists
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(log_dir, 'model_training.log')
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
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, 'data_processing_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config['data_processing']
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Use default paths if config not found
        return {
            'output_dir': 'data/3_train_test_data',
            'target_column': 'concrete_compressive_strength'
        }


def load_train_test_data(data_dir):
    """
    Load training and test data from the data processing pipeline.
    
    Args:
        data_dir (str): Directory containing train/test subdirectories
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading training and test data...")
    
    try:
        # Load training data
        train_dir = os.path.join(data_dir, 'train')
        X_train = pd.read_csv(os.path.join(train_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(train_dir, 'y_train.csv')).values.ravel()
        
        # Load test data
        test_dir = os.path.join(data_dir, 'test')
        X_test = pd.read_csv(os.path.join(test_dir, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv')).values.ravel()
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Training target shape: {y_train.shape}")
        logger.info(f"Test target shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a machine learning model.
    
    Args:
        model: ML model to train
        X_train, X_test, y_train, y_test: Training and test data
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    logger.info(f"Training {model_name}...")
    
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = {
            'Model': model_name,
            'Train_R2': r2_score(y_train, y_pred_train),
            'Test_R2': r2_score(y_test, y_pred_test),
            'Train_MSE': mean_squared_error(y_train, y_pred_train),
            'Test_MSE': mean_squared_error(y_test, y_pred_test),
            'Train_MAE': mean_absolute_error(y_train, y_pred_train),
            'Test_MAE': mean_absolute_error(y_test, y_pred_test),
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train_ExplainedVar': explained_variance_score(y_train, y_pred_train),
            'Test_ExplainedVar': explained_variance_score(y_test, y_pred_test)
        }
        
        logger.info(f"{model_name} trained successfully - Test R²: {metrics['Test_R2']:.4f}")
        return metrics, model
        
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}")
        return None, None


def train_models(X_train, X_test, y_train, y_test):
    """
    Train Random Forest, XGBoost, and Extra Trees models.
    
    Returns:
        tuple: (results_list, trained_models_dict)
    """
    logger.info("Starting model training pipeline...")
    
    # Define models with optimized parameters
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = []
    trained_models = {}
    
    # Train each model
    for model_name, model in models.items():
        metrics, trained_model = evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        
        if metrics is not None:
            results.append(metrics)
            trained_models[model_name] = trained_model
    
    logger.info(f"Model training completed. {len(results)} models trained successfully.")
    return results, trained_models


def save_results(results, trained_models, X_train, X_test, y_train, y_test):
    """
    Save comprehensive results to text files in results directory.
    
    Args:
        results (list): List of model evaluation metrics
        trained_models (dict): Dictionary of trained models
        X_train, X_test, y_train, y_test: Training and test data
    """
    logger.info("Saving results...")
    
    # Create main results directory in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    main_results_dir = os.path.join(project_root, 'results')
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Create training_results subdirectory
    training_results_dir = os.path.join(main_results_dir, 'training_results')
    os.makedirs(training_results_dir, exist_ok=True)
    
    # Find the next sample number
    existing_samples = [d for d in os.listdir(training_results_dir) 
                       if os.path.isdir(os.path.join(training_results_dir, d)) and d.startswith('sample_')]
    
    if existing_samples:
        sample_numbers = [int(d.split('_')[1]) for d in existing_samples if d.split('_')[1].isdigit()]
        next_sample = max(sample_numbers) + 1 if sample_numbers else 1
    else:
        next_sample = 1
    
    # Create sample-specific directory
    sample_dir = os.path.join(training_results_dir, f'sample_{next_sample}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Saving results to: {sample_dir}")
    
    # Convert results to DataFrame and sort by Test R²
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    # 1. Save Model Comparison Summary
    summary_file = os.path.join(sample_dir, f'model_comparison_summary_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CONCRETE STRENGTH PREDICTION - MODEL TRAINING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Number: {next_sample}\n")
        f.write(f"Models Trained: Random Forest, XGBoost, Extra Trees\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Features: {X_train.shape[1]}\n\n")
        
        f.write("MODEL PERFORMANCE RANKING (by Test R² Score):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<4} {'Model':<15} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<10}\n")
        f.write("-" * 60 + "\n")
        
        for idx, row in results_df.iterrows():
            f.write(f"{idx+1:<4} {row['Model']:<15} {row['Test_R2']:<10.4f} {row['Test_RMSE']:<12.2f} {row['Test_MAE']:<10.2f}\n")
        
        f.write(f"\nBEST PERFORMING MODEL: {results_df.iloc[0]['Model']}\n")
        f.write(f"Best Test R² Score: {results_df.iloc[0]['Test_R2']:.6f}\n")
        f.write(f"Best Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}\n")
        f.write(f"Best Test MAE: {results_df.iloc[0]['Test_MAE']:.4f}\n")
    
    # 2. Save Detailed Model Results
    detailed_file = os.path.join(sample_dir, f'detailed_model_results_{timestamp}.txt')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("CONCRETE STRENGTH PREDICTION - DETAILED MODEL METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"MODEL: {row['Model']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"R² Score (Training):     {row['Train_R2']:.6f}\n")
            f.write(f"R² Score (Test):         {row['Test_R2']:.6f}\n")
            f.write(f"MSE (Training):          {row['Train_MSE']:.6f}\n")
            f.write(f"MSE (Test):              {row['Test_MSE']:.6f}\n")
            f.write(f"MAE (Training):          {row['Train_MAE']:.6f}\n")
            f.write(f"MAE (Test):              {row['Test_MAE']:.6f}\n")
            f.write(f"RMSE (Training):         {row['Train_RMSE']:.6f}\n")
            f.write(f"RMSE (Test):             {row['Test_RMSE']:.6f}\n")
            f.write(f"Explained Variance (Training): {row['Train_ExplainedVar']:.6f}\n")
            f.write(f"Explained Variance (Test):     {row['Test_ExplainedVar']:.6f}\n")
            f.write("\n")
    
    # 3. Save Feature Importance Analysis
    importance_file = os.path.join(sample_dir, f'feature_importance_{timestamp}.txt')
    with open(importance_file, 'w', encoding='utf-8') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                f.write(f"{model_name.upper()} - FEATURE IMPORTANCE:\n")
                f.write("-" * 40 + "\n")
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                for _, row in feature_importance.iterrows():
                    f.write(f"{row['Feature']:<30}: {row['Importance']:.6f}\n")
                f.write("\n")
    
    # 4. Save CSV format for easy analysis
    csv_file = os.path.join(sample_dir, f'model_results_{timestamp}.csv')
    results_df.to_csv(csv_file, index=False)
    
    # Log file locations
    logger.info(f"Results saved successfully!")
    logger.info(f"Sample Directory: {sample_dir}")
    logger.info(f"Summary: {summary_file}")
    logger.info(f"Detailed: {detailed_file}")
    logger.info(f"Feature Importance: {importance_file}")
    logger.info(f"CSV: {csv_file}")
    
    return results_df


def main():
    """
    Main function to execute the model training pipeline.
    """
    try:
        logger.info("Starting Model Training Pipeline...")
        
        # Load configuration
        config = load_config()
        
        # Load training and test data
        X_train, X_test, y_train, y_test = load_train_test_data(config['output_dir'])
        
        # Train models
        results, trained_models = train_models(X_train, X_test, y_train, y_test)
        
        if not results:
            logger.error("No models were trained successfully!")
            return None
        
        # Save results
        results_df = save_results(results, trained_models, X_train, X_test, y_train, y_test)
        
        # Display summary
        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Best Model: {results_df.iloc[0]['Model']}")
        print(f"Test R² Score: {results_df.iloc[0]['Test_R2']:.4f}")
        print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}")
        print(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.2f}")
        print("="*70)
        
        logger.info("Model training pipeline completed successfully!")
        return results_df, trained_models
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise e


if __name__ == "__main__":
    main()