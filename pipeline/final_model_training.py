import os
import pandas as pd
import numpy as np
import logging
import yaml
import datetime
import warnings
import json
import time
import joblib
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    explained_variance_score, max_error, median_absolute_error
)

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('final_model_training')
logger.setLevel(logging.DEBUG)

# Create handlers if not already exists
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(log_dir, 'final_model_training.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data_config(config_path=None):
    """Load data processing configuration."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, 'data_processing_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Data configuration loaded from: {config_path}")
        return config['data_processing']
    except Exception as e:
        logger.error(f"Error loading data config: {e}")
        return {
            'output_dir': 'data/3_train_test_data',
            'target_column': 'concrete_compressive_strength'
        }


def load_final_training_config(config_path=None):
    """Load final model training configuration."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, 'final_model_training_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Final training configuration loaded from: {config_path}")
        return config['final_model_training']
    except Exception as e:
        logger.error(f"Error loading final training config: {e}")
        logger.info("Using default final training configuration")
        return {
            'model_selection': 'auto',
            'auto_selection': {'metric': 'test_r2', 'source': 'hyperparameter_tuning'},
            'training': {'cv_folds': 5, 'save_model': True},
            'results': {'create_sample_directory': True}
        }


def load_train_test_data(data_dir):
    """Load training and test data."""
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
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e


def get_best_model_from_previous_runs(config):
    """Get the best model configuration from previous hyperparameter tuning or basic training."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')
    
    source = config['auto_selection']['source']
    metric = config['auto_selection']['metric']
    
    best_model_info = None
    
    try:
        if source == 'hyperparameter_tuning':
            hp_dir = os.path.join(results_dir, 'hyperparameter_tuning')
            if os.path.exists(hp_dir):
                # Find the latest tuning directory
                tuning_dirs = [d for d in os.listdir(hp_dir) if d.startswith('tuning_')]
                if tuning_dirs:
                    latest_dir = max(tuning_dirs, key=lambda x: int(x.split('_')[1]))
                    latest_path = os.path.join(hp_dir, latest_dir)
                    
                    # Look for CSV results file
                    csv_files = [f for f in os.listdir(latest_path) if f.endswith('_results.csv')]
                    if csv_files:
                        csv_file = os.path.join(latest_path, csv_files[0])
                        results_df = pd.read_csv(csv_file)
                        
                        # Sort by the specified metric
                        if metric == 'test_r2':
                            results_df = results_df.sort_values('Test_R2', ascending=False)
                        elif metric == 'cv_score':
                            results_df = results_df.sort_values('Best_CV_Score', ascending=False)
                        elif metric == 'test_rmse':
                            results_df = results_df.sort_values('Test_RMSE', ascending=True)
                        elif metric == 'test_mae':
                            results_df = results_df.sort_values('Test_MAE', ascending=True)
                        
                        best_model_info = {
                            'model': results_df.iloc[0]['Model'],
                            'source': 'hyperparameter_tuning',
                            'metric_value': results_df.iloc[0][metric.replace('_', ' ').title().replace(' ', '_')],
                            'tuning_dir': latest_path
                        }
        
        elif source == 'basic_training':
            training_dir = os.path.join(results_dir, 'training_results')
            if os.path.exists(training_dir):
                # Find the latest training directory
                sample_dirs = [d for d in os.listdir(training_dir) if d.startswith('sample_')]
                if sample_dirs:
                    latest_dir = max(sample_dirs, key=lambda x: int(x.split('_')[1]))
                    latest_path = os.path.join(training_dir, latest_dir)
                    
                    # Look for CSV results file
                    csv_files = [f for f in os.listdir(latest_path) if f.endswith('.csv')]
                    if csv_files:
                        csv_file = os.path.join(latest_path, csv_files[0])
                        results_df = pd.read_csv(csv_file)
                        
                        # Sort by the specified metric
                        if metric == 'test_r2':
                            results_df = results_df.sort_values('Test_R2', ascending=False)
                        
                        best_model_info = {
                            'model': results_df.iloc[0]['Model'],
                            'source': 'basic_training',
                            'metric_value': results_df.iloc[0]['Test_R2'],
                            'training_dir': latest_path
                        }
        
        if best_model_info:
            logger.info(f"Best model found: {best_model_info['model']} "
                       f"from {best_model_info['source']} with {metric}: {best_model_info['metric_value']:.4f}")
        else:
            logger.warning("No previous results found, will use default model configuration")
            
    except Exception as e:
        logger.error(f"Error finding best model from previous runs: {e}")
    
    return best_model_info


def get_best_parameters_for_model(model_name, best_model_info):
    """Extract best parameters for a specific model from previous tuning results."""
    if not best_model_info or best_model_info['source'] != 'hyperparameter_tuning':
        return None
    
    try:
        tuning_dir = best_model_info['tuning_dir']
        
        # Look for the model-specific detailed results file
        model_file_name = model_name.lower().replace(' ', '_')
        detail_files = [f for f in os.listdir(tuning_dir) 
                       if f.startswith(f'{model_file_name}_detailed_results_')]
        
        if detail_files:
            detail_file = os.path.join(tuning_dir, detail_files[0])
            
            # Parse the text file to extract best parameters
            with open(detail_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find the best parameters section
            if 'Best Parameters:' in content:
                params_start = content.find('Best Parameters:') + len('Best Parameters:')
                params_end = content.find('\n\n', params_start)
                params_str = content[params_start:params_end].strip()
                
                # Parse the parameters (they should be in dictionary format)
                try:
                    # Clean up the string and evaluate as Python dict
                    params_str = params_str.replace("'", '"')
                    best_params = eval(params_str)
                    logger.info(f"Best parameters found for {model_name}: {best_params}")
                    return best_params
                except:
                    logger.warning(f"Could not parse best parameters for {model_name}")
    
    except Exception as e:
        logger.error(f"Error extracting best parameters for {model_name}: {e}")
    
    return None


def create_model_with_config(model_name, config, best_model_info=None):
    """Create a model instance with the specified configuration."""
    model_config = config['models'][model_name.lower().replace(' ', '_')]
    
    if not model_config.get('enabled', True):
        return None
    
    # Determine parameters to use
    if model_config['parameters'] == 'auto' and best_model_info:
        # Try to get best parameters from previous tuning
        best_params = get_best_parameters_for_model(model_name, best_model_info)
        if best_params:
            params = best_params
        else:
            params = model_config['custom_parameters']
            logger.info(f"Using custom parameters for {model_name} (auto parameters not found)")
    else:
        params = model_config['custom_parameters']
        logger.info(f"Using custom parameters for {model_name}")
    
    # Create model instance
    try:
        if model_name == 'Random Forest':
            model = RandomForestRegressor(**params)
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(**params)
        elif model_name == 'Extra Trees':
            model = ExtraTreesRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Created {model_name} model with parameters: {params}")
        return model, params
        
    except Exception as e:
        logger.error(f"Error creating {model_name} model: {e}")
        return None, None


def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name, config):
    """Perform comprehensive model evaluation."""
    logger.info(f"Training and evaluating {model_name}...")
    
    start_time = time.time()
    
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = {
            'Model': model_name,
            'Training_Time': training_time,
            'Train_R2': r2_score(y_train, y_pred_train),
            'Test_R2': r2_score(y_test, y_pred_test),
            'Train_MSE': mean_squared_error(y_train, y_pred_train),
            'Test_MSE': mean_squared_error(y_test, y_pred_test),
            'Train_MAE': mean_absolute_error(y_train, y_pred_train),
            'Test_MAE': mean_absolute_error(y_test, y_pred_test),
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train_ExplainedVar': explained_variance_score(y_train, y_pred_train),
            'Test_ExplainedVar': explained_variance_score(y_test, y_pred_test),
            'Test_MaxError': max_error(y_test, y_pred_test),
            'Test_MedianAE': median_absolute_error(y_test, y_pred_test)
        }
        
        # Cross-validation scores if enabled
        if config.get('training', {}).get('cv_folds', 0) > 0:
            cv_folds = config['training']['cv_folds']
            cv_scoring = config['training'].get('cv_scoring', 'r2')
            
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=cv_scoring)
            
            metrics.update({
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'CV_Min': cv_scores.min(),
                'CV_Max': cv_scores.max()
            })
        
        logger.info(f"Model evaluation completed for {model_name}")
        logger.info(f"Test R² Score: {metrics['Test_R2']:.6f}")
        logger.info(f"Test RMSE: {metrics['Test_RMSE']:.4f}")
        
        return metrics, model, y_pred_train, y_pred_test
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return None, None, None, None


def create_results_directory(config):
    """Create directory structure for final training results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    main_results_dir = os.path.join(project_root, 'results')
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Create final training subdirectory
    final_training_dir = os.path.join(main_results_dir, 'final_model_training')
    os.makedirs(final_training_dir, exist_ok=True)
    
    if config.get('results', {}).get('create_sample_directory', True):
        # Find the next sample number
        existing_samples = [d for d in os.listdir(final_training_dir) 
                           if os.path.isdir(os.path.join(final_training_dir, d)) and d.startswith('final_')]
        
        if existing_samples:
            sample_numbers = [int(d.split('_')[1]) for d in existing_samples if d.split('_')[1].isdigit()]
            next_sample = max(sample_numbers) + 1 if sample_numbers else 1
        else:
            next_sample = 1
        
        # Create sample-specific directory
        sample_dir = os.path.join(final_training_dir, f'final_{next_sample}')
        os.makedirs(sample_dir, exist_ok=True)
        
        logger.info(f"Results will be saved to: {sample_dir}")
        return sample_dir
    else:
        logger.info(f"Results will be saved to: {final_training_dir}")
        return final_training_dir


def save_model(model, model_name, results_dir, config):
    """Save the trained model to disk."""
    if not config.get('training', {}).get('save_model', True):
        return None
    
    try:
        model_format = config.get('training', {}).get('model_format', 'joblib')
        safe_model_name = model_name.lower().replace(' ', '_')
        
        if model_format == 'joblib':
            model_file = os.path.join(results_dir, f'{safe_model_name}_final_model.joblib')
            joblib.dump(model, model_file)
        elif model_format == 'pickle':
            model_file = os.path.join(results_dir, f'{safe_model_name}_final_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Model saved: {model_file}")
        return model_file
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None


def save_comprehensive_results(results, model_params, config_used, results_dir, config):
    """Save comprehensive results and reports."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('Test_R2', ascending=False)
    
    report_files = []
    
    # 1. Performance Summary
    if 'performance_summary' in config.get('results', {}).get('generate_reports', []):
        summary_file = os.path.join(results_dir, f'final_training_summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("FINAL MODEL TRAINING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration Used: {config_used['model_selection']}\n\n")
            
            if len(results_df) > 0:
                f.write("MODEL PERFORMANCE RANKING:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Model':<15} {'Test R²':<10} {'Test RMSE':<12} {'CV Mean':<10}\n")
                f.write("-" * 60 + "\n")
                
                for _, row in results_df.iterrows():
                    cv_mean = row.get('CV_Mean', 'N/A')
                    cv_str = f"{cv_mean:.4f}" if cv_mean != 'N/A' else 'N/A'
                    f.write(f"{row['Model']:<15} {row['Test_R2']:<10.4f} {row['Test_RMSE']:<12.2f} {cv_str:<10}\n")
                
                f.write(f"\nBEST MODEL: {results_df.iloc[0]['Model']}\n")
                f.write(f"Best Test R² Score: {results_df.iloc[0]['Test_R2']:.6f}\n")
                f.write(f"Best Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}\n")
        
        report_files.append(summary_file)
        logger.info(f"Performance summary saved: {summary_file}")
    
    # 2. Detailed Metrics
    if 'detailed_metrics' in config.get('results', {}).get('generate_reports', []):
        detailed_file = os.path.join(results_dir, f'detailed_metrics_{timestamp}.txt')
        with open(detailed_file, 'w', encoding='utf-8') as f:
            f.write("DETAILED MODEL METRICS\n")
            f.write("=" * 70 + "\n\n")
            
            for _, row in results_df.iterrows():
                f.write(f"MODEL: {row['Model']}\n")
                f.write("-" * 40 + "\n")
                for col, val in row.items():
                    if col != 'Model':
                        f.write(f"{col}: {val}\n")
                f.write("\n")
        
        report_files.append(detailed_file)
        logger.info(f"Detailed metrics saved: {detailed_file}")
    
    # 3. Save CSV and JSON formats
    if 'csv' in config.get('results', {}).get('export_formats', []):
        csv_file = os.path.join(results_dir, f'final_training_results_{timestamp}.csv')
        results_df.to_csv(csv_file, index=False)
        report_files.append(csv_file)
    
    if 'json' in config.get('results', {}).get('export_formats', []):
        json_file = os.path.join(results_dir, f'final_training_results_{timestamp}.json')
        results_dict = {
            'results': results_df.to_dict('records'),
            'model_parameters': model_params,
            'configuration': config_used,
            'timestamp': timestamp
        }
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        report_files.append(json_file)
    
    return results_df, report_files


def main():
    """Main function to execute final model training pipeline."""
    try:
        logger.info("Starting Final Model Training Pipeline...")
        
        # Load configurations
        data_config = load_data_config()
        training_config = load_final_training_config()
        
        # Load data
        X_train, X_test, y_train, y_test = load_train_test_data(data_config['output_dir'])
        
        # Create results directory
        results_dir = create_results_directory(training_config)
        
        # Determine which model(s) to train
        model_selection = training_config.get('model_selection', 'auto')
        
        if model_selection == 'auto':
            # Get best model from previous runs
            best_model_info = get_best_model_from_previous_runs(training_config)
            if best_model_info:
                models_to_train = [best_model_info['model']]
                logger.info(f"Auto-selected model: {best_model_info['model']}")
            else:
                # Fallback to training all enabled models
                models_to_train = ['Random Forest', 'XGBoost', 'Extra Trees']
                logger.info("Auto-selection failed, training all enabled models")
                best_model_info = None
        else:
            # Train specific model
            model_mapping = {
                'random_forest': 'Random Forest',
                'xgboost': 'XGBoost', 
                'extra_trees': 'Extra Trees'
            }
            models_to_train = [model_mapping.get(model_selection, model_selection)]
            best_model_info = None
        
        # Train selected models
        results = []
        model_params = {}
        trained_models = {}
        
        for model_name in models_to_train:
            if model_name in ['Random Forest', 'XGBoost', 'Extra Trees']:
                # Create model with configuration
                model, params = create_model_with_config(model_name, training_config, best_model_info)
                
                if model is not None:
                    # Evaluate model
                    metrics, trained_model, y_pred_train, y_pred_test = evaluate_model_comprehensive(
                        model, X_train, X_test, y_train, y_test, model_name, training_config
                    )
                    
                    if metrics is not None:
                        results.append(metrics)
                        model_params[model_name] = params
                        trained_models[model_name] = trained_model
                        
                        # Save individual model
                        save_model(trained_model, model_name, results_dir, training_config)
        
        # Save comprehensive results
        if results:
            results_df, report_files = save_comprehensive_results(
                results, model_params, training_config, results_dir, training_config
            )
            
            # Display final summary
            print("\n" + "="*70)
            print("FINAL MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            if len(results_df) > 0:
                print(f"Best Model: {results_df.iloc[0]['Model']}")
                print(f"Test R² Score: {results_df.iloc[0]['Test_R2']:.4f}")
                print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}")
                print(f"Training Time: {results_df.iloc[0]['Training_Time']:.2f} seconds")
                
                if 'CV_Mean' in results_df.columns:
                    cv_score = results_df.iloc[0]['CV_Mean']
                    if pd.notna(cv_score):
                        print(f"Cross-Validation Score: {cv_score:.4f}")
            
            print(f"\nResults saved to: {results_dir}")
            print(f"Reports generated: {len(report_files)}")
            print("="*70)
            
            logger.info("Final model training pipeline completed successfully!")
            return results_df, trained_models
        else:
            logger.error("No models were trained successfully!")
            return None, None
        
    except Exception as e:
        logger.error(f"Error in final model training pipeline: {e}")
        raise e


if __name__ == "__main__":
    main()
