import os
import pandas as pd
import numpy as np
import logging
import yaml
import datetime
import warnings
import json
from itertools import product
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           explained_variance_score)
from scipy.stats import uniform, randint

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('hyper_parameter_tuning')
logger.setLevel(logging.DEBUG)

# Create handlers if not already exists
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(log_dir, 'hyper_parameter_tuning.log')
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


def load_hyperparameter_config(config_path=None):
    """
    Load hyperparameter tuning configuration from YAML file.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        config_path = os.path.join(parent_dir, 'hyperparameter_tuning_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Hyperparameter tuning configuration loaded from: {config_path}")
        return config['hyperparameter_tuning']
    except Exception as e:
        logger.error(f"Error loading hyperparameter config: {e}")
        # Use default configuration if file not found
        logger.info("Using default hyperparameter tuning configuration")
        return {
            'n_iterations': 500,
            'cv_folds': 5,
            'scoring': 'r2',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1,
            'models': {
                'random_forest': {'enabled': True, 'iterations': 500},
                'xgboost': {'enabled': True, 'iterations': 500},
                'extra_trees': {'enabled': True, 'iterations': 500}
            },
            'top_n_results': 10
        }


def load_train_test_data(data_dir):
    """
    Load training and test data from the data processing pipeline.
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
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e


def get_parameter_distributions(hp_config=None):
    """
    Define parameter distributions for hyperparameter tuning.
    """
    # Use default ranges if no config provided
    if hp_config is None or 'parameter_ranges' not in hp_config:
        param_distributions = {
            'Random Forest': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'random_state': [42]
            },
            
            'XGBoost': {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 15),
                'min_child_weight': randint(1, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 0.5),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1),
                'random_state': [42]
            },
            
            'Extra Trees': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'random_state': [42]
            }
        }
    else:
        # Use configuration from YAML file
        ranges = hp_config['parameter_ranges']
        random_state = hp_config.get('random_state', 42)
        
        param_distributions = {
            'Random Forest': {
                'n_estimators': randint(
                    ranges['random_forest']['n_estimators_min'],
                    ranges['random_forest']['n_estimators_max']
                ),
                'max_depth': randint(
                    ranges['random_forest']['max_depth_min'],
                    ranges['random_forest']['max_depth_max']
                ),
                'min_samples_split': randint(
                    ranges['random_forest']['min_samples_split_min'],
                    ranges['random_forest']['min_samples_split_max']
                ),
                'min_samples_leaf': randint(
                    ranges['random_forest']['min_samples_leaf_min'],
                    ranges['random_forest']['min_samples_leaf_max']
                ),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'random_state': [random_state]
            },
            
            'XGBoost': {
                'n_estimators': randint(
                    ranges['xgboost']['n_estimators_min'],
                    ranges['xgboost']['n_estimators_max']
                ),
                'learning_rate': uniform(
                    ranges['xgboost']['learning_rate_min'],
                    ranges['xgboost']['learning_rate_max'] - ranges['xgboost']['learning_rate_min']
                ),
                'max_depth': randint(
                    ranges['xgboost']['max_depth_min'],
                    ranges['xgboost']['max_depth_max']
                ),
                'min_child_weight': randint(
                    ranges['xgboost']['min_child_weight_min'],
                    ranges['xgboost']['min_child_weight_max']
                ),
                'subsample': uniform(
                    ranges['xgboost']['subsample_min'],
                    ranges['xgboost']['subsample_max'] - ranges['xgboost']['subsample_min']
                ),
                'colsample_bytree': uniform(
                    ranges['xgboost']['colsample_bytree_min'],
                    ranges['xgboost']['colsample_bytree_max'] - ranges['xgboost']['colsample_bytree_min']
                ),
                'gamma': uniform(
                    ranges['xgboost']['gamma_min'],
                    ranges['xgboost']['gamma_max'] - ranges['xgboost']['gamma_min']
                ),
                'reg_alpha': uniform(
                    ranges['xgboost']['reg_alpha_min'],
                    ranges['xgboost']['reg_alpha_max'] - ranges['xgboost']['reg_alpha_min']
                ),
                'reg_lambda': uniform(
                    ranges['xgboost']['reg_lambda_min'],
                    ranges['xgboost']['reg_lambda_max'] - ranges['xgboost']['reg_lambda_min']
                ),
                'random_state': [random_state]
            },
            
            'Extra Trees': {
                'n_estimators': randint(
                    ranges['extra_trees']['n_estimators_min'],
                    ranges['extra_trees']['n_estimators_max']
                ),
                'max_depth': randint(
                    ranges['extra_trees']['max_depth_min'],
                    ranges['extra_trees']['max_depth_max']
                ),
                'min_samples_split': randint(
                    ranges['extra_trees']['min_samples_split_min'],
                    ranges['extra_trees']['min_samples_split_max']
                ),
                'min_samples_leaf': randint(
                    ranges['extra_trees']['min_samples_leaf_min'],
                    ranges['extra_trees']['min_samples_leaf_max']
                ),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'random_state': [random_state]
            }
        }
    
    return param_distributions


def tune_hyperparameters(model, param_dist, X_train, y_train, model_name, hp_config, n_iter=None):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    # Get iterations from config or use provided value
    if n_iter is None:
        n_iter = hp_config.get('n_iterations', 500)
    
    logger.info(f"Starting hyperparameter tuning for {model_name}...")
    logger.info(f"Number of iterations: {n_iter}")
    
    try:
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=hp_config.get('cv_folds', 5),
            scoring=hp_config.get('scoring', 'r2'),
            n_jobs=hp_config.get('n_jobs', -1),
            random_state=hp_config.get('random_state', 42),
            verbose=hp_config.get('verbose', 0)
        )
        
        # Fit the random search
        random_search.fit(X_train, y_train)
        
        logger.info(f"Hyperparameter tuning completed for {model_name}")
        logger.info(f"Best CV Score: {random_search.best_score_:.6f}")
        
        return random_search
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning for {model_name}: {e}")
        raise e


def evaluate_best_model(random_search, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate the best model on test data.
    """
    try:
        best_model = random_search.best_estimator_
        
        # Make predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Best_CV_Score': random_search.best_score_,
            'Train_R2': r2_score(y_train, y_pred_train),
            'Test_R2': r2_score(y_test, y_pred_test),
            'Train_MSE': mean_squared_error(y_train, y_pred_train),
            'Test_MSE': mean_squared_error(y_test, y_pred_test),
            'Train_MAE': mean_absolute_error(y_train, y_pred_train),
            'Test_MAE': mean_absolute_error(y_test, y_pred_test),
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        return metrics, best_model
        
    except Exception as e:
        logger.error(f"Error evaluating best model for {model_name}: {e}")
        raise e


def save_detailed_results(random_search, model_name, results_dir, timestamp):
    """
    Save detailed results for each model including all parameter combinations tested.
    """
    detailed_file = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_detailed_results_{timestamp}.txt')
    
    # Get all results
    results_df = pd.DataFrame(random_search.cv_results_)
    
    # Sort by mean test score
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write(f"HYPERPARAMETER TUNING RESULTS - {model_name.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Parameter Combinations Tested: {len(results_df)}\n")
        f.write(f"Best CV Score: {random_search.best_score_:.6f}\n")
        f.write(f"Best Parameters: {random_search.best_params_}\n\n")
        
        f.write("ALL PARAMETER COMBINATIONS AND RESULTS:\n")
        f.write("-" * 80 + "\n\n")
        
        for idx, row in results_df.iterrows():
            f.write(f"Combination #{idx + 1}:\n")
            f.write(f"CV Score: {row['mean_test_score']:.6f} (+/- {row['std_test_score'] * 2:.6f})\n")
            f.write(f"Parameters: {row['params']}\n")
            f.write(f"Fit Time: {row['mean_fit_time']:.3f}s\n")
            f.write(f"Score Time: {row['mean_score_time']:.3f}s\n")
            f.write("-" * 50 + "\n")
    
    logger.info(f"Detailed results saved for {model_name}: {detailed_file}")
    return detailed_file


def create_results_directories():
    """
    Create directory structure for hyperparameter tuning results.
    """
    # Create main results directory in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    main_results_dir = os.path.join(project_root, 'results')
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Create hyperparameter tuning subdirectory
    hp_tuning_dir = os.path.join(main_results_dir, 'hyperparameter_tuning')
    os.makedirs(hp_tuning_dir, exist_ok=True)
    
    # Find the next sample number
    existing_samples = [d for d in os.listdir(hp_tuning_dir) 
                       if os.path.isdir(os.path.join(hp_tuning_dir, d)) and d.startswith('tuning_')]
    
    if existing_samples:
        sample_numbers = [int(d.split('_')[1]) for d in existing_samples if d.split('_')[1].isdigit()]
        next_sample = max(sample_numbers) + 1 if sample_numbers else 1
    else:
        next_sample = 1
    
    # Create sample-specific directory
    sample_dir = os.path.join(hp_tuning_dir, f'tuning_{next_sample}')
    os.makedirs(sample_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {sample_dir}")
    return sample_dir


def save_top_parameters(all_results, results_dir, timestamp, hp_config):
    """
    Save top N best parameters for each model in a single file.
    """
    top_n = hp_config.get('top_n_results', 10)
    top_params_file = os.path.join(results_dir, f'top_{top_n}_best_parameters_{timestamp}.txt')
    
    with open(top_params_file, 'w', encoding='utf-8') as f:
        f.write(f"TOP {top_n} BEST HYPERPARAMETERS FOR ALL MODELS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, random_search in all_results.items():
            f.write(f"{model_name.upper()} - TOP {top_n} PARAMETER COMBINATIONS\n")
            f.write("-" * 60 + "\n\n")
            
            # Get results dataframe and sort by score
            results_df = pd.DataFrame(random_search.cv_results_)
            results_df = results_df.sort_values('mean_test_score', ascending=False)
            
            # Get top N
            top_n_results = results_df.head(top_n)
            
            for idx, (_, row) in enumerate(top_n_results.iterrows(), 1):
                f.write(f"Rank #{idx}:\n")
                f.write(f"CV Score: {row['mean_test_score']:.6f} (+/- {row['std_test_score'] * 2:.6f})\n")
                f.write(f"Parameters:\n")
                for param, value in row['params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"Fit Time: {row['mean_fit_time']:.3f}s\n")
                f.write("\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    logger.info(f"Top {top_n} parameters saved: {top_params_file}")
    return top_params_file


def save_summary_results(model_evaluations, results_dir, timestamp, hp_config):
    """
    Save summary of best model performance.
    """
    summary_file = os.path.join(results_dir, f'hyperparameter_tuning_summary_{timestamp}.txt')
    
    # Convert to dataframe and sort by test R2
    summary_df = pd.DataFrame(model_evaluations)
    summary_df = summary_df.sort_values('Test_R2', ascending=False)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("HYPERPARAMETER TUNING SUMMARY RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tuning Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Tuned: Random Forest, XGBoost, Extra Trees\n")
        f.write(f"Default Iterations per Model: {hp_config.get('n_iterations', 500)}\n")
        f.write(f"Cross-Validation Folds: {hp_config.get('cv_folds', 5)}\n")
        f.write(f"Scoring Metric: {hp_config.get('scoring', 'r2')}\n\n")
        
        f.write("BEST MODEL PERFORMANCE AFTER TUNING:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<15} {'Best CV':<10} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<10}\n")
        f.write("-" * 70 + "\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['Model']:<15} {row['Best_CV_Score']:<10.4f} {row['Test_R2']:<10.4f} {row['Test_RMSE']:<12.2f} {row['Test_MAE']:<10.2f}\n")
        
        f.write(f"\nBEST OVERALL MODEL: {summary_df.iloc[0]['Model']}\n")
        f.write(f"Best Test R² Score: {summary_df.iloc[0]['Test_R2']:.6f}\n")
        f.write(f"Best CV Score: {summary_df.iloc[0]['Best_CV_Score']:.6f}\n")
    
    # Save CSV version
    csv_file = os.path.join(results_dir, f'hyperparameter_tuning_results_{timestamp}.csv')
    summary_df.to_csv(csv_file, index=False)
    
    logger.info(f"Summary results saved: {summary_file}")
    logger.info(f"CSV results saved: {csv_file}")
    
    return summary_df


def main():
    """
    Main function to execute hyperparameter tuning pipeline.
    """
    try:
        logger.info("Starting Hyperparameter Tuning Pipeline...")
        
        # Load configurations
        config = load_config()
        hp_config = load_hyperparameter_config()
        
        # Load data
        X_train, X_test, y_train, y_test = load_train_test_data(config['output_dir'])
        
        # Create results directory
        results_dir = create_results_directories()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get parameter distributions
        param_distributions = get_parameter_distributions(hp_config)
        
        # Define models with configuration
        models = {
            'Random Forest': RandomForestRegressor(n_jobs=hp_config.get('n_jobs', -1)),
            'XGBoost': xgb.XGBRegressor(),
            'Extra Trees': ExtraTreesRegressor(n_jobs=hp_config.get('n_jobs', -1))
        }
        
        # Filter models based on configuration
        enabled_models = {}
        for model_name, model in models.items():
            model_key = model_name.lower().replace(' ', '_')
            if hp_config.get('models', {}).get(model_key, {}).get('enabled', True):
                enabled_models[model_name] = model
        
        logger.info(f"Enabled models: {list(enabled_models.keys())}")
        
        all_results = {}
        model_evaluations = []
        detailed_files = []
        
        # Tune each enabled model
        for model_name, model in enabled_models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"TUNING {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Get model-specific iterations if configured
            model_key = model_name.lower().replace(' ', '_')
            model_iterations = hp_config.get('models', {}).get(model_key, {}).get('iterations')
            
            # Perform hyperparameter tuning
            random_search = tune_hyperparameters(
                model, 
                param_distributions[model_name], 
                X_train, 
                y_train, 
                model_name,
                hp_config,
                n_iter=model_iterations
            )
            
            # Store results
            all_results[model_name] = random_search
            
            # Evaluate best model
            metrics, best_model = evaluate_best_model(
                random_search, X_train, X_test, y_train, y_test, model_name
            )
            model_evaluations.append(metrics)
            
            # Save detailed results for this model
            detailed_file = save_detailed_results(
                random_search, model_name, results_dir, timestamp
            )
            detailed_files.append(detailed_file)
            
            logger.info(f"Completed tuning for {model_name}")
        
        # Save top N parameters for all models
        top_params_file = save_top_parameters(all_results, results_dir, timestamp, hp_config)
        
        # Save summary results
        summary_df = save_summary_results(model_evaluations, results_dir, timestamp, hp_config)
        
        # Display final summary
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best Overall Model: {summary_df.iloc[0]['Model']}")
        print(f"Best Test R² Score: {summary_df.iloc[0]['Test_R2']:.4f}")
        print(f"Best CV Score: {summary_df.iloc[0]['Best_CV_Score']:.4f}")
        print(f"\nConfiguration Used:")
        print(f"- Default Iterations: {hp_config.get('n_iterations', 500)}")
        print(f"- CV Folds: {hp_config.get('cv_folds', 5)}")
        print(f"- Scoring: {hp_config.get('scoring', 'r2')}")
        print(f"\nResults saved to: {results_dir}")
        print(f"Detailed files: {len(detailed_files)} files created")
        print(f"Top parameters file: {top_params_file}")
        print("="*80)
        
        logger.info("Hyperparameter tuning pipeline completed successfully!")
        
        return summary_df, all_results
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning pipeline: {e}")
        raise e


if __name__ == "__main__":
    main()
