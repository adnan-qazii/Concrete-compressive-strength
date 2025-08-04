import os
import pandas as pd
import numpy as np
import logging
import yaml
import datetime
import warnings
import joblib
import pickle
from pathlib import Path
import json
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    explained_variance_score, max_error, median_absolute_error,
    mean_absolute_percentage_error
)

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('final_model_evaluation')
logger.setLevel(logging.DEBUG)

# Create handlers if not already exists
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(log_dir, 'final_model_evaluation.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data_config():
    """Load data processing configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_path = os.path.join(parent_dir, 'data_processing_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['data_processing']
    except Exception as e:
        logger.error(f"Error loading data config: {e}")
        return {'output_dir': 'data/3_train_test_data'}


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


def find_best_trained_model():
    """Find the best trained model from previous training runs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')
    
    best_model_path = None
    best_model_info = None
    best_score = -np.inf
    
    # Check final model training results first
    final_training_dir = os.path.join(results_dir, 'final_model_training')
    if os.path.exists(final_training_dir):
        for final_dir in os.listdir(final_training_dir):
            final_path = os.path.join(final_training_dir, final_dir)
            if os.path.isdir(final_path):
                # Look for CSV results and model files
                csv_files = [f for f in os.listdir(final_path) if f.endswith('.csv')]
                model_files = [f for f in os.listdir(final_path) if f.endswith('.joblib') or f.endswith('.pkl')]
                
                if csv_files and model_files:
                    csv_file = os.path.join(final_path, csv_files[0])
                    try:
                        results_df = pd.read_csv(csv_file)
                        if len(results_df) > 0:
                            best_row = results_df.loc[results_df['Test_R2'].idxmax()]
                            if best_row['Test_R2'] > best_score:
                                best_score = best_row['Test_R2']
                                model_name = best_row['Model'].lower().replace(' ', '_')
                                
                                # Find corresponding model file
                                for model_file in model_files:
                                    if model_name in model_file:
                                        best_model_path = os.path.join(final_path, model_file)
                                        best_model_info = {
                                            'model_name': best_row['Model'],
                                            'test_r2': best_row['Test_R2'],
                                            'test_rmse': best_row['Test_RMSE'],
                                            'source': 'final_training',
                                            'model_file': best_model_path
                                        }
                                        break
                    except Exception as e:
                        logger.warning(f"Error reading results from {csv_file}: {e}")
    
    # If no final training results, check basic training results
    if best_model_path is None:
        training_results_dir = os.path.join(results_dir, 'training_results')
        if os.path.exists(training_results_dir):
            logger.info("No final training results found, checking basic training results...")
            # This would require us to retrain the best model since basic training doesn't save models
            logger.warning("Basic training results found but no saved models. Please run final_model_training.py first.")
    
    if best_model_path and os.path.exists(best_model_path):
        logger.info(f"Best model found: {best_model_info['model_name']} with Test R² = {best_model_info['test_r2']:.4f}")
        logger.info(f"Model file: {best_model_path}")
        return best_model_path, best_model_info
    else:
        logger.error("No trained model found. Please run final_model_training.py first.")
        return None, None


def load_model(model_path):
    """Load a trained model from file."""
    try:
        if model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model file format: {model_path}")
        
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise e


def perform_comprehensive_evaluation(model, X_train, X_test, y_train, y_test, model_info):
    """Perform comprehensive model evaluation."""
    logger.info("Performing comprehensive model evaluation...")
    
    evaluation_results = {}
    
    # Basic predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Basic metrics
    evaluation_results['basic_metrics'] = {
        'Train_R2': r2_score(y_train, y_pred_train),
        'Test_R2': r2_score(y_test, y_pred_test),
        'Train_MSE': mean_squared_error(y_train, y_pred_train),
        'Test_MSE': mean_squared_error(y_test, y_pred_test),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train_MAE': mean_absolute_error(y_train, y_pred_train),
        'Test_MAE': mean_absolute_error(y_test, y_pred_test),
        'Train_ExplainedVar': explained_variance_score(y_train, y_pred_train),
        'Test_ExplainedVar': explained_variance_score(y_test, y_pred_test),
        'Test_MaxError': max_error(y_test, y_pred_test),
        'Test_MedianAE': median_absolute_error(y_test, y_pred_test)
    }
    
    # Mean Absolute Percentage Error
    try:
        train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
        test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
        evaluation_results['basic_metrics']['Train_MAPE'] = train_mape
        evaluation_results['basic_metrics']['Test_MAPE'] = test_mape
    except:
        pass
    
    # Cross-validation performance
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    evaluation_results['cross_validation'] = {
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std(),
        'CV_R2_Min': cv_scores.min(),
        'CV_R2_Max': cv_scores.max(),
        'CV_Scores': cv_scores.tolist()
    }
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        logger.info("Analyzing feature importance...")
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        evaluation_results['feature_importance'] = {
            'top_10_features': feature_importance.head(10).to_dict('records'),
            'all_features': feature_importance.to_dict('records')
        }
    
    # Prediction analysis
    logger.info("Analyzing predictions...")
    
    # Residuals analysis
    test_residuals = y_test - y_pred_test
    train_residuals = y_train - y_pred_train
    
    evaluation_results['prediction_analysis'] = {
        'test_residuals_mean': np.mean(test_residuals),
        'test_residuals_std': np.std(test_residuals),
        'test_residuals_min': np.min(test_residuals),
        'test_residuals_max': np.max(test_residuals),
        'train_residuals_mean': np.mean(train_residuals),
        'train_residuals_std': np.std(train_residuals),
        'prediction_range_test': {
            'min_actual': np.min(y_test),
            'max_actual': np.max(y_test),
            'min_predicted': np.min(y_pred_test),
            'max_predicted': np.max(y_pred_test)
        },
        'accuracy_within_tolerance': {}
    }
    
    # Calculate accuracy within different tolerance levels
    for tolerance in [5, 10, 15, 20]:  # MPa tolerance
        within_tolerance = np.sum(np.abs(test_residuals) <= tolerance)
        percentage = (within_tolerance / len(y_test)) * 100
        evaluation_results['prediction_analysis']['accuracy_within_tolerance'][f'{tolerance}_MPa'] = {
            'count': int(within_tolerance),
            'percentage': percentage
        }
    
    # Model complexity analysis
    evaluation_results['model_info'] = {
        'model_type': type(model).__name__,
        'model_name': model_info['model_name'],
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    # Add model-specific parameters
    if hasattr(model, 'get_params'):
        evaluation_results['model_parameters'] = model.get_params()
    
    logger.info("Comprehensive evaluation completed")
    return evaluation_results, y_pred_train, y_pred_test


def create_final_results_directory():
    """Create directory for final results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')
    final_results_dir = os.path.join(results_dir, 'final_results')
    os.makedirs(final_results_dir, exist_ok=True)
    
    logger.info(f"Final results will be saved to: {final_results_dir}")
    return final_results_dir


def save_final_evaluation_report(evaluation_results, model_info, y_pred_train, y_pred_test, 
                                y_train, y_test, X_train, X_test, results_dir):
    """Save comprehensive evaluation report to a single text file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(results_dir, f'final_model_evaluation_report_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("FINAL MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_info['model_name']}\n")
        f.write(f"Source: {model_info['source']}\n")
        f.write(f"Model File: {model_info['model_file']}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        basic_metrics = evaluation_results['basic_metrics']
        f.write(f"Model Performance (Test Set):\n")
        f.write(f"  R² Score: {basic_metrics['Test_R2']:.6f}\n")
        f.write(f"  RMSE: {basic_metrics['Test_RMSE']:.4f} MPa\n")
        f.write(f"  MAE: {basic_metrics['Test_MAE']:.4f} MPa\n")
        f.write(f"  Max Error: {basic_metrics['Test_MaxError']:.4f} MPa\n")
        
        cv_results = evaluation_results['cross_validation']
        f.write(f"\nCross-Validation Performance:\n")
        f.write(f"  Mean R² Score: {cv_results['CV_R2_Mean']:.6f} (±{cv_results['CV_R2_Std']:.6f})\n")
        f.write(f"  Range: {cv_results['CV_R2_Min']:.6f} to {cv_results['CV_R2_Max']:.6f}\n")
        
        # Model generalization assessment
        overfitting_check = basic_metrics['Train_R2'] - basic_metrics['Test_R2']
        f.write(f"\nModel Generalization:\n")
        f.write(f"  Training R²: {basic_metrics['Train_R2']:.6f}\n")
        f.write(f"  Test R²: {basic_metrics['Test_R2']:.6f}\n")
        f.write(f"  Difference: {overfitting_check:.6f}")
        if overfitting_check > 0.1:
            f.write(" (Potential Overfitting)")
        elif overfitting_check < 0.05:
            f.write(" (Good Generalization)")
        f.write("\n\n")
        
        # Detailed Performance Metrics
        f.write("DETAILED PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("Training Set Performance:\n")
        f.write(f"  R² Score: {basic_metrics['Train_R2']:.6f}\n")
        f.write(f"  MSE: {basic_metrics['Train_MSE']:.6f}\n")
        f.write(f"  RMSE: {basic_metrics['Train_RMSE']:.4f} MPa\n")
        f.write(f"  MAE: {basic_metrics['Train_MAE']:.4f} MPa\n")
        f.write(f"  Explained Variance: {basic_metrics['Train_ExplainedVar']:.6f}\n")
        
        if 'Train_MAPE' in basic_metrics:
            f.write(f"  MAPE: {basic_metrics['Train_MAPE']:.4f}%\n")
        
        f.write(f"\nTest Set Performance:\n")
        f.write(f"  R² Score: {basic_metrics['Test_R2']:.6f}\n")
        f.write(f"  MSE: {basic_metrics['Test_MSE']:.6f}\n")
        f.write(f"  RMSE: {basic_metrics['Test_RMSE']:.4f} MPa\n")
        f.write(f"  MAE: {basic_metrics['Test_MAE']:.4f} MPa\n")
        f.write(f"  Explained Variance: {basic_metrics['Test_ExplainedVar']:.6f}\n")
        f.write(f"  Max Error: {basic_metrics['Test_MaxError']:.4f} MPa\n")
        f.write(f"  Median AE: {basic_metrics['Test_MedianAE']:.4f} MPa\n")
        
        if 'Test_MAPE' in basic_metrics:
            f.write(f"  MAPE: {basic_metrics['Test_MAPE']:.4f}%\n")
        
        # Cross-Validation Results
        f.write(f"\nCross-Validation Results (5-Fold):\n")
        f.write(f"  Mean R² Score: {cv_results['CV_R2_Mean']:.6f}\n")
        f.write(f"  Standard Deviation: {cv_results['CV_R2_Std']:.6f}\n")
        f.write(f"  Minimum Score: {cv_results['CV_R2_Min']:.6f}\n")
        f.write(f"  Maximum Score: {cv_results['CV_R2_Max']:.6f}\n")
        f.write(f"  Individual Scores: {[f'{score:.6f}' for score in cv_results['CV_Scores']]}\n\n")
        
        # Prediction Accuracy Analysis
        f.write("PREDICTION ACCURACY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        pred_analysis = evaluation_results['prediction_analysis']
        
        f.write("Residual Statistics (Test Set):\n")
        f.write(f"  Mean Residual: {pred_analysis['test_residuals_mean']:.4f} MPa\n")
        f.write(f"  Std Residual: {pred_analysis['test_residuals_std']:.4f} MPa\n")
        f.write(f"  Min Residual: {pred_analysis['test_residuals_min']:.4f} MPa\n")
        f.write(f"  Max Residual: {pred_analysis['test_residuals_max']:.4f} MPa\n")
        
        f.write(f"\nPrediction Range Analysis:\n")
        pred_range = pred_analysis['prediction_range_test']
        f.write(f"  Actual Values Range: {pred_range['min_actual']:.2f} to {pred_range['max_actual']:.2f} MPa\n")
        f.write(f"  Predicted Values Range: {pred_range['min_predicted']:.2f} to {pred_range['max_predicted']:.2f} MPa\n")
        
        f.write(f"\nAccuracy Within Tolerance Levels:\n")
        for tolerance, stats in pred_analysis['accuracy_within_tolerance'].items():
            f.write(f"  Within ±{tolerance.replace('_', ' ')}: {stats['count']}/{len(y_test)} ({stats['percentage']:.1f}%)\n")
        
        # Feature Importance Analysis
        if 'feature_importance' in evaluation_results:
            f.write(f"\nFEATURE IMPORTANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write("Top 10 Most Important Features:\n")
            
            for i, feature in enumerate(evaluation_results['feature_importance']['top_10_features'], 1):
                f.write(f"  {i:2d}. {feature['Feature']:<30}: {feature['Importance']:.6f}\n")
            
            f.write(f"\nAll Features Importance (sorted by importance):\n")
            for i, feature in enumerate(evaluation_results['feature_importance']['all_features'], 1):
                f.write(f"  {i:2d}. {feature['Feature']:<30}: {feature['Importance']:.6f}\n")
        
        # Model Information
        f.write(f"\nMODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        model_info_details = evaluation_results['model_info']
        f.write(f"Model Type: {model_info_details['model_type']}\n")
        f.write(f"Model Name: {model_info_details['model_name']}\n")
        f.write(f"Number of Features: {model_info_details['n_features']}\n")
        f.write(f"Training Samples: {model_info_details['n_train_samples']}\n")
        f.write(f"Test Samples: {model_info_details['n_test_samples']}\n")
        
        # Model Parameters
        if 'model_parameters' in evaluation_results:
            f.write(f"\nModel Parameters:\n")
            for param, value in evaluation_results['model_parameters'].items():
                f.write(f"  {param}: {value}\n")
        
        # Data Quality Assessment
        f.write(f"\nDATA QUALITY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Data:\n")
        f.write(f"  Shape: {X_train.shape}\n")
        f.write(f"  Target Mean: {np.mean(y_train):.2f} MPa\n")
        f.write(f"  Target Std: {np.std(y_train):.2f} MPa\n")
        f.write(f"  Target Range: {np.min(y_train):.2f} to {np.max(y_train):.2f} MPa\n")
        
        f.write(f"\nTest Data:\n")
        f.write(f"  Shape: {X_test.shape}\n")
        f.write(f"  Target Mean: {np.mean(y_test):.2f} MPa\n")
        f.write(f"  Target Std: {np.std(y_test):.2f} MPa\n")
        f.write(f"  Target Range: {np.min(y_test):.2f} to {np.max(y_test):.2f} MPa\n")
        
        # Performance Interpretation
        f.write(f"\nPERFORMANCE INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        
        test_r2 = basic_metrics['Test_R2']
        if test_r2 >= 0.9:
            performance_level = "Excellent"
        elif test_r2 >= 0.8:
            performance_level = "Very Good"
        elif test_r2 >= 0.7:
            performance_level = "Good"
        elif test_r2 >= 0.6:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        f.write(f"Overall Model Performance: {performance_level}\n")
        f.write(f"R² Score of {test_r2:.3f} indicates that the model explains ")
        f.write(f"{test_r2*100:.1f}% of the variance in concrete strength.\n\n")
        
        rmse = basic_metrics['Test_RMSE']
        mae = basic_metrics['Test_MAE']
        target_mean = np.mean(y_test)
        
        f.write(f"Error Analysis:\n")
        f.write(f"  RMSE of {rmse:.2f} MPa represents {(rmse/target_mean)*100:.1f}% of mean target value\n")
        f.write(f"  MAE of {mae:.2f} MPa represents {(mae/target_mean)*100:.1f}% of mean target value\n")
        
        # Recommendations
        f.write(f"\nRECOMMENDations\n")
        f.write("-" * 40 + "\n")
        
        if test_r2 >= 0.85:
            f.write("✓ Model shows excellent predictive performance\n")
            f.write("✓ Suitable for production use with confidence\n")
        elif test_r2 >= 0.75:
            f.write("✓ Model shows good predictive performance\n")
            f.write("• Consider further hyperparameter tuning for improvement\n")
        else:
            f.write("• Model performance could be improved\n")
            f.write("• Consider feature engineering or different algorithms\n")
        
        if overfitting_check > 0.1:
            f.write("• Model shows signs of overfitting - consider regularization\n")
        elif overfitting_check < 0.02:
            f.write("✓ Model generalizes well to unseen data\n")
        
        cv_std = cv_results['CV_R2_Std']
        if cv_std < 0.05:
            f.write("✓ Model shows consistent performance across different data splits\n")
        else:
            f.write("• Model performance varies across data splits - investigate data quality\n")
        
        f.write(f"\nReport generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Final evaluation report saved: {report_file}")
    return report_file


def main():
    """Main function to execute final model evaluation."""
    try:
        logger.info("Starting Final Model Evaluation...")
        
        # Find and load the best trained model
        model_path, model_info = find_best_trained_model()
        
        if model_path is None:
            logger.error("No trained model found. Please run final_model_training.py first.")
            return None
        
        # Load the model
        model = load_model(model_path)
        
        # Load data
        data_config = load_data_config()
        X_train, X_test, y_train, y_test = load_train_test_data(data_config['output_dir'])
        
        # Perform comprehensive evaluation
        evaluation_results, y_pred_train, y_pred_test = perform_comprehensive_evaluation(
            model, X_train, X_test, y_train, y_test, model_info
        )
        
        # Create results directory
        results_dir = create_final_results_directory()
        
        # Save comprehensive report
        report_file = save_final_evaluation_report(
            evaluation_results, model_info, y_pred_train, y_pred_test,
            y_train, y_test, X_train, X_test, results_dir
        )
        
        # Display summary
        basic_metrics = evaluation_results['basic_metrics']
        cv_results = evaluation_results['cross_validation']
        
        print("\n" + "="*70)
        print("FINAL MODEL EVALUATION COMPLETED!")
        print("="*70)
        print(f"Model: {model_info['model_name']}")
        print(f"Test R² Score: {basic_metrics['Test_R2']:.6f}")
        print(f"Test RMSE: {basic_metrics['Test_RMSE']:.4f} MPa")
        print(f"Test MAE: {basic_metrics['Test_MAE']:.4f} MPa")
        print(f"Cross-Validation R²: {cv_results['CV_R2_Mean']:.6f} (±{cv_results['CV_R2_Std']:.6f})")
        print(f"\nDetailed report saved to: {report_file}")
        print("="*70)
        
        logger.info("Final model evaluation completed successfully!")
        return evaluation_results, report_file
        
    except Exception as e:
        logger.error(f"Error in final model evaluation: {e}")
        raise e


if __name__ == "__main__":
    main()
