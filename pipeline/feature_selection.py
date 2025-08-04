import os
import pandas as pd
import numpy as np
from data_ingestion import load_data, preprocess_data
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Configure logging
# ensure the logging directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_selection')
logger.setLevel(logging.DEBUG)

# Create handlers if not already exists
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_file_path = os.path.join(log_dir, 'feature_selection.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def create_engineered_features(df):
    """
    Create engineered features for concrete strength prediction.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    logger.info("Starting feature engineering...")
    
    # Create a copy to avoid modifying original data
    featured_df = df.copy()
    
    # 1. Total Cement Content (cement + blast furnace slag)
    if 'cement' in featured_df.columns and 'blast_furnace_slag' in featured_df.columns:
        featured_df['total_cement'] = featured_df['cement'] + featured_df['blast_furnace_slag']
        logger.info("Created 'total_cement' feature.")
    
    # 2. Water-to-Cement Ratio (critical for concrete strength)
    if 'water' in featured_df.columns and 'cement' in featured_df.columns:
        featured_df['water_cement_ratio'] = featured_df['water'] / (featured_df['cement'] + 1e-8)
        logger.info("Created 'water_cement_ratio' feature.")
    
    # 3. Total Aggregate Content
    if 'coarse_aggregate' in featured_df.columns and 'fine_aggregate' in featured_df.columns:
        featured_df['total_aggregate'] = featured_df['coarse_aggregate'] + featured_df['fine_aggregate']
        logger.info("Created 'total_aggregate' feature.")
    
    # 4. Age Categories (concrete strength development stages)
    if 'age' in featured_df.columns:
        featured_df['age_category_early'] = (featured_df['age'] <= 7).astype(int)
        featured_df['age_category_standard'] = ((featured_df['age'] > 7) & (featured_df['age'] <= 28)).astype(int)
        featured_df['age_category_extended'] = ((featured_df['age'] > 28) & (featured_df['age'] <= 90)).astype(int)
        featured_df['age_category_longterm'] = (featured_df['age'] > 90).astype(int)
        logger.info("Created age category features.")
    
    # 5. Cement-to-Total-Material Ratio
    material_cols = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 
                     'coarse_aggregate', 'fine_aggregate']
    available_cols = [col for col in material_cols if col in featured_df.columns]
    if len(available_cols) > 1:
        featured_df['total_material'] = featured_df[available_cols].sum(axis=1)
        if 'cement' in featured_df.columns:
            featured_df['cement_material_ratio'] = featured_df['cement'] / (featured_df['total_material'] + 1e-8)
            logger.info("Created 'cement_material_ratio' feature.")
    
    # 6. Supplementary Cementitious Materials (SCM) ratio
    scm_cols = ['blast_furnace_slag', 'fly_ash']
    available_scm = [col for col in scm_cols if col in featured_df.columns]
    if available_scm and 'cement' in featured_df.columns:
        featured_df['scm_total'] = featured_df[available_scm].sum(axis=1)
        featured_df['scm_cement_ratio'] = featured_df['scm_total'] / (featured_df['cement'] + 1e-8)
        logger.info("Created 'scm_cement_ratio' feature.")
    
    # 7. Superplasticizer Efficiency (superplasticizer per unit cement)
    if 'superplasticizer' in featured_df.columns and 'cement' in featured_df.columns:
        featured_df['superplasticizer_efficiency'] = featured_df['superplasticizer'] / (featured_df['cement'] + 1e-8)
        logger.info("Created 'superplasticizer_efficiency' feature.")
    
    # 8. Aggregate-to-Cement Ratio
    if 'total_aggregate' in featured_df.columns and 'cement' in featured_df.columns:
        featured_df['aggregate_cement_ratio'] = featured_df['total_aggregate'] / (featured_df['cement'] + 1e-8)
        logger.info("Created 'aggregate_cement_ratio' feature.")
    
    # 9. Fine-to-Coarse Aggregate Ratio
    if 'fine_aggregate' in featured_df.columns and 'coarse_aggregate' in featured_df.columns:
        featured_df['fine_coarse_ratio'] = featured_df['fine_aggregate'] / (featured_df['coarse_aggregate'] + 1e-8)
        logger.info("Created 'fine_coarse_ratio' feature.")
    
    # 10. Water-to-Total-Powder Ratio (water to all powder materials)
    powder_cols = ['cement', 'blast_furnace_slag', 'fly_ash']
    available_powder = [col for col in powder_cols if col in featured_df.columns]
    if available_powder and 'water' in featured_df.columns:
        featured_df['total_powder'] = featured_df[available_powder].sum(axis=1)
        featured_df['water_powder_ratio'] = featured_df['water'] / (featured_df['total_powder'] + 1e-8)
        logger.info("Created 'water_powder_ratio' feature.")
    
    # 11. Age-Strength Development Factor (logarithmic age transformation)
    if 'age' in featured_df.columns:
        featured_df['log_age'] = np.log1p(featured_df['age'])  # log(1 + age) to handle age=0
        featured_df['sqrt_age'] = np.sqrt(featured_df['age'])
        logger.info("Created age transformation features.")
    
    # 12. Cement Quality Index (cement + supplementary materials efficiency)
    if 'cement' in featured_df.columns and 'scm_total' in featured_df.columns:
        featured_df['cement_quality_index'] = featured_df['cement'] + 0.7 * featured_df['scm_total']  # SCM efficiency factor
        logger.info("Created 'cement_quality_index' feature.")
    
    # 13. Workability Index (water + superplasticizer interaction)
    if 'water' in featured_df.columns and 'superplasticizer' in featured_df.columns:
        featured_df['workability_index'] = featured_df['water'] + 5 * featured_df['superplasticizer']  # SP efficiency factor
        logger.info("Created 'workability_index' feature.")
    
    logger.info(f"Feature engineering completed. New shape: {featured_df.shape}")
    return featured_df


def perform_feature_selection(df, target_column='concrete_compressive_strength', k_best=15):
    """
    Perform feature selection using statistical and mutual information methods.
    
    Args:
        df (pd.DataFrame): Dataset with engineered features
        target_column (str): Target variable column name
        k_best (int): Number of best features to select
        
    Returns:
        tuple: (selected_features_df, selected_feature_names, feature_scores)
    """
    logger.info("Starting feature selection...")
    
    # Separate features and target
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataset")
        return df, df.columns.tolist(), {}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Remove any non-numeric columns for feature selection
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_features]
    
    logger.info(f"Performing feature selection on {len(numeric_features)} numeric features")
    
    # Method 1: F-regression (linear relationship)
    f_selector = SelectKBest(score_func=f_regression, k=min(k_best, len(numeric_features)))
    X_f_selected = f_selector.fit_transform(X_numeric, y)
    f_scores = dict(zip(numeric_features, f_selector.scores_))
    f_selected_features = numeric_features[f_selector.get_support()]
    
    # Method 2: Mutual Information (non-linear relationships)
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(k_best, len(numeric_features)))
    X_mi_selected = mi_selector.fit_transform(X_numeric, y)
    mi_scores = dict(zip(numeric_features, mi_selector.scores_))
    mi_selected_features = numeric_features[mi_selector.get_support()]
    
    # Combine both methods (union of selected features)
    combined_features = list(set(f_selected_features) | set(mi_selected_features))
    
    # Create final dataset with selected features + target
    final_features = combined_features + [target_column]
    selected_df = df[final_features]
    
    # Feature importance scores
    feature_scores = {
        'f_regression_scores': f_scores,
        'mutual_info_scores': mi_scores,
        'selected_by_f_regression': f_selected_features.tolist(),
        'selected_by_mutual_info': mi_selected_features.tolist(),
        'final_selected_features': combined_features
    }
    
    logger.info(f"Feature selection completed. Selected {len(combined_features)} features from {len(numeric_features)}")
    logger.info(f"Selected features: {combined_features}")
    
    return selected_df, combined_features, feature_scores


def normalize_features(df, target_column='concrete_compressive_strength', method='standard'):
    """
    Normalize features while keeping target variable unchanged.
    
    Args:
        df (pd.DataFrame): Dataset to normalize
        target_column (str): Target variable column name
        method (str): 'standard' or 'minmax'
        
    Returns:
        tuple: (normalized_df, scaler_object)
    """
    logger.info(f"Starting feature normalization using {method} scaling...")
    
    # Separate features and target
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    # Select numeric features only
    numeric_features = features.select_dtypes(include=[np.number])
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        logger.error(f"Unknown normalization method: {method}")
        return df, None
    
    # Fit and transform features
    scaled_features = scaler.fit_transform(numeric_features)
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_features.columns, index=df.index)
    
    # Add target column back
    scaled_df[target_column] = target
    
    logger.info(f"Feature normalization completed using {method} scaling")
    return scaled_df, scaler


def main():
    """
    Main function to execute feature selection pipeline.
    """
    try:
        logger.info("Starting feature selection pipeline...")
        
        # Load and preprocess data
        file_path = 'data/raw/concrete_data.csv'
        raw_data = load_data(file_path)
        preprocessed_data = preprocess_data(raw_data)
        
        # Create engineered features
        featured_data = create_engineered_features(preprocessed_data)
        
        # Perform feature selection
        selected_data, selected_features, feature_scores = perform_feature_selection(featured_data)
        
        # Normalize features
        normalized_data, scaler = normalize_features(selected_data)
        
        # Create output directory
        output_dir = 'data/2_feature_selection'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        # 1. Featured dataset (with all engineered features)
        featured_path = os.path.join(output_dir, 'featured_concrete_data.csv')
        featured_data.to_csv(featured_path, index=False)
        logger.info(f"Featured dataset saved to {featured_path}")
        
        # 2. Selected features dataset
        selected_path = os.path.join(output_dir, 'selected_features_data.csv')
        selected_data.to_csv(selected_path, index=False)
        logger.info(f"Selected features dataset saved to {selected_path}")
        
        # 3. Normalized dataset (final for ML)
        normalized_path = os.path.join(output_dir, 'normalized_concrete_data.csv')
        normalized_data.to_csv(normalized_path, index=False)
        logger.info(f"Normalized dataset saved to {normalized_path}")
        
        # 4. Feature selection report
        report_path = os.path.join(output_dir, 'feature_selection_report.txt')
        with open(report_path, 'w') as f:
            f.write("CONCRETE STRENGTH PREDICTION - FEATURE SELECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Original features: {len(preprocessed_data.columns)}\n")
            f.write(f"Engineered features: {len(featured_data.columns)}\n")
            f.write(f"Selected features: {len(selected_features)}\n\n")
            f.write("SELECTED FEATURES:\n")
            f.write("-" * 20 + "\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"{i:2d}. {feature}\n")
            f.write(f"\nDataset saved to: {output_dir}\n")
        
        logger.info(f"Feature selection report saved to {report_path}")
        logger.info("Feature selection pipeline completed successfully!")
        
        return normalized_data, selected_features, feature_scores
        
    except Exception as e:
        logger.error(f"An error occurred in feature selection pipeline: {e}")
        raise e


if __name__ == "__main__":
    main()

