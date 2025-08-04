"""
Quick runner script for individual pipeline steps.
Use this when you want to run specific steps without the full pipeline.
"""

import sys
import subprocess
import os
from pathlib import Path

def print_usage():
    """Print usage information."""
    print("Individual ML Pipeline Step Runner")
    print("=" * 40)
    print("Usage: python run_step.py <step_name>")
    print()
    print("Available steps:")
    print("  1. data_ingestion        - Load and preprocess raw data")
    print("  2. feature_selection     - Create engineered features")
    print("  3. data_processing       - Split data into train/test")
    print("  4. model_training        - Train basic models")
    print("  5. hyperparameter_tuning - Optimize model parameters")
    print("  6. final_model_training  - Train final model")
    print("  7. final_model_evaluation- Evaluate final model")
    print("  8. full_pipeline         - Run complete pipeline")
    print()
    print("Examples:")
    print("  python run_step.py data_ingestion")
    print("  python run_step.py model_training")
    print("  python run_step.py full_pipeline")

def run_step(step_name):
    """Run a specific pipeline step."""
    
    # Step mapping
    steps = {
        'data_ingestion': 'pipeline/data_ingestion.py',
        'feature_selection': 'pipeline/feature_selection.py',
        'data_processing': 'pipeline/data_processing.py',
        'model_training': 'pipeline/model_training.py',
        'hyperparameter_tuning': 'pipeline/hyper_parameter_tuning.py',
        'final_model_training': 'pipeline/final_model_training.py',
        'final_model_evaluation': 'pipeline/final_model_evaluation.py',
        'full_pipeline': 'run_ml_pipeline.py'
    }
    
    if step_name not in steps:
        print(f"Error: Unknown step '{step_name}'")
        print_usage()
        return False
    
    script_path = steps[step_name]
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False
    
    print(f"Running step: {step_name}")
    print(f"Script: {script_path}")
    print("-" * 40)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"\n✅ Step '{step_name}' completed successfully!")
        else:
            print(f"\n❌ Step '{step_name}' failed with return code {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"\n❌ Error running step '{step_name}': {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    step_name = sys.argv[1].lower()
    
    # Handle aliases
    aliases = {
        '1': 'data_ingestion',
        '2': 'feature_selection',
        '3': 'data_processing',
        '4': 'model_training',
        '5': 'hyperparameter_tuning',
        '6': 'final_model_training',
        '7': 'final_model_evaluation',
        '8': 'full_pipeline',
        'all': 'full_pipeline',
        'complete': 'full_pipeline'
    }
    
    if step_name in aliases:
        step_name = aliases[step_name]
    
    success = run_step(step_name)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
