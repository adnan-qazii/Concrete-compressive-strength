import os
import sys
import subprocess
import yaml

def load_config():
    """Load simple pipeline configuration."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        # Default config if file doesn't exist
        return {
            'steps': ['data_ingestion', 'feature_selection', 'data_processing', 'model_training', 'final_model_training', 'final_model_evaluation'],
            'python_path': 'python'
        }

def run_pipeline():
    """Run the ML pipeline."""
    config = load_config()
    
    # Pipeline steps mapping
    scripts = {
        'data_ingestion': 'pipeline/data_ingestion.py',
        'feature_selection': 'pipeline/feature_selection.py', 
        'data_processing': 'pipeline/data_processing.py',
        'model_training': 'pipeline/model_training.py',
        'hyperparameter_tuning': 'pipeline/hyper_parameter_tuning.py',
        'final_model_training': 'pipeline/final_model_training.py',
        'final_model_evaluation': 'pipeline/final_model_evaluation.py'
    }
    
    python_cmd = config.get('python_path', 'python')
    steps_to_run = config.get('steps', [])
    
    print("🚀 Starting ML Pipeline...")
    print(f"Steps to run: {', '.join(steps_to_run)}")
    print("-" * 50)
    
    for i, step in enumerate(steps_to_run, 1):
        if step not in scripts:
            print(f"❌ Unknown step: {step}")
            continue
            
        script = scripts[step]
        if not os.path.exists(script):
            print(f"❌ Script not found: {script}")
            continue
            
        print(f"[{i}/{len(steps_to_run)}] Running {step}...")
        
        try:
            result = subprocess.run([python_cmd, script], check=True)
            print(f"✅ {step} completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ {step} failed with code {e.returncode}")
            if config.get('stop_on_error', True):
                print("Stopping pipeline due to error")
                return False
        except Exception as e:
            print(f"❌ Error running {step}: {e}")
            return False
    
    print("🎉 Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
