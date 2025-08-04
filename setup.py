"""
Setup script for ML Pipeline environment.
Checks dependencies and installs missing packages.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required for this pipeline")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_virtual_environment():
    """Check if running in virtual environment."""
    print("\nChecking virtual environment...")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✅ Running in virtual environment")
        print(f"Environment path: {sys.prefix}")
    else:
        print("⚠️  Not running in virtual environment")
        print("It's recommended to use a virtual environment")
    
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    print("\nInstalling required packages...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt file not found")
        return False
    
    try:
        # Install packages
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully")
            return True
        else:
            print("❌ Error installing packages:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during package installation: {e}")
        return False

def check_required_packages():
    """Check if all required packages are available."""
    print("\nChecking required packages...")
    
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'PyYAML',
        'scipy',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All required packages are available")
        return True

def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")
    
    required_files = [
        'data/concrete_data.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All required data files are available")
        return True

def check_pipeline_files():
    """Check if all pipeline scripts exist."""
    print("\nChecking pipeline scripts...")
    
    pipeline_files = [
        'pipeline/data_ingestion.py',
        'pipeline/feature_selection.py',
        'pipeline/data_processing.py',
        'pipeline/model_training.py',
        'pipeline/hyper_parameter_tuning.py',
        'pipeline/final_model_training.py',
        'pipeline/final_model_evaluation.py'
    ]
    
    missing_files = []
    
    for file_path in pipeline_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing scripts: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All pipeline scripts are available")
        return True

def check_config_files():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")
    
    config_files = [
        'ml_pipeline_config.yaml',
        'data_processing_config.yaml',
        'hyperparameter_tuning_config.yaml',
        'final_model_training_config.yaml'
    ]
    
    missing_files = []
    
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing config files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All configuration files are available")
        return True

def create_directories():
    """Create necessary directories."""
    print("\nCreating necessary directories...")
    
    directories = [
        'logs',
        'results',
        'data/1_preprocessed',
        'data/2_feature_selection',
        'data/3_processed',
        'results/basic_training',
        'results/hyperparameter_tuning',
        'results/final_model',
        'results/final_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}")
    
    print("\n✅ All directories created/verified")
    return True

def run_setup():
    """Run complete setup check."""
    print("ML Pipeline Setup and Environment Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Data Files", check_data_files),
        ("Pipeline Scripts", check_pipeline_files),
        ("Configuration Files", check_config_files),
        ("Required Packages", check_required_packages),
        ("Directory Structure", create_directories)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    failed_checks = []
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<25}: {status}")
        if not passed:
            failed_checks.append(check_name)
    
    if failed_checks:
        print(f"\n❌ Setup incomplete. Failed checks: {', '.join(failed_checks)}")
        print("\nRecommendations:")
        
        if "Required Packages" in failed_checks:
            print("- Run: python setup.py install_packages")
        if "Data Files" in failed_checks:
            print("- Ensure concrete_data.csv is in the data/ directory")
        if "Pipeline Scripts" in failed_checks:
            print("- Ensure all pipeline scripts are present")
        if "Configuration Files" in failed_checks:
            print("- Ensure all YAML configuration files are present")
        
        return False
    else:
        print("\n✅ Setup complete! Pipeline is ready to run.")
        print("\nNext steps:")
        print("- Run individual step: python run_step.py <step_name>")
        print("- Run full pipeline: python run_ml_pipeline.py")
        return True

def main():
    """Main function with command line options."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'install_packages':
            success = install_requirements()
            sys.exit(0 if success else 1)
        elif command == 'check_packages':
            success = check_required_packages()
            sys.exit(0 if success else 1)
        elif command == 'create_dirs':
            success = create_directories()
            sys.exit(0 if success else 1)
        else:
            print("Unknown command. Available commands:")
            print("- install_packages: Install packages from requirements.txt")
            print("- check_packages: Check if required packages are installed")
            print("- create_dirs: Create necessary directories")
            sys.exit(1)
    else:
        # Run full setup check
        success = run_setup()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
