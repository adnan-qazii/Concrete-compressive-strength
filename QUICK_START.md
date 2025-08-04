# 🔧 ML Pipeline Quick Start Guide

## Quick Commands

### 1. Setup Environment
```bash
# Check setup and install dependencies
python setup.py

# Install packages only
python setup.py install_packages

# Check if packages are installed
python setup.py check_packages
```

### 2. Run Pipeline
```bash
# Run complete pipeline
python run_ml_pipeline.py

# Run with custom config
python run_ml_pipeline.py custom_config.yaml
```

### 3. Run Individual Steps
```bash
# Run specific step
python run_step.py data_ingestion
python run_step.py model_training
python run_step.py full_pipeline

# Available steps:
# 1. data_ingestion
# 2. feature_selection  
# 3. data_processing
# 4. model_training
# 5. hyperparameter_tuning
# 6. final_model_training
# 7. final_model_evaluation
```

### 4. Configuration
Edit `ml_pipeline_config.yaml` to customize:
- Which steps to run
- Execution settings
- Resource management
- Output preferences

### 5. Results
- Pipeline results: `results/pipeline_run_YYYYMMDD_HHMMSS/`
- Logs: `logs/ml_pipeline_execution_YYYYMMDD_HHMMSS.log`
- Model files: `results/final_model/`
- Evaluation reports: `results/final_results/`

## File Structure
```
📁 Concrete-compressive-strength/
├── 🐍 run_ml_pipeline.py         # Main pipeline executor
├── 🐍 run_step.py                # Individual step runner
├── 🐍 setup.py                   # Environment setup
├── ⚙️ ml_pipeline_config.yaml    # Master configuration
├── 📁 pipeline/                  # Pipeline scripts
├── 📁 data/                      # Data files
├── 📁 results/                   # Output results
└── 📁 logs/                      # Execution logs
```

## Getting Started
1. Run `python setup.py` to check environment
2. Edit `ml_pipeline_config.yaml` as needed
3. Run `python run_ml_pipeline.py`
4. Check results in `results/` directory

## Need Help?
- Check logs in `logs/` directory
- Run `python run_step.py` for individual step options
- Verify config files are present and valid
