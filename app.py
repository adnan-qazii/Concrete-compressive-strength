import os
import sys
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import threading
import time

app = Flask(__name__)
app.secret_key = 'concrete-strength-prediction-key-2025'

# Global variables for pipeline status
pipeline_status = {
    'running': False,
    'current_step': '',
    'progress': 0,
    'total_steps': 0,
    'logs': [],
    'completed': False,
    'error': None,
    'start_time': None,
    'end_time': None
}

def load_config():
    """Load pipeline configuration."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'steps': ['data_ingestion', 'feature_selection', 'data_processing', 
                     'model_training', 'final_model_training', 'final_model_evaluation'],
            'python_path': 'python',
            'stop_on_error': True
        }

def get_latest_results():
    """Get the latest results from the results directory."""
    results = {}
    
    # Get basic training results
    basic_results_dir = Path('results/basic_training')
    if basic_results_dir.exists():
        basic_files = list(basic_results_dir.glob('*.txt'))
        if basic_files:
            latest_basic = max(basic_files, key=lambda x: x.stat().st_mtime)
            results['basic_training'] = {
                'file': latest_basic,
                'timestamp': datetime.fromtimestamp(latest_basic.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # Get hyperparameter tuning results
    hp_results_dir = Path('results/hyperparameter_tuning')
    if hp_results_dir.exists():
        hp_files = list(hp_results_dir.glob('*.txt'))
        if hp_files:
            latest_hp = max(hp_files, key=lambda x: x.stat().st_mtime)
            results['hyperparameter'] = {
                'file': latest_hp,
                'timestamp': datetime.fromtimestamp(latest_hp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # Get final results
    final_results_dir = Path('results/final_results')
    if final_results_dir.exists():
        final_files = list(final_results_dir.glob('*.txt'))
        if final_files:
            latest_final = max(final_files, key=lambda x: x.stat().st_mtime)
            results['final'] = {
                'file': latest_final,
                'timestamp': datetime.fromtimestamp(latest_final.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return results

def get_model_metrics():
    """Extract model metrics from results."""
    metrics = {}
    try:
        final_results_dir = Path('results/final_results')
        if final_results_dir.exists():
            final_files = list(final_results_dir.glob('*.txt'))
            if final_files:
                latest_file = max(final_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract R² score
                if 'R² Score:' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if 'Test Set Performance:' in line:
                            break
                        if 'R² Score:' in line and 'Test' not in line:
                            score = line.split(':')[1].strip()
                            metrics['r2_score'] = score
                            break
                
                # Extract RMSE
                if 'RMSE:' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if 'RMSE:' in line and 'Test' not in line:
                            rmse = line.split(':')[1].strip()
                            metrics['rmse'] = rmse
                            break
                
                # Extract best model type
                if 'Model Type:' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if 'Model Type:' in line:
                            model_type = line.split(':')[1].strip()
                            metrics['model_type'] = model_type
                            break
    except Exception as e:
        print(f"Error extracting metrics: {e}")
    
    return metrics

def run_pipeline_background(steps_to_run):
    """Run pipeline in background thread."""
    global pipeline_status
    
    scripts = {
        'data_ingestion': 'pipeline/data_ingestion.py',
        'feature_selection': 'pipeline/feature_selection.py', 
        'data_processing': 'pipeline/data_processing.py',
        'model_training': 'pipeline/model_training.py',
        'hyperparameter_tuning': 'pipeline/hyper_parameter_tuning.py',
        'final_model_training': 'pipeline/final_model_training.py',
        'final_model_evaluation': 'pipeline/final_model_evaluation.py'
    }
    
    config = load_config()
    python_cmd = config.get('python_path', 'python')
    
    pipeline_status['running'] = True
    pipeline_status['total_steps'] = len(steps_to_run)
    pipeline_status['progress'] = 0
    pipeline_status['logs'] = []
    pipeline_status['completed'] = False
    pipeline_status['error'] = None
    pipeline_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pipeline_status['end_time'] = None
    
    for i, step in enumerate(steps_to_run, 1):
        pipeline_status['current_step'] = step
        pipeline_status['progress'] = i - 1
        
        if step not in scripts:
            pipeline_status['logs'].append(f"❌ Unknown step: {step}")
            continue
            
        script = scripts[step]
        if not os.path.exists(script):
            pipeline_status['logs'].append(f"❌ Script not found: {script}")
            continue
            
        pipeline_status['logs'].append(f"🚀 [{i}/{len(steps_to_run)}] Starting {step.replace('_', ' ').title()}...")
        
        try:
            result = subprocess.run([python_cmd, script], 
                                  capture_output=True, text=True, check=True, timeout=3600)
            pipeline_status['logs'].append(f"✅ {step.replace('_', ' ').title()} completed successfully!")
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ {step.replace('_', ' ').title()} failed: {e.stderr if e.stderr else str(e)}"
            pipeline_status['logs'].append(error_msg)
            pipeline_status['error'] = error_msg
            if config.get('stop_on_error', True):
                break
        except subprocess.TimeoutExpired:
            error_msg = f"⏰ {step.replace('_', ' ').title()} timed out (1 hour limit)"
            pipeline_status['logs'].append(error_msg)
            pipeline_status['error'] = error_msg
            break
        except Exception as e:
            error_msg = f"❌ Error running {step.replace('_', ' ').title()}: {str(e)}"
            pipeline_status['logs'].append(error_msg)
            pipeline_status['error'] = error_msg
            break
    
    pipeline_status['progress'] = len(steps_to_run)
    pipeline_status['running'] = False
    pipeline_status['completed'] = True
    pipeline_status['current_step'] = 'Completed'
    pipeline_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if not pipeline_status['error']:
        pipeline_status['logs'].append("🎉 Pipeline completed successfully! All steps finished.")

@app.route('/')
def index():
    """Main dashboard page."""
    config = load_config()
    results = get_latest_results()
    metrics = get_model_metrics()
    return render_template('index.html', config=config, results=results, 
                         status=pipeline_status, metrics=metrics)

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    """Start pipeline execution."""
    if pipeline_status['running']:
        flash('Pipeline is already running! Please wait for it to complete.', 'warning')
        return redirect(url_for('index'))
    
    selected_steps = request.form.getlist('steps')
    if not selected_steps:
        flash('Please select at least one step to run!', 'error')
        return redirect(url_for('index'))
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline_background, args=(selected_steps,))
    thread.daemon = True
    thread.start()
    
    flash(f'Pipeline started successfully with {len(selected_steps)} steps!', 'success')
    return redirect(url_for('monitor'))

@app.route('/monitor')
def monitor():
    """Pipeline monitoring page."""
    return render_template('monitor.html', status=pipeline_status)

@app.route('/status')
def status():
    """API endpoint for pipeline status."""
    return jsonify(pipeline_status)

@app.route('/results')
def results():
    """Results page."""
    results = get_latest_results()
    metrics = get_model_metrics()
    return render_template('results.html', results=results, metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page."""
    if request.method == 'POST':
        try:
            # Get form data
            features = {
                'cement': float(request.form['cement']),
                'blast_furnace_slag': float(request.form['blast_furnace_slag']),
                'fly_ash': float(request.form['fly_ash']),
                'water': float(request.form['water']),
                'superplasticizer': float(request.form['superplasticizer']),
                'coarse_aggregate': float(request.form['coarse_aggregate']),
                'fine_aggregate': float(request.form['fine_aggregate']),
                'age': int(request.form['age'])
            }
            
            # Simple prediction formula (replace with actual model loading in production)
            # This is a simplified approximation based on concrete engineering principles
            cement_water_ratio = features['cement'] / features['water'] if features['water'] > 0 else 0
            total_binder = features['cement'] + features['blast_furnace_slag'] + features['fly_ash']
            age_factor = min(features['age'] / 28, 2.0)  # Normalize age effect
            
            predicted_strength = (
                cement_water_ratio * 12.5 +  # Cement-water ratio is critical
                total_binder * 0.08 +        # Total binder content
                age_factor * 15 +            # Age factor
                features['superplasticizer'] * 2.5 +  # Superplasticizer effect
                10  # Base strength
            )
            
            # Add some realistic variation and bounds
            predicted_strength = max(10, min(80, predicted_strength))  # Realistic range
            
            # Calculate confidence based on typical ranges
            confidence = 85 if all([
                100 <= features['cement'] <= 500,
                150 <= features['water'] <= 250,
                800 <= features['coarse_aggregate'] <= 1200,
                600 <= features['fine_aggregate'] <= 900,
                1 <= features['age'] <= 365
            ]) else 70
            
            return render_template('predict.html', 
                                 features=features, 
                                 prediction=round(predicted_strength, 2),
                                 confidence=confidence)
        
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('predict.html')

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download result files."""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('results'))

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Configuration page."""
    if request.method == 'POST':
        try:
            # Update configuration
            new_config = {
                'steps': request.form.getlist('steps'),
                'python_path': request.form['python_path'],
                'stop_on_error': 'stop_on_error' in request.form
            }
            
            with open('config.yaml', 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            
            flash('Configuration updated successfully!', 'success')
        except Exception as e:
            flash(f'Error updating configuration: {str(e)}', 'error')
    
    current_config = load_config()
    all_steps = [
        'data_ingestion', 'feature_selection', 'data_processing', 
        'model_training', 'hyperparameter_tuning', 'final_model_training', 
        'final_model_evaluation'
    ]
    
    return render_template('config.html', config=current_config, all_steps=all_steps)

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    print("🚀 Starting Concrete Strength Prediction Web App...")
    print("🌟 Beautiful Dashboard will be available at: http://localhost:5000")
    print("📊 Features: Real-time monitoring, Interactive prediction, Results visualization")
    app.run(debug=True, host='0.0.0.0', port=5000)
