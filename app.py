import os
import sys
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
import threading
import time
from functools import wraps

# Import authentication system
sys.path.append(os.path.join(os.path.dirname(__file__), 'credentials'))
from auth import verify_user, update_last_login, change_password, get_user_info, get_password_history, list_users, create_user

app = Flask(__name__)
app.secret_key = 'concrete-strength-prediction-key-2025'

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin privileges required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

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

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            if verify_user(username, password):
                session['user'] = username
                session['role'] = get_user_info(username).get('role', 'user')
                update_last_login(username)
                flash(f'Welcome back, {username}!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password.', 'error')
        except Exception as e:
            flash(f'Login error: {str(e)}', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user."""
    username = session.get('user', 'User')
    session.clear()
    flash(f'Goodbye, {username}!', 'info')
    return redirect(url_for('login'))

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password_route():
    """Change password page."""
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('change_password.html')
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('change_password.html')
        
        try:
            success, message = change_password(session['user'], current_password, new_password)
            if success:
                flash(message, 'success')
                return redirect(url_for('index'))
            else:
                flash(message, 'error')
        except Exception as e:
            flash(f'Error changing password: {str(e)}', 'error')
    
    return render_template('change_password.html')

@app.route('/user_management')
@admin_required
def user_management():
    """User management page for admins."""
    users = list_users()
    return render_template('user_management.html', users=users)

@app.route('/create_user', methods=['POST'])
@admin_required
def create_user_route():
    """Create new user."""
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    
    try:
        success, message = create_user(username, password, role, session['user'])
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
    except Exception as e:
        flash(f'Error creating user: {str(e)}', 'error')
    
    return redirect(url_for('user_management'))

@app.route('/password_history')
@login_required
def password_history():
    """View password change history."""
    try:
        history = get_password_history(session['user'])
        return render_template('password_history.html', history=history)
    except Exception as e:
        flash(f'Error loading password history: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/')
@login_required
def index():
    """Main dashboard page."""
    config = load_config()
    results = get_latest_results()
    metrics = get_model_metrics()
    return render_template('index.html', config=config, results=results, 
                         status=pipeline_status, metrics=metrics)

@app.route('/run_pipeline', methods=['POST'])
@login_required
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
    
    pipeline_status['logs'].append(f"🔐 Pipeline started by user: {session['user']}")
    flash(f'Pipeline started successfully with {len(selected_steps)} steps!', 'success')
    return redirect(url_for('monitor'))

@app.route('/monitor')
@login_required
def monitor():
    """Pipeline monitoring page."""
    return render_template('monitor.html', status=pipeline_status)

@app.route('/status')
@login_required
def status():
    """API endpoint for pipeline status."""
    return jsonify(pipeline_status)

@app.route('/results')
@login_required
def results():
    """Results page."""
    results = get_latest_results()
    metrics = get_model_metrics()
    return render_template('results.html', results=results, metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
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
@login_required
def download_file(filename):
    """Download result files."""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('results'))

@app.route('/config', methods=['GET', 'POST'])
@admin_required
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
@login_required
def about():
    """About page."""
    return render_template('about.html')

# ============================================================================
# COMPREHENSIVE ML OPERATIONS ROUTES
# ============================================================================

@app.route('/ml_operations')
@login_required
def ml_operations():
    """ML Operations dashboard."""
    return render_template('ml_operations.html')

@app.route('/data_operations')
@login_required
def data_operations():
    """Data operations interface."""
    return render_template('data_operations.html')

@app.route('/training_center')
@login_required
def training_center():
    """Model training interface."""
    return render_template('training_center.html')

@app.route('/tuning_center')
@login_required
def tuning_center():
    """Hyperparameter tuning interface."""
    return render_template('tuning_center.html')

@app.route('/evaluation_center')
@login_required
def evaluation_center():
    """Model evaluation interface."""
    return render_template('evaluation_center.html')

# ============================================================================
# INDIVIDUAL ML OPERATION APIS
# ============================================================================

@app.route('/api/run_step/<step_name>', methods=['POST'])
@login_required
def api_run_step(step_name):
    """Run individual pipeline step."""
    try:
        data = request.get_json() or {}
        
        # Validate step name
        valid_steps = [
            'data_ingestion', 'feature_selection', 'data_processing', 
            'model_training', 'hyperparameter_tuning', 'final_model_training', 
            'final_model_evaluation'
        ]
        
        if step_name not in valid_steps:
            return jsonify({'success': False, 'message': f'Invalid step: {step_name}'})
        
        # Check if pipeline is already running
        if pipeline_status['running']:
            return jsonify({'success': False, 'message': 'Pipeline is already running'})
        
        # Start step in background
        def run_single_step():
            global pipeline_status
            pipeline_status['running'] = True
            pipeline_status['current_step'] = step_name.replace('_', ' ').title()
            pipeline_status['logs'].append(f"🚀 Starting {step_name.replace('_', ' ').title()}...")
            pipeline_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                config = load_config()
                script_path = f"pipeline/{step_name}.py"
                
                # Add custom parameters if provided
                cmd = [config.get('python_path', 'python'), script_path]
                if data.get('params'):
                    for key, value in data['params'].items():
                        cmd.extend([f"--{key}", str(value)])
                
                pipeline_status['logs'].append(f"📋 Command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
                
                if result.stdout:
                    pipeline_status['logs'].append(f"📊 Output: {result.stdout}")
                
                pipeline_status['logs'].append(f"✅ {step_name.replace('_', ' ').title()} completed successfully!")
                pipeline_status['completed'] = True
                
            except subprocess.CalledProcessError as e:
                error_msg = f"❌ {step_name.replace('_', ' ').title()} failed: {e.stderr if e.stderr else str(e)}"
                pipeline_status['logs'].append(error_msg)
                pipeline_status['error'] = error_msg
            except Exception as e:
                error_msg = f"❌ Error running {step_name.replace('_', ' ').title()}: {str(e)}"
                pipeline_status['logs'].append(error_msg)
                pipeline_status['error'] = error_msg
            finally:
                pipeline_status['running'] = False
                pipeline_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        thread = threading.Thread(target=run_single_step)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': f'{step_name.replace("_", " ").title()} started'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/train_models', methods=['POST'])
@login_required
def api_train_models():
    """Train specific models."""
    try:
        data = request.get_json()
        selected_models = data.get('models', [])
        
        if not selected_models:
            return jsonify({'success': False, 'message': 'No models selected'})
        
        # Start training in background
        def train_selected_models():
            global pipeline_status
            pipeline_status['running'] = True
            pipeline_status['current_step'] = 'Model Training'
            pipeline_status['logs'].append(f"🤖 Training models: {', '.join(selected_models)}")
            pipeline_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                config = load_config()
                
                # Create custom training parameters
                training_params = {
                    'models': selected_models,
                    'test_size': data.get('test_size', 0.2),
                    'random_state': data.get('random_state', 42),
                    'cv_folds': data.get('cv_folds', 5),
                    'n_jobs': data.get('n_jobs', -1)
                }
                
                # Save temporary training config
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(training_params, f)
                    temp_config_path = f.name
                
                # Run training with custom config
                cmd = [
                    config.get('python_path', 'python'), 
                    'pipeline/model_training.py',
                    '--config', temp_config_path
                ]
                
                pipeline_status['logs'].append(f"📋 Running: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
                
                if result.stdout:
                    pipeline_status['logs'].append(f"📊 Training Output:\n{result.stdout}")
                
                pipeline_status['logs'].append("✅ Model training completed successfully!")
                pipeline_status['completed'] = True
                
                # Clean up temp file
                os.unlink(temp_config_path)
                
            except Exception as e:
                error_msg = f"❌ Training failed: {str(e)}"
                pipeline_status['logs'].append(error_msg)
                pipeline_status['error'] = error_msg
            finally:
                pipeline_status['running'] = False
                pipeline_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        thread = threading.Thread(target=train_selected_models)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Model training started'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/hyperparameter_tune', methods=['POST'])
@login_required
def api_hyperparameter_tune():
    """Start hyperparameter tuning."""
    try:
        data = request.get_json()
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({'success': False, 'message': 'No model selected'})
        
        # Start tuning in background
        def tune_hyperparameters():
            global pipeline_status
            pipeline_status['running'] = True
            pipeline_status['current_step'] = f'{model_name} Hyperparameter Tuning'
            pipeline_status['logs'].append(f"⚙️ Starting hyperparameter tuning for {model_name}...")
            pipeline_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                config = load_config()
                
                # Create tuning parameters
                tuning_params = {
                    'model': model_name,
                    'n_iter': data.get('iterations', 500),
                    'cv_folds': data.get('cv_folds', 5),
                    'scoring': data.get('scoring', 'r2'),
                    'n_jobs': data.get('n_jobs', -1),
                    'random_state': data.get('random_state', 42)
                }
                
                # Save temporary tuning config
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(tuning_params, f)
                    temp_config_path = f.name
                
                # Run hyperparameter tuning
                cmd = [
                    config.get('python_path', 'python'), 
                    'pipeline/hyper_parameter_tuning.py',
                    '--config', temp_config_path
                ]
                
                pipeline_status['logs'].append(f"📋 Running: {' '.join(cmd)}")
                pipeline_status['logs'].append(f"🔧 Tuning {tuning_params['n_iter']} iterations...")
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2 hour timeout
                
                if result.stdout:
                    pipeline_status['logs'].append(f"📊 Tuning Output:\n{result.stdout}")
                
                pipeline_status['logs'].append(f"✅ Hyperparameter tuning for {model_name} completed!")
                pipeline_status['completed'] = True
                
                # Clean up temp file
                os.unlink(temp_config_path)
                
            except Exception as e:
                error_msg = f"❌ Hyperparameter tuning failed: {str(e)}"
                pipeline_status['logs'].append(error_msg)
                pipeline_status['error'] = error_msg
            finally:
                pipeline_status['running'] = False
                pipeline_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        thread = threading.Thread(target=tune_hyperparameters)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': f'Hyperparameter tuning started for {model_name}'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/evaluate_model', methods=['POST'])
@login_required
def api_evaluate_model():
    """Evaluate trained model."""
    try:
        data = request.get_json()
        
        # Start evaluation in background
        def evaluate_model():
            global pipeline_status
            pipeline_status['running'] = True
            pipeline_status['current_step'] = 'Model Evaluation'
            pipeline_status['logs'].append("📊 Starting comprehensive model evaluation...")
            pipeline_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                config = load_config()
                
                # Create evaluation parameters
                eval_params = {
                    'generate_plots': data.get('generate_plots', True),
                    'detailed_analysis': data.get('detailed_analysis', True),
                    'cross_validate': data.get('cross_validate', True),
                    'feature_importance': data.get('feature_importance', True)
                }
                
                # Save temporary evaluation config
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(eval_params, f)
                    temp_config_path = f.name
                
                # Run evaluation
                cmd = [
                    config.get('python_path', 'python'), 
                    'pipeline/final_model_evaluation.py',
                    '--config', temp_config_path
                ]
                
                pipeline_status['logs'].append(f"📋 Running: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)  # 30 min timeout
                
                if result.stdout:
                    pipeline_status['logs'].append(f"📊 Evaluation Output:\n{result.stdout}")
                
                pipeline_status['logs'].append("✅ Model evaluation completed successfully!")
                pipeline_status['completed'] = True
                
                # Clean up temp file
                os.unlink(temp_config_path)
                
            except Exception as e:
                error_msg = f"❌ Model evaluation failed: {str(e)}"
                pipeline_status['logs'].append(error_msg)
                pipeline_status['error'] = error_msg
            finally:
                pipeline_status['running'] = False
                pipeline_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        thread = threading.Thread(target=evaluate_model)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Model evaluation started'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_results')
@login_required
def api_get_results():
    """Get all available results."""
    try:
        results = {}
        results_dir = 'results'
        
        if os.path.exists(results_dir):
            for subdir in os.listdir(results_dir):
                subdir_path = os.path.join(results_dir, subdir)
                if os.path.isdir(subdir_path):
                    results[subdir] = []
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        if os.path.isfile(file_path):
                            file_info = {
                                'name': file,
                                'path': file_path,
                                'size': os.path.getsize(file_path),
                                'modified': os.path.getmtime(file_path)
                            }
                            results[subdir].append(file_info)
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system_info')
@login_required
def api_system_info():
    """Get system information."""
    try:
        import psutil
        import platform
        
        system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('.').percent,
            'pipeline_status': pipeline_status
        }
        
        return jsonify({'success': True, 'info': system_info})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'info': {}})

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
