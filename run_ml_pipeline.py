import os
import sys
import subprocess
import time
import datetime
import logging
import yaml
import json
import traceback
from pathlib import Path
import pandas as pd

class MLPipelineExecutor:
    """
    Master pipeline executor for concrete strength prediction ML pipeline.
    Runs all pipeline steps in sequence or selectively based on configuration.
    """
    
    def __init__(self, config_path='ml_pipeline_config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.pipeline_start_time = None
        self.step_results = {}
        self.step_timings = {}
        self.failed_steps = []
        
        # Setup logging
        self.setup_logging()
        
        # Create results directory
        self.create_results_directory()
        
    def load_config(self):
        """Load pipeline configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['ml_pipeline']
        except Exception as e:
            print(f"Error loading pipeline config: {e}")
            # Return minimal default config
            return {
                'execution': {'steps': {}, 'stop_on_error': True},
                'pipeline_settings': {'python_executable': 'python'},
                'output': {'create_pipeline_summary': True}
            }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('execution', {}).get('log_level', 'INFO')
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('ml_pipeline_executor')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # File handler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/ml_pipeline_execution_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"ML Pipeline Executor initialized - Log file: {log_file}")
    
    def create_results_directory(self):
        """Create main results directory for pipeline execution."""
        base_dir = self.config.get('output', {}).get('base_results_dir', 'results')
        
        if self.config.get('output', {}).get('create_timestamped_dir', True):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = os.path.join(base_dir, f'pipeline_run_{timestamp}')
        else:
            self.results_dir = base_dir
        
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f"Pipeline results directory: {self.results_dir}")
    
    def validate_environment(self):
        """Validate the execution environment."""
        self.logger.info("Validating execution environment...")
        
        # Check Python executable
        python_exe = self.config.get('pipeline_settings', {}).get('python_executable', 'python')
        try:
            result = subprocess.run([python_exe, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info(f"Python executable validated: {result.stdout.strip()}")
            else:
                self.logger.error(f"Python executable validation failed: {python_exe}")
                return False
        except Exception as e:
            self.logger.error(f"Error validating Python executable: {e}")
            return False
        
        # Check required packages if enabled
        if self.config.get('environment', {}).get('check_packages', False):
            required_packages = self.config.get('environment', {}).get('required_packages', [])
            for package in required_packages:
                try:
                    result = subprocess.run([python_exe, '-c', f'import {package}'], 
                                          capture_output=True, timeout=10)
                    if result.returncode != 0:
                        self.logger.error(f"Required package not found: {package}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error checking package {package}: {e}")
                    return False
            
            self.logger.info("All required packages validated")
        
        return True
    
    def check_dependencies(self, step_name):
        """Check if dependencies for a step are satisfied."""
        step_config = self.config.get('step_configs', {}).get(step_name, {})
        dependencies = step_config.get('depends_on', [])
        
        for dep in dependencies:
            if dep not in self.step_results or not self.step_results[dep]['success']:
                self.logger.error(f"Dependency not satisfied for {step_name}: {dep}")
                return False
        
        return True
    
    def execute_step(self, step_name, script_path):
        """Execute a single pipeline step."""
        self.logger.info(f"Starting step: {step_name}")
        step_start_time = time.time()
        
        # Check dependencies
        if not self.check_dependencies(step_name):
            self.logger.error(f"Dependencies not satisfied for {step_name}")
            self.step_results[step_name] = {
                'success': False,
                'error': 'Dependencies not satisfied',
                'execution_time': 0
            }
            return False
        
        try:
            # Get Python executable
            python_exe = self.config.get('pipeline_settings', {}).get('python_executable', 'python')
            
            # Prepare command
            cmd = [python_exe, script_path]
            
            # Execute the script
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=3600  # 1 hour timeout
            )
            
            execution_time = time.time() - step_start_time
            self.step_timings[step_name] = execution_time
            
            if result.returncode == 0:
                self.logger.info(f"Step {step_name} completed successfully in {execution_time:.2f} seconds")
                self.step_results[step_name] = {
                    'success': True,
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return True
            else:
                self.logger.error(f"Step {step_name} failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                self.step_results[step_name] = {
                    'success': False,
                    'error': result.stderr,
                    'execution_time': execution_time,
                    'return_code': result.returncode
                }
                return False
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - step_start_time
            self.logger.error(f"Step {step_name} timed out after {execution_time:.2f} seconds")
            self.step_results[step_name] = {
                'success': False,
                'error': 'Timeout',
                'execution_time': execution_time
            }
            return False
            
        except Exception as e:
            execution_time = time.time() - step_start_time
            self.logger.error(f"Error executing step {step_name}: {e}")
            self.step_results[step_name] = {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
            return False
    
    def run_pipeline(self):
        """Execute the complete ML pipeline."""
        self.logger.info("="*70)
        self.logger.info("STARTING ML PIPELINE EXECUTION")
        self.logger.info("="*70)
        
        self.pipeline_start_time = time.time()
        
        # Validate environment
        if not self.validate_environment():
            self.logger.error("Environment validation failed. Aborting pipeline.")
            return False
        
        # Define pipeline steps and their corresponding scripts
        pipeline_steps = {
            'data_ingestion': 'pipeline/data_ingestion.py',
            'feature_selection': 'pipeline/feature_selection.py',
            'data_processing': 'pipeline/data_processing.py',
            'model_training': 'pipeline/model_training.py',
            'hyperparameter_tuning': 'pipeline/hyper_parameter_tuning.py',
            'final_model_training': 'pipeline/final_model_training.py',
            'final_model_evaluation': 'pipeline/final_model_evaluation.py'
        }
        
        # Get execution configuration
        execution_steps = self.config.get('execution', {}).get('steps', {})
        stop_on_error = self.config.get('execution', {}).get('stop_on_error', True)
        
        # Execute steps
        total_steps = sum(1 for step, enabled in execution_steps.items() if enabled)
        completed_steps = 0
        
        for step_name, script_path in pipeline_steps.items():
            # Check if step is enabled
            if not execution_steps.get(step_name, False):
                self.logger.info(f"Skipping step: {step_name} (disabled in config)")
                continue
            
            self.logger.info(f"Progress: {completed_steps}/{total_steps} steps completed")
            
            # Execute step
            success = self.execute_step(step_name, script_path)
            
            if success:
                completed_steps += 1
            else:
                self.failed_steps.append(step_name)
                if stop_on_error:
                    self.logger.error(f"Pipeline stopped due to failure in step: {step_name}")
                    break
                else:
                    self.logger.warning(f"Step {step_name} failed, but continuing due to configuration")
        
        # Calculate total execution time
        total_execution_time = time.time() - self.pipeline_start_time
        
        # Generate summary
        self.generate_pipeline_summary(total_execution_time, completed_steps, total_steps)
        
        # Determine overall success
        pipeline_success = len(self.failed_steps) == 0
        
        self.logger.info("="*70)
        if pipeline_success:
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        else:
            self.logger.error(f"PIPELINE EXECUTION COMPLETED WITH {len(self.failed_steps)} FAILED STEPS")
        self.logger.info("="*70)
        
        return pipeline_success
    
    def generate_pipeline_summary(self, total_time, completed_steps, total_steps):
        """Generate a comprehensive pipeline execution summary."""
        if not self.config.get('output', {}).get('create_pipeline_summary', True):
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.results_dir, f'pipeline_execution_summary_{timestamp}.txt')
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("ML PIPELINE EXECUTION SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Execution Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n")
                f.write(f"Steps Completed: {completed_steps}/{total_steps}\n")
                f.write(f"Configuration File: {self.config_path}\n\n")
                
                # Step-by-step results
                f.write("STEP EXECUTION RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                for step_name, result in self.step_results.items():
                    status = "SUCCESS" if result['success'] else "FAILED"
                    f.write(f"{step_name:<25}: {status:<8} ({result['execution_time']:.2f}s)\n")
                    if not result['success']:
                        f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                
                # Timing breakdown
                if self.step_timings:
                    f.write(f"\nTIMING BREAKDOWN:\n")
                    f.write("-" * 40 + "\n")
                    for step_name, timing in self.step_timings.items():
                        percentage = (timing / total_time) * 100
                        f.write(f"{step_name:<25}: {timing:>8.2f}s ({percentage:>5.1f}%)\n")
                
                # Failed steps
                if self.failed_steps:
                    f.write(f"\nFAILED STEPS:\n")
                    f.write("-" * 40 + "\n")
                    for step in self.failed_steps:
                        f.write(f"- {step}\n")
                
                # Configuration summary
                f.write(f"\nCONFIGURATION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                execution_config = self.config.get('execution', {})
                f.write(f"Stop on Error: {execution_config.get('stop_on_error', True)}\n")
                f.write(f"Log Level: {execution_config.get('log_level', 'INFO')}\n")
                
                enabled_steps = [step for step, enabled in execution_config.get('steps', {}).items() if enabled]
                f.write(f"Enabled Steps: {', '.join(enabled_steps)}\n")
                
                # Performance metrics (if available)
                if self.config.get('output', {}).get('include_performance_metrics', True):
                    self.add_performance_metrics_to_summary(f)
                
                f.write(f"\nSummary generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"Pipeline summary saved: {summary_file}")
            
            # Also save as JSON for programmatic access
            json_summary = {
                'execution_date': datetime.datetime.now().isoformat(),
                'total_execution_time': total_time,
                'steps_completed': completed_steps,
                'total_steps': total_steps,
                'step_results': self.step_results,
                'step_timings': self.step_timings,
                'failed_steps': self.failed_steps,
                'configuration': self.config
            }
            
            json_file = os.path.join(self.results_dir, f'pipeline_execution_summary_{timestamp}.json')
            with open(json_file, 'w') as f:
                json.dump(json_summary, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline summary: {e}")
    
    def add_performance_metrics_to_summary(self, file_handle):
        """Add performance metrics to the summary file if available."""
        try:
            # Look for final evaluation results
            final_results_dir = os.path.join('results', 'final_results')
            if os.path.exists(final_results_dir):
                # Find the most recent evaluation report
                report_files = [f for f in os.listdir(final_results_dir) if f.startswith('final_model_evaluation_report_')]
                if report_files:
                    latest_report = max(report_files)
                    report_path = os.path.join(final_results_dir, latest_report)
                    
                    file_handle.write(f"\nFINAL MODEL PERFORMANCE:\n")
                    file_handle.write("-" * 40 + "\n")
                    
                    # Extract key metrics from the report
                    try:
                        with open(report_path, 'r', encoding='utf-8') as report_f:
                            content = report_f.read()
                            
                        # Simple extraction of key metrics
                        if 'R² Score:' in content:
                            lines = content.split('\n')
                            for line in lines:
                                if 'R² Score:' in line and 'Test' not in line:
                                    file_handle.write(f"Model R² Score: {line.split(':')[1].strip()}\n")
                                    break
                        
                        if 'RMSE:' in content:
                            lines = content.split('\n')
                            for line in lines:
                                if 'RMSE:' in line and 'Test' not in line:
                                    file_handle.write(f"Model RMSE: {line.split(':')[1].strip()}\n")
                                    break
                        
                        file_handle.write(f"Detailed report: {report_path}\n")
                        
                    except Exception as e:
                        file_handle.write(f"Error reading performance metrics: {e}\n")
        
        except Exception as e:
            self.logger.warning(f"Could not add performance metrics to summary: {e}")


def main():
    """Main function to execute the ML pipeline."""
    print("ML Pipeline Executor for Concrete Strength Prediction")
    print("=" * 60)
    
    # Check for config file argument
    config_file = 'ml_pipeline_config.yaml'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        print("Please ensure the configuration file exists or provide the correct path.")
        sys.exit(1)
    
    try:
        # Create and run pipeline executor
        executor = MLPipelineExecutor(config_file)
        success = executor.run_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error in pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
