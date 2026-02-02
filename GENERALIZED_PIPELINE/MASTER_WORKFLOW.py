#!/usr/bin/env python3
"""
GENERALIZED DSSAT DUERNAST Analysis Workflow

Purpose: Complete end-to-end workflow for Duernast experiments (2015, 2017, and future)
         Supports both Spring and Winter Wheat with automatic parameterization.

Usage:
    python MASTER_WORKFLOW.py 2015    # Run 2015 experiment
    python MASTER_WORKFLOW.py 2017    # Run 2017 experiment
    python MASTER_WORKFLOW.py --all   # Run all available experiments
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

# Import configuration
from config import get_config, list_available_experiments, ExperimentConfig


class GeneralizedDuernastWorkflowManager:
    """Generalized workflow manager for Duernast experiments"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize workflow manager with experiment configuration
        
        Args:
            config: ExperimentConfig object containing all experiment parameters
        """
        self.config = config
        self.start_time = datetime.now()
        self.workflow_steps = []
        self.results = {}
        self.errors = []
        
    def log_step(self, step_name, status, details="", execution_time=0):
        """Log workflow step results"""
        step_info = {
            'step': step_name,
            'status': status,
            'details': details,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        self.workflow_steps.append(step_info)
        
    def print_header(self, title, level=1):
        """Print formatted headers"""
        if level == 1:
            print(f"\n{'='*90}")
            print(f"{title.center(90)}")
            print('='*90)
        elif level == 2:
            print(f"\n{'-'*70}")
            print(f"{title}")
            print('-'*70)
        else:
            print(f"\n{title}")
            print('~' * len(title))
    
    def get_experiment_path(self):
        """Get the path to the experiment directory, handling both project root and GENERALIZED_PIPELINE locations"""
        experiment_path = Path(self.config.experiment_dir)
        if not experiment_path.exists():
            # Try parent directory (if running from GENERALIZED_PIPELINE)
            experiment_path = Path('..') / self.config.experiment_dir
        # Resolve to absolute path to avoid issues with relative paths
        return experiment_path.resolve()
    
    def check_prerequisites(self):
        """Check all prerequisites for the workflow"""
        
        self.print_header("STEP 1: PREREQUISITES CHECK", 1)
        
        start_time = time.time()
        
        # Check if we're in the right directory structure
        # The script can be run from project root or GENERALIZED_PIPELINE directory
        experiment_path = self.get_experiment_path()
        if not experiment_path.exists():
            error_msg = f"Experiment directory not found: {self.config.experiment_dir}"
            self.log_step("Prerequisites", "FAILED", error_msg)
            print(f"[ERROR] {error_msg}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Tried: {experiment_path.absolute()}")
            return False
        
        # Change to experiment directory for the rest of the workflow
        original_dir = os.getcwd()
        os.chdir(experiment_path)
        
        try:
            if not Path('input').exists():
                error_msg = f"Input folder missing in {self.config.experiment_dir}!"
                self.log_step("Prerequisites", "FAILED", error_msg)
                print(f"[ERROR] {error_msg}")
                return False
            
            print(f"[OK] Working directory: {os.getcwd()}")
            print(f"[OK] Input folder exists: input/")
            
            # Ensure output directory exists
            output_dir = Path('output')
            if not output_dir.exists():
                print(f"[INFO] Creating output directory...")
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build required files dictionary dynamically from config
            required_files = {
                f'input/{self.config.file_prefix}.WHX': 'Experiment definition (N-Wheat model)',
                f'input/{self.config.file_prefix}.WTH': 'Weather data',
                f'input/orignal data/{self.config.file_prefix}.WHT': 'Observed field data (yield+grain+N)',
                f'input/DE.SOL': 'Soil profile',
                f'Genotype/WHAPS048.CUL': 'N-Wheat cultivar parameters',
                f'output/Summary.OUT': 'Main simulation results',
                f'output/PlantGro.OUT': 'Plant growth time series data',
                f'output/Weather.OUT': 'Weather data processing results',
                f'output/PlantN.OUT': 'Nitrogen dynamics data',
                f'output/SoilNi.OUT': 'Soil nitrogen dynamics',
                f'output/SoilWat.OUT': 'Soil water balance',
                f'output/OVERVIEW.OUT': 'Comprehensive treatment overview'
            }
            
            print("\nChecking Required Files:")
            missing_files = []
            missing_outputs = []
            total_size = 0
            
            for filename, description in required_files.items():
                if Path(filename).exists():
                    size = Path(filename).stat().st_size
                    total_size += size
                    print(f"  [OK] {filename:<45} ({size:>10,} bytes) - {description}")
                else:
                    print(f"  [MISSING] {filename:<45} {'':>19} - {description}")
                    if filename.startswith('output/'):
                        missing_outputs.append(filename)
                    else:
                        missing_files.append(filename)
            
            # Check for missing input files (critical)
            if missing_files:
                error_msg = f"Missing required input files: {', '.join(missing_files)}"
                self.log_step("Prerequisites", "FAILED", error_msg)
                print(f"\n[ERROR] {error_msg}")
                print("Check that input files exist in input/ and Genotype/ folders")
                return False
            
            # Missing output files is expected (will be generated)
            if missing_outputs:
                print(f"\n[INFO] Output files will be generated by simulation")
                print(f"  Missing: {len(missing_outputs)} output files")
            else:
                print(f"\n[OK] All required files present ({total_size:,} bytes = {total_size/1024/1024:.1f} MB)")
            
            # Check Python dependencies
            print("\nChecking Python Dependencies:")
            required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
            
            for package in required_packages:
                try:
                    __import__(package)
                    print(f"  [OK] {package}")
                except ImportError:
                    error_msg = f"Missing Python package: {package}"
                    self.log_step("Prerequisites", "FAILED", error_msg)
                    print(f"  [MISSING] {package}")
                    print(f"\n[ERROR] {error_msg}")
                    print(f"Install with: pip install {package}")
                    return False
            
            # Check generalized visualization script
            print("\nChecking Generalized Visualization Script:")
            generalized_script = experiment_path.parent / 'GENERALIZED_PIPELINE' / 'create_duernast_visualizations.py'
            if generalized_script.exists():
                print(f"  [OK] {generalized_script}")
            else:
                print(f"  [WARNING] {generalized_script} not found")
            
            execution_time = time.time() - start_time
            self.log_step("Prerequisites", "SUCCESS", "All prerequisites satisfied", execution_time)
            
            print(f"\n[SUCCESS] Prerequisites check completed! ({execution_time:.2f}s)")
            return True
            
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def run_dssat_simulation(self):
        """Run DSSAT N-Wheat simulation with outputs directed to output folder"""
        
        self.print_header("STEP 2: DSSAT N-WHEAT SIMULATION", 1)
        
        start_time = time.time()
        
        # Change to experiment directory
        original_dir = os.getcwd()
        experiment_path = self.get_experiment_path()
        os.chdir(str(experiment_path))
        
        try:
            # Ensure output directory exists
            output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # DELETE ALL PREVIOUS OUTPUT FILES to ensure fresh simulation
            print("Cleaning previous simulation outputs...")
            import glob
            
            # List of DSSAT output file patterns to delete
            output_patterns = [
                '*.OUT',           # All DSSAT output files
                'DSSAT48.INP',     # DSSAT input file (regenerated)
                'DSSAT48.INH',     # DSSAT header file (regenerated)
                'Evaluate.OUT',    # Evaluation output
                '*.WTH',           # Weather files (will be recopied)
                '*.WHX',           # Experiment files (will be recopied)
                '*.WHA',           # Observed average files (will be recopied)
                '*.WHT',           # Observed time-series files (will be recopied)
                '*.SOL',           # Soil files (will be recopied)
            ]
            
            deleted_count = 0
            for pattern in output_patterns:
                for file_path in output_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_count += 1
            
            # Delete visualization outputs to force regeneration
            viz_patterns = [
                f'{self.config.output_prefix}_comprehensive_analysis.png',
                f'{self.config.output_prefix}_comprehensive_analysis.pdf',
            ]
            for pattern in viz_patterns:
                viz_file = output_dir / pattern
                if viz_file.exists():
                    viz_file.unlink()
                    deleted_count += 1
            
            # Delete Model_analysis directory if it exists
            model_analysis_dir = output_dir / 'Model_analysis'
            if model_analysis_dir.exists():
                import shutil
                shutil.rmtree(model_analysis_dir)
                deleted_count += 1
                print(f"  [OK] Deleted Model_analysis directory")
            
            if deleted_count > 0:
                print(f"  [OK] Deleted {deleted_count} previous output file(s)")
            else:
                print(f"  [OK] No previous outputs to clean")
            
            # Copy necessary files to output directory
            # Executable and config files from main folder or DSSAT48
            main_files = ['DSCSM048.EXE', 'DSCSM048.CTR', 'DATA.CDE', 'DETAIL.CDE']
            # Input files from input folder (dynamically built from config)
            input_files = [
                f'{self.config.file_prefix}.WHX',
                f'{self.config.file_prefix}.WTH',
                f'{self.config.file_prefix}.WHA',
                'DE.SOL'
            ]
            
            print("Preparing simulation environment...")
            import shutil
            
            # Copy DSSAT executable and config files (check multiple locations)
            for filename in main_files:
                src = Path(filename)
                if not src.exists():
                    # Try parent DSSAT48 folder
                    src = Path('../DSSAT48') / filename
                
                if src.exists():
                    dst = output_dir / filename
                    shutil.copy2(src, dst)
                    print(f"  [OK] Copied {filename}")
                else:
                    print(f"  [WARNING] {filename} not found")
            
            # Copy input files from input folder
            for filename in input_files:
                src = Path('input') / filename
                if src.exists():
                    dst = output_dir / filename
                    # Explicitly remove existing file to ensure fresh copy
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(src, dst)
                    print(f"  [OK] Copied {filename} (overwrote existing if present)")
                    
                    # For multi-year winter wheat simulations, also create additional weather file
                    if self.config.is_multi_year and filename == f'{self.config.file_prefix}.WTH':
                        if self.config.additional_weather_file:
                            # Copy the additional weather file (e.g., TUDU1601.WTH for 2017)
                            additional_src = Path('input') / self.config.additional_weather_file
                            if additional_src.exists():
                                additional_dst = output_dir / self.config.additional_weather_file
                                shutil.copy2(additional_src, additional_dst)
                                print(f"  [OK] Copied {self.config.additional_weather_file} (for multi-year simulation)")
                            else:
                                print(f"  [WARNING] {self.config.additional_weather_file} not found in input folder")
                else:
                    print(f"  [WARNING] {filename} not found in input folder")
            
            # Copy observed data file (WHT format with grain weight and nitrogen data)
            wht_src = Path('input/orignal data') / f'{self.config.file_prefix}.WHT'
            if wht_src.exists():
                wht_dst = output_dir / f'{self.config.file_prefix}.WHT'
                # Explicitly remove existing file to ensure fresh copy
                if wht_dst.exists():
                    wht_dst.unlink()
                shutil.copy2(wht_src, wht_dst)
                print(f"  [OK] Copied {self.config.file_prefix}.WHT (observed data: yield+grain wt+grain N)")
            else:
                print(f"  [WARNING] {self.config.file_prefix}.WHT not found")
            
            # Copy Genotype directory - ALWAYS replace to ensure fresh files
            src_genotype = Path('Genotype')
            dst_genotype = output_dir / 'Genotype'
            if src_genotype.exists():
                if dst_genotype.exists():
                    shutil.rmtree(dst_genotype)
                    print(f"  [OK] Removed previous Genotype directory")
                shutil.copytree(src_genotype, dst_genotype)
                
                # Verify key cultivar file was copied with latest timestamp
                cul_file = dst_genotype / 'WHAPS048.CUL'
                if cul_file.exists():
                    src_cul = src_genotype / 'WHAPS048.CUL'
                    src_mtime = src_cul.stat().st_mtime
                    dst_mtime = cul_file.stat().st_mtime
                    print(f"  [OK] Copied Genotype directory (N-Wheat cultivar parameters)")
                    print(f"  [VERIFY] Cultivar file timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dst_mtime))}")
                else:
                    print(f"  [WARNING] WHAPS048.CUL not found in copied Genotype directory")
            else:
                print(f"  [ERROR] Genotype directory not found!")
            
            # Verify key input files are fresh (show modification times)
            print("\nVerifying input files are up-to-date...")
            key_files_to_check = [
                (output_dir / f'{self.config.file_prefix}.WHX', 'input'),
                (output_dir / 'Genotype/WHAPS048.CUL', 'Genotype'),
            ]
            for file_path, source_dir in key_files_to_check:
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  [VERIFY] {file_path.name}: {mtime_str}")
            
            # Run DSSAT from output directory
            print("\nRunning DSSAT N-Wheat simulation...")
            try:
                os.chdir('output')
                experiment_file = f'{self.config.file_prefix}.WHX'
                result = subprocess.run(['DSCSM048.EXE', 'A', experiment_file], 
                                      capture_output=True, text=True, timeout=300)
                os.chdir(experiment_path)
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    # Check if key output files were created
                    key_outputs = ['Summary.OUT', 'OVERVIEW.OUT', 'PlantGro.OUT']
                    outputs_created = all((output_dir / f).exists() for f in key_outputs)
                    
                    if outputs_created:
                        print(f"[SUCCESS] DSSAT N-Wheat simulation completed ({execution_time:.2f}s)")
                        print(f"  Output files saved in: output/")
                        self.log_step("DSSAT Simulation", "SUCCESS", "Simulation completed", execution_time)
                        return True
                    else:
                        print(f"[WARNING] Simulation ran but some output files missing")
                        self.log_step("DSSAT Simulation", "WARNING", "Some outputs missing", execution_time)
                        return False
                else:
                    print(f"[WARNING] DSSAT returned code {result.returncode}")
                    self.log_step("DSSAT Simulation", "WARNING", f"Return code {result.returncode}", execution_time)
                    return False
                    
            except Exception as e:
                os.chdir(experiment_path)
                print(f"[ERROR] Simulation failed: {e}")
                self.log_step("DSSAT Simulation", "FAILED", str(e))
                return False
                
        finally:
            os.chdir(original_dir)
    
    def run_visualization(self):
        """Run visualization generation"""
        
        self.print_header("STEP 3: VISUALIZATION GENERATION", 1)
        
        # Change to experiment directory
        original_dir = os.getcwd()
        experiment_path = self.get_experiment_path()
        os.chdir(str(experiment_path))
        
        try:
            # Change to output directory for visualizations
            os.chdir('output')
            
            # Use the generalized visualization script from GENERALIZED_PIPELINE
            # The script reads configuration from environment variables
            script = str(experiment_path.parent / 'GENERALIZED_PIPELINE' / 'create_duernast_visualizations.py')
            
            description = 'Comprehensive 12-Panel Visualization'
            expected_output = f'{self.config.output_prefix}_comprehensive_analysis.png'
            
            if not Path(script).exists():
                print(f"[ERROR] {description} - script not found at {script}")
                return False
            
            print(f"[INFO] Using generalized visualization script: {script}")
            
            self.print_header(description, 2)
            
            start_time = time.time()
            
            try:
                # Pass config as environment variable or modify script to accept it
                # For now, we'll pass the year and normalization flag as environment variables
                env = os.environ.copy()
                env['DUERNAST_YEAR'] = str(self.config.year)
                env['DUERNAST_NORMALIZE_DAS'] = str(self.config.normalize_das)
                env['DUERNAST_FILE_PREFIX'] = self.config.file_prefix
                env['DUERNAST_OUTPUT_PREFIX'] = self.config.output_prefix
                
                result = subprocess.run([sys.executable, script], 
                                      capture_output=True, text=True, timeout=600, env=env)
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"[SUCCESS] {description} completed ({execution_time:.2f}s)")
                    
                    # Check outputs
                    if Path(expected_output).exists():
                        size = Path(expected_output).stat().st_size
                        print(f"  Generated: {expected_output} ({size:,} bytes)")
                        
                        pdf_version = expected_output.replace('.png', '.pdf')
                        if Path(pdf_version).exists():
                            pdf_size = Path(pdf_version).stat().st_size
                            print(f"  Generated: {pdf_version} ({pdf_size:,} bytes)")
                    
                    self.log_step(description, "SUCCESS", f"Generated {expected_output}", execution_time)
                    return True
                else:
                    print(f"[ERROR] Visualization returned code {result.returncode}")
                    if result.stderr:
                        print(f"  Error: {result.stderr[:500]}")
                    self.log_step(description, "FAILED", "Non-zero exit", execution_time)
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"[ERROR] Visualization timed out")
                self.log_step(description, "FAILED", "Timeout")
                return False
            except Exception as e:
                print(f"[ERROR] Could not run {description}: {e}")
                self.log_step(description, "FAILED", str(e))
                return False
                
        finally:
            os.chdir(original_dir)
    
    def run_model_evaluation(self):
        """Run model evaluation and statistical analysis"""
        
        self.print_header("STEP 4: MODEL EVALUATION AND STATISTICAL ANALYSIS", 1)
        
        # Change to experiment directory
        original_dir = os.getcwd()
        experiment_path = self.get_experiment_path()
        os.chdir(str(experiment_path))
        
        try:
            # Change to output directory for model evaluation
            os.chdir('output')
            
            # Use the model evaluation script from GENERALIZED_PIPELINE
            script = str(experiment_path.parent / 'GENERALIZED_PIPELINE' / 'model_evaluation_analysis.py')
            
            description = 'Model Evaluation and Statistical Analysis'
            expected_output_dir = 'Model_analysis'
            
            if not Path(script).exists():
                print(f"[ERROR] {description} - script not found at {script}")
                return False
            
            print(f"[INFO] Using model evaluation script: {script}")
            self.print_header(description, 2)
            
            start_time = time.time()
            
            try:
                # Pass config as environment variables
                env = os.environ.copy()
                env['DUERNAST_YEAR'] = str(self.config.year)
                env['DUERNAST_FILE_PREFIX'] = self.config.file_prefix
                env['DUERNAST_OUTPUT_PREFIX'] = self.config.output_prefix
                
                result = subprocess.run([sys.executable, script], 
                                      capture_output=True, text=True, timeout=300, env=env)
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"[SUCCESS] {description} completed ({execution_time:.2f}s)")
                    
                    # Check outputs
                    model_analysis_dir = Path(expected_output_dir)
                    if model_analysis_dir.exists():
                        csv_files = list(model_analysis_dir.glob('*.csv'))
                        png_files = list(model_analysis_dir.glob('*.png'))
                        pdf_files = list(model_analysis_dir.glob('*.pdf'))
                        
                        print(f"  Generated files in {expected_output_dir}/:")
                        for f in csv_files + png_files + pdf_files:
                            size = f.stat().st_size
                            print(f"    - {f.name} ({size:,} bytes)")
                    
                    self.log_step(description, "SUCCESS", f"Generated analysis files", execution_time)
                    return True
                else:
                    print(f"[ERROR] Model evaluation returned code {result.returncode}")
                    if result.stderr:
                        print(f"  Error: {result.stderr[:500]}")
                    if result.stdout:
                        print(f"  Output: {result.stdout[-500:]}")
                    self.log_step(description, "FAILED", "Non-zero exit", execution_time)
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"[ERROR] Model evaluation timed out")
                self.log_step(description, "FAILED", "Timeout")
                return False
            except Exception as e:
                print(f"[ERROR] Could not run {description}: {e}")
                self.log_step(description, "FAILED", str(e))
                return False
                
        finally:
            os.chdir(original_dir)
    
    def generate_summary(self):
        """Generate workflow summary"""
        
        self.print_header("WORKFLOW SUMMARY", 1)
        
        # Change to experiment directory
        original_dir = os.getcwd()
        experiment_path = self.get_experiment_path()
        os.chdir(str(experiment_path))
        
        try:
            # Collect generated outputs
            output_files = {
                'Visualization Files': [
                    f'output/{self.config.output_prefix}_comprehensive_analysis.png',
                    f'output/{self.config.output_prefix}_comprehensive_analysis.pdf'
                ],
                'Model Evaluation Files': [
                    f'output/Model_analysis/{self.config.output_prefix}_comparison_all_treatments.csv',
                    f'output/Model_analysis/{self.config.output_prefix}_model_metrics_summary.csv',
                    f'output/Model_analysis/{self.config.output_prefix}_model_evaluation.png',
                    f'output/Model_analysis/{self.config.output_prefix}_model_evaluation.pdf'
                ],
                'DSSAT Output Files': [
                    'output/Summary.OUT', 'output/OVERVIEW.OUT', 'output/PlantGro.OUT', 'output/PlantN.OUT',
                    'output/SoilWat.OUT', 'output/SoilNi.OUT', 'output/Weather.OUT'
                ]
            }
            
            print("Generated Output Files:")
            print("=" * 70)
            
            total_files = 0
            total_size = 0
            
            for category, files in output_files.items():
                print(f"\n{category}:")
                print("-" * len(category))
                
                for filename in files:
                    if Path(filename).exists():
                        size = Path(filename).stat().st_size
                        total_files += 1
                        total_size += size
                        print(f"  [OK] {filename:<50} ({size:>10,} bytes)")
            
            # Performance summary
            print(f"\n\nWorkflow Performance:")
            print("=" * 40)
            
            workflow_duration = (datetime.now() - self.start_time).total_seconds()
            successful_steps = len([s for s in self.workflow_steps if s['status'] == 'SUCCESS'])
            total_steps = len(self.workflow_steps)
            
            print(f"  Experiment: Duernast {self.config.year} {self.config.crop_type}")
            print(f"  Model: N-Wheat (WHAPS048)")
            print(f"  Treatments: 15 (various N levels and types)")
            print(f"  Total Steps: {total_steps}")
            print(f"  Successful: {successful_steps}")
            print(f"  Success Rate: {(successful_steps/total_steps)*100:.1f}%")
            print(f"  Workflow Time: {workflow_duration:.2f} seconds")
            print(f"  Files Generated: {total_files}")
            print(f"  Total Size: {total_size/1024/1024:.1f} MB")
            
            return True
            
        finally:
            os.chdir(original_dir)
    
    def run_complete_workflow(self):
        """Execute the complete workflow"""
        
        self.print_header(f"DUERNAST {self.config.year} N-WHEAT ANALYSIS WORKFLOW", 1)
        print("Complete end-to-end analysis from DSSAT simulation to visualization")
        print(f"Experiment: {self.config.crop_type} Nitrogen Response Trial (15 treatments)")
        print(f"Location: Dürnast, Freising, Bayern, Germany")
        print(f"Model: N-Wheat (WHAPS048)")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute workflow steps
        workflow_steps = [
            (self.check_prerequisites, "Prerequisites Check"),
            (self.run_dssat_simulation, "DSSAT Simulation"),
            (self.run_visualization, "Visualization Generation"),
            (self.run_model_evaluation, "Model Evaluation and Statistical Analysis"),
            (self.generate_summary, "Workflow Summary")
        ]
        
        failed = False
        for step_func, step_name in workflow_steps:
            if not step_func():
                self.print_header(f"WORKFLOW FAILED AT: {step_name}", 1)
                print(f"[ERROR] Step '{step_name}' failed")
                print(f"[INFO] Check error messages above")
                failed = True
                break
        
        if not failed:
            # Success summary
            self.print_header("WORKFLOW COMPLETED SUCCESSFULLY!", 1)
            
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()
            
            print(f"Complete Duernast {self.config.year} N-Wheat analysis workflow finished.")
            print(f"Total execution time: {total_duration:.2f} seconds")
            print(f"Analyzed 15 nitrogen treatments")
            print(f"Generated comprehensive visualization")
            
            print(f"\nOUTPUT FILES:")
            print(f"  - output/{self.config.output_prefix}_comprehensive_analysis.png")
            print(f"  - output/{self.config.output_prefix}_comprehensive_analysis.pdf")
            print(f"  - output/Model_analysis/{self.config.output_prefix}_comparison_all_treatments.csv")
            print(f"  - output/Model_analysis/{self.config.output_prefix}_model_metrics_summary.csv")
            print(f"  - output/Model_analysis/{self.config.output_prefix}_model_evaluation.png")
            print(f"  - output/Summary.OUT (main results)")
            print(f"  - output/PlantGro.OUT (growth time series)")
            print(f"  - output/PlantN.OUT (nitrogen dynamics)")
            
            return True
        
        return False


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Generalized Duernast DSSAT Analysis Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python MASTER_WORKFLOW.py 2015        # Run 2015 experiment
  python MASTER_WORKFLOW.py 2017        # Run 2017 experiment
  python MASTER_WORKFLOW.py --all       # Run all available experiments
        """
    )
    
    parser.add_argument(
        'year',
        nargs='?',
        type=int,
        help='Year of experiment to run (2015 or 2017)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available experiments sequentially'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    
    args = parser.parse_args()
    
    # List available experiments
    if args.list:
        available = list_available_experiments()
        print("Available experiments:")
        for year in available:
            config = get_config(year)
            print(f"  {year}: {config.experiment_name} ({config.crop_type})")
        return 0
    
    # Run all experiments
    if args.all:
        available = list_available_experiments()
        print(f"Running all {len(available)} experiments...")
        results = {}
        for year in available:
            print(f"\n{'='*90}")
            print(f"Running experiment {year}")
            print('='*90)
            config = get_config(year)
            workflow = GeneralizedDuernastWorkflowManager(config)
            success = workflow.run_complete_workflow()
            results[year] = success
        
        # Summary
        print(f"\n{'='*90}")
        print("ALL EXPERIMENTS SUMMARY")
        print('='*90)
        for year, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {year}: {status}")
        
        return 0 if all(results.values()) else 1
    
    # Run single experiment
    if not args.year:
        parser.print_help()
        return 1
    
    try:
        config = get_config(args.year)
    except ValueError as e:
        print(f"[ERROR] {e}")
        print(f"Available experiments: {list_available_experiments()}")
        return 1
    
    # Create workflow manager
    workflow = GeneralizedDuernastWorkflowManager(config)
    
    # Run complete workflow
    success = workflow.run_complete_workflow()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    
    print(f"\n{'='*90}")
    print(f"WORKFLOW EXIT CODE: {exit_code}")
    if exit_code == 0:
        print("STATUS: SUCCESS")
    else:
        print("STATUS: FAILED")
    print('='*90)
    
    sys.exit(exit_code)

