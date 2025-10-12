"""
Parameter sweep runner for Transit ISP Pricing Competition Analysis.

This script runs the analysis across all parameter combinations from params.json.
Generates 30×30×30×2 = 54,000 simulation runs.
"""

import json
import itertools
import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime
import multiprocessing
from functools import partial


def setup_logger(log_file=None):
    """Set up logger for the parameter sweep."""
    logger = logging.getLogger('parameter_sweep')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_parameters(params_file: str) -> dict:
    """Load parameter combinations from JSON file."""
    with open(params_file, 'r') as f:
        return json.load(f)


def generate_combinations(params: dict) -> list:
    """Generate all parameter combinations."""
    param_names = ['gamma', 'beta', 'alpha', 's0']  # Fixed order
    param_values = [params[name] for name in param_names]
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Convert to list of dictionaries
    param_combinations = []
    for combo in combinations:
        param_dict = dict(zip(param_names, combo))
        param_combinations.append(param_dict)
    
    return param_combinations


def create_run_id(combo: dict, index: int) -> str:
    """Create a unique run ID for the parameter combination."""
    gamma = combo['gamma']
    beta = combo['beta']
    alpha = combo['alpha']
    s0 = combo['s0']
    
    return f"run_{index:05d}_g{gamma:.4f}_b{beta:.2f}_a{alpha:.2f}_s{s0:.2f}"


def run_single_simulation(args):
    """
    Run simulation for a single parameter combination.
    Designed to be called by multiprocessing.
    """
    combo, index, base_output_dir, data_file, config_file = args
    
    # Create run ID
    run_id = create_run_id(combo, index)
    
    # Create output directory for this run
    output_dir = Path(base_output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        'python', '-m', 'src.main',
        '--base-path', str(output_dir),
        '--gamma', str(combo['gamma']),
        '--beta', str(combo['beta']),
        '--alpha', str(combo['alpha']),
        '--s0', str(combo['s0']),
        '--data', data_file,
        '--config', config_file,
        '--run-id', run_id,
        '--log-level', 'WARNING'  # Suppress verbose output for parallel runs
    ]
    
    try:
        # Run the simulation
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=Path(__file__).parent  # Run from exp/ directory
        )
        
        return (True, run_id, combo, None)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}: {e.stderr[:200]}"
        return (False, run_id, combo, error_msg)
        
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        return (False, run_id, combo, error_msg)


def create_summary_report(results: list, base_output_dir: str, params: dict, logger: logging.Logger):
    """Create a comprehensive summary report of all runs."""
    successful_runs = [r for r in results if r[0]]
    failed_runs = [r for r in results if not r[0]]
    
    summary = {
        'metadata': {
            'total_runs': len(results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'timestamp': datetime.now().isoformat(),
            'base_output_dir': str(base_output_dir)
        },
        'parameter_ranges': {
            'gamma': {'min': min(params['gamma']), 'max': max(params['gamma']), 'count': len(params['gamma'])},
            'beta': {'min': min(params['beta']), 'max': max(params['beta']), 'count': len(params['beta'])},
            'alpha': {'min': min(params['alpha']), 'max': max(params['alpha']), 'count': len(params['alpha'])},
            's0': {'min': min(params['s0']), 'max': max(params['s0']), 'count': len(params['s0'])}
        },
        'successful_runs': [
            {
                'run_id': run_id,
                'parameters': combo,
                'output_directory': str(Path(base_output_dir) / run_id)
            }
            for success, run_id, combo, _ in successful_runs
        ],
        'failed_runs': [
            {
                'run_id': run_id,
                'parameters': combo,
                'error': error_msg,
                'output_directory': str(Path(base_output_dir) / run_id)
            }
            for success, run_id, combo, error_msg in failed_runs
        ]
    }
    
    # Save summary report
    summary_file = Path(base_output_dir) / 'parameter_sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n📊 Summary report saved to: {summary_file}")


def main():
    """Main function to run parameter sweep."""
    # Set up paths
    script_dir = Path(__file__).parent
    params_file = script_dir / 'params.json'
    data_file = script_dir.parent / 'dataset' / 'netflow.csv'
    config_file = script_dir / 'tier_strategies.json'
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = script_dir / f'parameter_sweep_results_{timestamp}'
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    log_file = base_output_dir / 'parameter_sweep.log'
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("PARAMETER SWEEP SIMULATION")
    logger.info("="*80)
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Base output directory: {base_output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Check if required files exist
    if not params_file.exists():
        logger.error(f"❌ Missing params.json: {params_file}")
        sys.exit(1)
    if not data_file.exists():
        logger.error(f"❌ Missing data file: {data_file}")
        sys.exit(1)
    if not config_file.exists():
        logger.error(f"❌ Missing config file: {config_file}")
        sys.exit(1)
    
    # Load parameters
    logger.info("\n📋 Loading parameters...")
    params = load_parameters(params_file)
    logger.info(f"Parameter ranges:")
    for param, values in params.items():
        logger.info(f"  {param}: {len(values)} values from {min(values)} to {max(values)}")
    
    # Generate combinations
    logger.info("\n🔢 Generating parameter combinations...")
    combinations = generate_combinations(params)
    total_combinations = len(combinations)
    logger.info(f"Total combinations: {total_combinations:,}")
    
    # Ask for confirmation
    estimated_time = total_combinations * 5  # Rough estimate: 5 seconds per run
    logger.info(f"Estimated time: {estimated_time/3600:.1f} hours (assuming 5 sec/run)")
    
    # Determine number of parallel processes
    n_cores = multiprocessing.cpu_count()
    n_processes = max(1, n_cores - 2)  # Leave 2 cores free
    logger.info(f"Using {n_processes} parallel processes (out of {n_cores} available cores)")
    
    # Prepare arguments for parallel execution
    args_list = [
        (combo, i, base_output_dir, str(data_file), str(config_file))
        for i, combo in enumerate(combinations)
    ]
    
    # Run simulations in parallel
    logger.info("\n🚀 Starting parameter sweep...\n")
    start_time = datetime.now()
    
    results = []
    completed = 0
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        for result in pool.imap_unordered(run_single_simulation, args_list):
            results.append(result)
            completed += 1
            
            # Progress update every 100 runs
            if completed % 100 == 0 or completed == total_combinations:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_combinations - completed) / rate if rate > 0 else 0
                progress = (completed / total_combinations) * 100
                
                success_count = sum(1 for r in results if r[0])
                success_rate = (success_count / completed * 100) if completed > 0 else 0
                
                logger.info(
                    f"Progress: {completed:,}/{total_combinations:,} ({progress:.1f}%) | "
                    f"Success: {success_count:,} ({success_rate:.1f}%) | "
                    f"Rate: {rate:.1f}/sec | "
                    f"ETA: {remaining/60:.1f} min"
                )
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Create summary report
    logger.info("\n" + "="*80)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("="*80)
    create_summary_report(results, base_output_dir, params, logger)
    
    # Final statistics
    successful = sum(1 for r in results if r[0])
    failed = len(results) - successful
    
    logger.info("\n" + "="*80)
    logger.info("PARAMETER SWEEP COMPLETE")
    logger.info("="*80)
    logger.info(f"All results saved to: {base_output_dir}")
    logger.info(f"\nFinal Results:")
    logger.info(f"  Total runs: {len(results):,}")
    logger.info(f"  Successful: {successful:,}")
    logger.info(f"  Failed: {failed:,}")
    logger.info(f"  Success rate: {(successful/len(results)*100):.1f}%")
    logger.info(f"\nExecution Time:")
    logger.info(f"  Total time: {total_time/3600:.2f} hours")
    logger.info(f"  Average per run: {total_time/len(results):.2f} seconds")
    logger.info(f"  Rate: {len(results)/total_time:.2f} runs/second")
    logger.info("="*80)
    
    # Print failed runs if any
    if failed > 0:
        logger.info("\n❌ Failed runs:")
        for success, run_id, combo, error_msg in results:
            if not success:
                logger.info(f"  {run_id}: {error_msg}")


if __name__ == "__main__":
    main()
