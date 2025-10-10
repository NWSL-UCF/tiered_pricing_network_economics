"""
Parameter sweep runner for Transit ISP Pricing Competition Analysis.

This script runs the analysis across multiple parameter combinations.
"""

import json
import itertools
import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime
import os


def setup_logger():
    """Set up logger for the wrapper script."""
    logger = logging.getLogger('parameter_sweep')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_parameters(params_file: str) -> dict:
    """
    Load parameter combinations from JSON file.
    
    Args:
        params_file: Path to params.json file
        
    Returns:
        Dictionary with parameter lists
    """
    with open(params_file, 'r') as f:
        return json.load(f)


def generate_combinations(params: dict) -> list:
    """
    Generate all parameter combinations.
    
    Args:
        params: Dictionary with parameter lists
        
    Returns:
        List of parameter combinations
    """
    # Get parameter names and values
    param_names = list(params.keys())
    param_values = list(params.values())
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Convert to list of dictionaries
    param_combinations = []
    for combo in combinations:
        param_dict = dict(zip(param_names, combo))
        param_combinations.append(param_dict)
    
    return param_combinations


def create_run_id(combo: dict, index: int) -> str:
	"""
	Create a unique run ID for the parameter combination.
	
	Args:
		combo: Parameter combination dictionary
		index: Combination index
		
	Returns:
		Unique run ID string
	"""
	# Create a short identifier based on key parameters
	gamma = combo['gamma']
	s0 = combo['s0']
	alpha = combo['alpha']
    beta = combo['beta']
	
	return f"run_{index:04d}_gamma_{gamma}_s0_{s0}_alpha_{alpha}_beta_{beta}"


def run_simulation(combo: dict, index: int, base_output_dir: str, 
                  sim_script: str, data_file: str, config_file: str, logger: logging.Logger) -> bool:
    """
    Run simulation for a single parameter combination.
    
    Args:
        combo: Parameter combination dictionary
        index: Combination index
        base_output_dir: Base directory for all outputs
        sim_script: Path to sim.py script
        data_file: Path to data CSV file
        config_file: Path to tier strategies JSON file
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    # Create run ID
    run_id = create_run_id(combo, index)
    
    # Create output directory for this run
    output_dir = Path(base_output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
	# Build command - using the new modular structure
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
		'--log-level', 'INFO'
	]
	
	logger.info(f"Running simulation {index + 1}: {run_id}")
	logger.info(f"Parameters: γ={combo['gamma']}, β={combo['beta']}, α={combo['alpha']}, s0={combo['s0']} (P0 calculated endogenously)")
	logger.info(f"Output directory: {output_dir}")
    
    try:
        # Run the simulation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info(f"✅ Simulation {index + 1} completed successfully")
        logger.debug(f"Command output: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Simulation {index + 1} failed with error code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        logger.error(f"Command: {' '.join(cmd)}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Simulation {index + 1} failed with exception: {str(e)}")
        return False


def create_summary_report(results: list, base_output_dir: str, logger: logging.Logger):
    """
    Create a summary report of all runs.
    
    Args:
        results: List of (success, run_id, combo) tuples
        base_output_dir: Base directory for all outputs
        logger: Logger instance
    """
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
        'successful_runs': [
            {
                'run_id': run_id,
                'parameters': combo,
                'output_directory': str(Path(base_output_dir) / run_id)
            }
            for success, run_id, combo in successful_runs
        ],
        'failed_runs': [
            {
                'run_id': run_id,
                'parameters': combo,
                'output_directory': str(Path(base_output_dir) / run_id)
            }
            for success, run_id, combo in failed_runs
        ]
    }
    
    # Save summary report
    summary_file = Path(base_output_dir) / 'parameter_sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📊 Summary report saved to: {summary_file}")
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Successful: {len(successful_runs)}")
    logger.info(f"Failed: {len(failed_runs)}")


def main():
    """Main function to run parameter sweep."""
    # Set up paths
    script_dir = Path(__file__).parent
    params_file = script_dir / 'params.json'
    sim_script = script_dir / 'sim.py'
    data_file = script_dir / 'netflow_grouped_by_src_dst.csv'
    config_file = script_dir / 'tier_strategies.json'
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = script_dir / f'parameter_sweep_results_{timestamp}'
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger()
    
    logger.info("="*80)
    logger.info("PARAMETER SWEEP SIMULATION")
    logger.info("="*80)
    logger.info(f"Parameters file: {params_file}")
    logger.info(f"Simulation script: {sim_script}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Base output directory: {base_output_dir}")
    logger.info("="*80)
    
    # Check if required files exist
    required_files = [params_file, sim_script, data_file, config_file]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        sys.exit(1)
    
    # Load parameters
    logger.info("Loading parameters...")
    params = load_parameters(params_file)
    logger.info(f"Parameter ranges:")
    for param, values in params.items():
        logger.info(f"  {param}: {values}")
    
    # Generate combinations
    logger.info("Generating parameter combinations...")
    combinations = generate_combinations(params)
    total_combinations = len(combinations)
    logger.info(f"Total combinations: {total_combinations}")
    
    # Run simulations
    logger.info("Starting parameter sweep...")
    results = []
    
    for i, combo in enumerate(combinations):
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN {i + 1}/{total_combinations}")
        logger.info(f"{'='*60}")
        
        success = run_simulation(
            combo, i, base_output_dir, 
            str(sim_script), str(data_file), str(config_file), logger
        )
        
        # Create run ID for result tracking
        run_id = create_run_id(combo, i)
        results.append((success, run_id, combo))
        
        # Progress update
        completed = i + 1
        progress = (completed / total_combinations) * 100
        logger.info(f"Progress: {completed}/{total_combinations} ({progress:.1f}%)")
    
    # Create summary report
    logger.info("\n" + "="*80)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("="*80)
    create_summary_report(results, base_output_dir, logger)
    
    logger.info("\n" + "="*80)
    logger.info("PARAMETER SWEEP COMPLETE")
    logger.info("="*80)
    logger.info(f"All results saved to: {base_output_dir}")
    
    # Final statistics
    successful = sum(1 for r in results if r[0])
    failed = len(results) - successful
    
    logger.info(f"Final Results:")
    logger.info(f"  Total runs: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {(successful/len(results)*100):.1f}%")


if __name__ == "__main__":
    main()