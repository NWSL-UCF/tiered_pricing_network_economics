import json
import csv
import os
from pathlib import Path
import time
from datetime import datetime

def analyze_folder_data(summary, welfare_data, n):
    """
    Analyze data from summary.json and welfare_matrix.json.
    
    Returns a dictionary with analysis results.
    """
    nash_equilibria = summary.get("nash_equilibria", [])
    welfare_matrix = welfare_data.get("welfare_matrix", {})
    strategies = summary.get("strategies", {})
    
    # Get list of strategy names
    strategy_names = list(strategies.keys())
    
    results = {
        "nash_equilibria": [],
        "total_ne": len(nash_equilibria),
        "best_ne_consumer_surplus": None,
        "best_ne_total_welfare": None,
        "highest_profit_both_isps": None,
        "all_scenarios": []
    }
    
    # Process Nash Equilibria
    for ne in nash_equilibria:
        ne_info = {
            "strategy_A": ne["strategy_A"],
            "strategy_B": ne["strategy_B"],
            "profit_A": ne["profit_A"],
            "profit_B": ne["profit_B"],
            "total_profit": ne["total_profit"]
        }
        
        # Get welfare info for this NE
        strategy_a = ne["strategy_A"]
        strategy_b = ne["strategy_B"]
        
        if strategy_a in welfare_matrix and strategy_b in welfare_matrix[strategy_a]:
            welfare_info = welfare_matrix[strategy_a][strategy_b]
            ne_info.update({
                "consumer_surplus": welfare_info["consumer_surplus"],
                "social_welfare": welfare_info["social_welfare"]
            })
        
        results["nash_equilibria"].append(ne_info)
    
    # Find best NE by consumer surplus
    if nash_equilibria:
        best_cs_ne = max(results["nash_equilibria"], 
                        key=lambda x: x.get("consumer_surplus", float('-inf')))
        results["best_ne_consumer_surplus"] = best_cs_ne
        
        # Find best NE by total welfare
        best_tw_ne = max(results["nash_equilibria"], 
                        key=lambda x: x.get("social_welfare", float('-inf')))
        results["best_ne_total_welfare"] = best_tw_ne
    
    # Analyze all scenarios to find highest profit for both ISPs
    max_min_profit = float('-inf')
    best_scenario = None
    
    for strategy_a in strategy_names:
        for strategy_b in strategy_names:
            if strategy_a in welfare_matrix and strategy_b in welfare_matrix[strategy_a]:
                scenario = welfare_matrix[strategy_a][strategy_b]
                profit_a = scenario["profit_A"]
                profit_b = scenario["profit_B"]
                
                # Store all scenarios
                scenario_info = {
                    "strategy_A": strategy_a,
                    "strategy_B": strategy_b,
                    "profit_A": profit_a,
                    "profit_B": profit_b,
                    "total_profit": profit_a + profit_b,
                    "consumer_surplus": scenario["consumer_surplus"],
                    "social_welfare": scenario["social_welfare"],
                    "is_nash_equilibrium": False
                }
                
                # Check if this is a Nash Equilibrium
                for ne in nash_equilibria:
                    if ne["strategy_A"] == strategy_a and ne["strategy_B"] == strategy_b:
                        scenario_info["is_nash_equilibrium"] = True
                        break
                
                results["all_scenarios"].append(scenario_info)
                
                # Find scenario with highest profit for both ISPs
                # Using min of both profits to ensure both benefit
                min_profit = min(profit_a, profit_b)
                if min_profit > max_min_profit:
                    max_min_profit = min_profit
                    best_scenario = scenario_info
                # If same min profit, use total profit as tiebreaker
                elif min_profit == max_min_profit and best_scenario:
                    if (profit_a + profit_b) > (best_scenario["profit_A"] + best_scenario["profit_B"]):
                        best_scenario = scenario_info
    
    results["highest_profit_both_isps"] = best_scenario
    
    return results


def process_single_run(run_path, output_base_path):
    """
    Process a single run folder and generate the 5 CSV files.
    Returns True if successful, False otherwise.
    """
    run_path = Path(run_path)
    run_name = run_path.name
    
    # Create output directory for this run
    output_path = output_base_path / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    try:
        # Process each folder from 1x1 to 10x10
        for n in range(1, 11):
            folder_name = f"{n}x{n}"
            folder_path = run_path / folder_name
            
            if not folder_path.exists():
                continue
            
            # Read summary.json and welfare_matrix.json
            summary_path = folder_path / "summary.json"
            welfare_path = folder_path / "welfare_matrix.json"
            
            if not summary_path.exists() or not welfare_path.exists():
                continue
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            with open(welfare_path, 'r') as f:
                welfare_data = json.load(f)
            
            # Analyze the data
            analysis_results = analyze_folder_data(summary, welfare_data, n)
            analysis_results['config'] = folder_name
            analysis_results['n'] = n
            all_results.append(analysis_results)
        
        if not all_results:
            return False
        
        # Generate separate CSV files for each section
        write_separate_csvs(all_results, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {run_name}: {str(e)}")
        return False


def write_separate_csvs(all_results, base_path):
    """
    Write separate CSV files for each section of the analysis.
    """
    # 1. Configuration Summary
    csv_path = base_path / "1_configuration_summary.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Config", "Max Tiers per ISP", "Total Scenarios", "Total Nash Equilibria", 
                        "NE with Max Consumer Surplus", "NE with Max Total Welfare"])
        
        for result in all_results:
            ne_max_cs = result['best_ne_consumer_surplus']
            ne_max_tw = result['best_ne_total_welfare']
            
            ne_cs_str = f"{ne_max_cs['strategy_A']} vs {ne_max_cs['strategy_B']}" if ne_max_cs else "N/A"
            ne_tw_str = f"{ne_max_tw['strategy_A']} vs {ne_max_tw['strategy_B']}" if ne_max_tw else "N/A"
            
            writer.writerow([
                result['config'],
                result['n'],
                len(result['all_scenarios']),
                result['total_ne'],
                ne_cs_str,
                ne_tw_str
            ])
    
    # 2. All Nash Equilibria
    csv_path = base_path / "2_all_nash_equilibria.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Config", "Strategy A", "Strategy B", "Profit A", "Profit B", 
                        "Total Profit", "Consumer Surplus", "Social Welfare"])
        
        for result in all_results:
            for ne in result['nash_equilibria']:
                writer.writerow([
                    result['config'],
                    ne['strategy_A'],
                    ne['strategy_B'],
                    f"{ne['profit_A']:.2f}",
                    f"{ne['profit_B']:.2f}",
                    f"{ne['total_profit']:.2f}",
                    f"{ne.get('consumer_surplus', 0):.2f}",
                    f"{ne.get('social_welfare', 0):.2f}"
                ])
    
    # 3. Best NE by Consumer Surplus
    csv_path = base_path / "3_best_ne_consumer_surplus.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Config", "Strategy A", "Strategy B", "Profit A", "Profit B", 
                        "Total Profit", "Consumer Surplus", "Social Welfare"])
        
        for result in all_results:
            ne = result['best_ne_consumer_surplus']
            if ne:
                writer.writerow([
                    result['config'],
                    ne['strategy_A'],
                    ne['strategy_B'],
                    f"{ne['profit_A']:.2f}",
                    f"{ne['profit_B']:.2f}",
                    f"{ne['total_profit']:.2f}",
                    f"{ne['consumer_surplus']:.2f}",
                    f"{ne['social_welfare']:.2f}"
                ])
    
    # 4. Best NE by Total Welfare
    csv_path = base_path / "4_best_ne_total_welfare.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Config", "Strategy A", "Strategy B", "Profit A", "Profit B", 
                        "Total Profit", "Consumer Surplus", "Social Welfare"])
        
        for result in all_results:
            ne = result['best_ne_total_welfare']
            if ne:
                writer.writerow([
                    result['config'],
                    ne['strategy_A'],
                    ne['strategy_B'],
                    f"{ne['profit_A']:.2f}",
                    f"{ne['profit_B']:.2f}",
                    f"{ne['total_profit']:.2f}",
                    f"{ne['consumer_surplus']:.2f}",
                    f"{ne['social_welfare']:.2f}"
                ])
    
    # 5. Highest Profit for Both ISPs
    csv_path = base_path / "5_highest_profit_both_isps.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Config", "Strategy A", "Strategy B", "Profit A", "Profit B", 
                        "Total Profit", "Consumer Surplus", "Social Welfare", "Is Nash Equilibrium"])
        
        for result in all_results:
            scenario = result['highest_profit_both_isps']
            if scenario:
                writer.writerow([
                    result['config'],
                    scenario['strategy_A'],
                    scenario['strategy_B'],
                    f"{scenario['profit_A']:.2f}",
                    f"{scenario['profit_B']:.2f}",
                    f"{scenario['total_profit']:.2f}",
                    f"{scenario['consumer_surplus']:.2f}",
                    f"{scenario['social_welfare']:.2f}",
                    "Yes" if scenario['is_nash_equilibrium'] else "No"
                ])


def get_run_folders(input_path):
    """
    Get list of all run folders to process.
    """
    input_path = Path(input_path)
    run_folders = sorted([d for d in input_path.iterdir() 
                         if d.is_dir() and d.name.startswith("run_")])
    return run_folders


def batch_process_all_runs(input_path, output_path, start_from=0, max_runs=None):
    """
    Process all run folders in batch.
    
    Args:
        input_path: Path to directory containing all run_* folders
        output_path: Path where output CSV files will be saved
        start_from: Index to start from (for resuming)
        max_runs: Maximum number of runs to process (None = all)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all run folders
    run_folders = get_run_folders(input_path)
    total_runs = len(run_folders)
    
    print(f"Found {total_runs} run folders to process")
    
    if start_from > 0:
        run_folders = run_folders[start_from:]
        print(f"Starting from index {start_from}")
    
    if max_runs:
        run_folders = run_folders[:max_runs]
        print(f"Processing first {max_runs} runs")
    
    # Process each run
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Create log file
    log_path = output_path / "processing_log.txt"
    
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n\n=== Batch processing started at {datetime.now()} ===\n")
        log_file.write(f"Total runs to process: {len(run_folders)}\n")
        
        for idx, run_folder in enumerate(run_folders):
            current_idx = start_from + idx
            
            # Progress update every 100 runs
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                if idx > 0:
                    rate = idx / elapsed
                    remaining = (len(run_folders) - idx) / rate
                    print(f"Progress: {idx}/{len(run_folders)} ({100*idx/len(run_folders):.1f}%) "
                          f"- Rate: {rate:.1f} runs/sec - ETA: {remaining/60:.1f} min")
                else:
                    print(f"Processing run {current_idx}: {run_folder.name}")
            
            # Process the run
            success = process_single_run(run_folder, output_path)
            
            if success:
                successful += 1
                log_file.write(f"[{datetime.now()}] SUCCESS: {run_folder.name}\n")
            else:
                failed += 1
                log_file.write(f"[{datetime.now()}] FAILED: {run_folder.name}\n")
            
            # Flush log every 1000 runs
            if idx % 1000 == 0:
                log_file.flush()
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Batch processing complete!")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Rate: {successful/elapsed:.2f} runs/sec")
    print(f"Output saved to: {output_path}")
    print(f"Log saved to: {log_path}")
    print(f"{'='*80}")
    
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n=== Batch processing completed at {datetime.now()} ===\n")
        log_file.write(f"Total time: {elapsed/60:.1f} minutes\n")
        log_file.write(f"Successful: {successful}\n")
        log_file.write(f"Failed: {failed}\n")


if __name__ == "__main__":
    import sys
    
    input_path = "/home/ab823254/data/raw/tieredPricing"
    output_path = "/home/ab823254/data/tiered_pricing_network_economics/result_analysis/batch_results"
    
    # Optional: parse command line arguments for resuming
    start_from = 0
    max_runs = None
    
    if len(sys.argv) > 1:
        start_from = int(sys.argv[1])
    if len(sys.argv) > 2:
        max_runs = int(sys.argv[2])
    
    print("="*80)
    print("Batch Processing All Runs")
    print("="*80)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    batch_process_all_runs(input_path, output_path, start_from, max_runs)

