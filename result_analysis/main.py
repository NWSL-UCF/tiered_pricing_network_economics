import json
import csv
import os
from pathlib import Path

def analyze_competition_results(base_path):
    """
    Analyze competition results from all NxN folders and generate CSV reports.
    
    Args:
        base_path: Path to the run directory containing NxN folders
    """
    base_path = Path(base_path)
    
    # Process each folder from 1x1 to 10x10
    for n in range(1, 11):
        folder_name = f"{n}x{n}"
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"Folder {folder_name} not found, skipping...")
            continue
        
        print(f"Processing {folder_name}...")
        
        # Read summary.json and welfare_matrix.json
        summary_path = folder_path / "summary.json"
        welfare_path = folder_path / "welfare_matrix.json"
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        with open(welfare_path, 'r') as f:
            welfare_data = json.load(f)
        
        # Analyze the data
        analysis_results = analyze_folder_data(summary, welfare_data, n)
        
        # Generate CSV report
        csv_path = folder_path / f"{folder_name}_analysis.csv"
        generate_csv_report(analysis_results, csv_path, n)
        
        print(f"Generated CSV report: {csv_path}")

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

def generate_csv_report(analysis_results, csv_path, n):
    """
    Generate CSV report with analysis results.
    """
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header section
        writer.writerow(["Competition Analysis Report"])
        writer.writerow([f"Configuration: {n}x{n} (ISP-A: 1-{n} tiers, ISP-B: 1-{n} tiers)"])
        writer.writerow([])
        
        # Summary statistics
        writer.writerow(["Summary Statistics"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Number of Nash Equilibria", analysis_results["total_ne"]])
        writer.writerow(["Total Number of Scenarios", len(analysis_results["all_scenarios"])])
        writer.writerow([])
        
        # Nash Equilibria section
        writer.writerow(["Nash Equilibria"])
        writer.writerow(["Strategy A", "Strategy B", "Profit A", "Profit B", 
                        "Total Profit", "Consumer Surplus", "Social Welfare"])
        
        for ne in analysis_results["nash_equilibria"]:
            writer.writerow([
                ne["strategy_A"],
                ne["strategy_B"],
                f"{ne['profit_A']:.2f}",
                f"{ne['profit_B']:.2f}",
                f"{ne['total_profit']:.2f}",
                f"{ne.get('consumer_surplus', 'N/A'):.2f}" if isinstance(ne.get('consumer_surplus'), (int, float)) else "N/A",
                f"{ne.get('social_welfare', 'N/A'):.2f}" if isinstance(ne.get('social_welfare'), (int, float)) else "N/A"
            ])
        writer.writerow([])
        
        # Best NE by consumer surplus
        if analysis_results["best_ne_consumer_surplus"]:
            writer.writerow(["Best Nash Equilibrium (Consumer Surplus Maximized)"])
            writer.writerow(["Strategy A", "Strategy B", "Profit A", "Profit B", 
                            "Total Profit", "Consumer Surplus", "Social Welfare"])
            ne = analysis_results["best_ne_consumer_surplus"]
            writer.writerow([
                ne["strategy_A"],
                ne["strategy_B"],
                f"{ne['profit_A']:.2f}",
                f"{ne['profit_B']:.2f}",
                f"{ne['total_profit']:.2f}",
                f"{ne['consumer_surplus']:.2f}",
                f"{ne['social_welfare']:.2f}"
            ])
            writer.writerow([])
        
        # Best NE by total welfare
        if analysis_results["best_ne_total_welfare"]:
            writer.writerow(["Best Nash Equilibrium (Total Welfare Maximized)"])
            writer.writerow(["Strategy A", "Strategy B", "Profit A", "Profit B", 
                            "Total Profit", "Consumer Surplus", "Social Welfare"])
            ne = analysis_results["best_ne_total_welfare"]
            writer.writerow([
                ne["strategy_A"],
                ne["strategy_B"],
                f"{ne['profit_A']:.2f}",
                f"{ne['profit_B']:.2f}",
                f"{ne['total_profit']:.2f}",
                f"{ne['consumer_surplus']:.2f}",
                f"{ne['social_welfare']:.2f}"
            ])
            writer.writerow([])
        
        # Best scenario for both ISPs profit
        if analysis_results["highest_profit_both_isps"]:
            writer.writerow(["Scenario with Highest Profit for Both ISPs"])
            writer.writerow(["Strategy A", "Strategy B", "Profit A", "Profit B", 
                            "Total Profit", "Consumer Surplus", "Social Welfare", "Is Nash Equilibrium"])
            scenario = analysis_results["highest_profit_both_isps"]
            writer.writerow([
                scenario["strategy_A"],
                scenario["strategy_B"],
                f"{scenario['profit_A']:.2f}",
                f"{scenario['profit_B']:.2f}",
                f"{scenario['total_profit']:.2f}",
                f"{scenario['consumer_surplus']:.2f}",
                f"{scenario['social_welfare']:.2f}",
                "Yes" if scenario["is_nash_equilibrium"] else "No"
            ])
            writer.writerow([])
        
        # All scenarios
        writer.writerow(["All Scenarios (Payoff Matrix)"])
        writer.writerow(["Strategy A", "Strategy B", "Profit A", "Profit B", 
                        "Total Profit", "Consumer Surplus", "Social Welfare", "Is Nash Equilibrium"])
        
        for scenario in analysis_results["all_scenarios"]:
            writer.writerow([
                scenario["strategy_A"],
                scenario["strategy_B"],
                f"{scenario['profit_A']:.2f}",
                f"{scenario['profit_B']:.2f}",
                f"{scenario['total_profit']:.2f}",
                f"{scenario['consumer_surplus']:.2f}",
                f"{scenario['social_welfare']:.2f}",
                "Yes" if scenario["is_nash_equilibrium"] else "No"
            ])

def generate_consolidated_summary(base_path):
    """
    Generate separate CSV files for each section of the analysis.
    """
    base_path = Path(base_path)
    all_results = []
    
    # Process each folder from 1x1 to 10x10
    for n in range(1, 11):
        folder_name = f"{n}x{n}"
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"Folder {folder_name} not found, skipping...")
            continue
        
        print(f"Processing {folder_name}...")
        
        # Read summary.json and welfare_matrix.json
        summary_path = folder_path / "summary.json"
        welfare_path = folder_path / "welfare_matrix.json"
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        with open(welfare_path, 'r') as f:
            welfare_data = json.load(f)
        
        # Analyze the data
        analysis_results = analyze_folder_data(summary, welfare_data, n)
        analysis_results['config'] = folder_name
        analysis_results['n'] = n
        all_results.append(analysis_results)
    
    # Generate separate CSV files for each section
    csv_files = write_separate_csvs(all_results, base_path)
    print(f"\nAnalysis complete! Generated {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    return csv_files

def write_separate_csvs(all_results, base_path):
    """
    Write separate CSV files for each section of the analysis.
    """
    csv_files = []
    
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
    csv_files.append(csv_path)
    
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
    csv_files.append(csv_path)
    
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
    csv_files.append(csv_path)
    
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
    csv_files.append(csv_path)
    
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
    csv_files.append(csv_path)
    
    return csv_files

def write_consolidated_csv(all_results, csv_path):
    """
    Write consolidated summary CSV file.
    """
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(["Consolidated Competition Analysis Report"])
        writer.writerow(["Summary of Nash Equilibria and Key Scenarios Across All Configurations"])
        writer.writerow([])
        
        # Summary table
        writer.writerow(["Configuration Summary"])
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
        
        writer.writerow([])
        writer.writerow([])
        
        # Detailed Nash Equilibria for each configuration
        writer.writerow(["All Nash Equilibria by Configuration"])
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
        
        writer.writerow([])
        writer.writerow([])
        
        # Best NE by consumer surplus for each config
        writer.writerow(["Best Nash Equilibrium by Consumer Surplus (per configuration)"])
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
        
        writer.writerow([])
        writer.writerow([])
        
        # Best NE by total welfare for each config
        writer.writerow(["Best Nash Equilibrium by Total Welfare (per configuration)"])
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
        
        writer.writerow([])
        writer.writerow([])
        
        # Scenario with highest profit for both ISPs
        writer.writerow(["Scenario with Highest Profit for Both ISPs (per configuration)"])
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

if __name__ == "__main__":
    base_path = "/home/ab823254/data/tiered_pricing_network_economics/result_analysis/run_53513_g0.0500_b0.50_a2.27_s0.74"
    
    # Generate separate CSV files for each section
    csv_files = generate_consolidated_summary(base_path)
