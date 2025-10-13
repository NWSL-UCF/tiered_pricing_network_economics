import csv
import json
import re
from pathlib import Path
from collections import Counter
import time
from datetime import datetime


def parse_folder_name(folder_name):
    """
    Extract parameters from folder name.
    Format: run_XXXXX_gVALUE_bVALUE_aVALUE_sVALUE
    """
    pattern = r'run_(\d+)_g([\d.]+)_b([\d.]+)_a([\d.]+)_s([\d.]+)'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'run_id': match.group(1),
            'gamma': float(match.group(2)),
            'beta': float(match.group(3)),
            'alpha': float(match.group(4)),
            's0': float(match.group(5))
        }
    return None


def analyze_csv_file(csv_path):
    """
    Analyze a CSV file to extract:
    - Average consumer surplus
    - Most commonly occurred strategy combination
    """
    if not csv_path.exists():
        return None, None
    
    strategies = []
    consumer_surpluses = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get strategy combination
                strategy_a = row.get('Strategy A', '')
                strategy_b = row.get('Strategy B', '')
                strategy_combo = f"{strategy_a} vs {strategy_b}"
                strategies.append(strategy_combo)
                
                # Get consumer surplus
                cs = row.get('Consumer Surplus', '')
                if cs:
                    try:
                        consumer_surpluses.append(float(cs))
                    except ValueError:
                        pass
        
        # Calculate average consumer surplus
        avg_cs = sum(consumer_surpluses) / len(consumer_surpluses) if consumer_surpluses else None
        
        # Find most common strategy
        if strategies:
            strategy_counts = Counter(strategies)
            most_common_strategy = strategy_counts.most_common(1)[0][0]
        else:
            most_common_strategy = None
        
        return avg_cs, most_common_strategy
    
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None


def analyze_producer_surplus_csv(csv_path):
    """
    Analyze CSV file to extract producer surplus metrics.
    Producer surplus = Total Profit = Profit A + Profit B
    """
    if not csv_path.exists():
        return None, None
    
    strategies = []
    producer_surpluses = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get strategy combination
                strategy_a = row.get('Strategy A', '')
                strategy_b = row.get('Strategy B', '')
                strategy_combo = f"{strategy_a} vs {strategy_b}"
                strategies.append(strategy_combo)
                
                # Get producer surplus (Total Profit)
                total_profit = row.get('Total Profit', '')
                if total_profit:
                    try:
                        producer_surpluses.append(float(total_profit))
                    except ValueError:
                        pass
        
        # Calculate average producer surplus
        avg_ps = sum(producer_surpluses) / len(producer_surpluses) if producer_surpluses else None
        
        # Find most common strategy
        if strategies:
            strategy_counts = Counter(strategies)
            most_common_strategy = strategy_counts.most_common(1)[0][0]
        else:
            most_common_strategy = None
        
        return avg_ps, most_common_strategy
    
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None


def analyze_total_welfare_csv(csv_path):
    """
    Analyze 4_best_ne_total_welfare.csv to extract:
    - Average total welfare
    - Most commonly occurred strategy combination
    """
    if not csv_path.exists():
        return None, None
    
    strategies = []
    total_welfares = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get strategy combination
                strategy_a = row.get('Strategy A', '')
                strategy_b = row.get('Strategy B', '')
                strategy_combo = f"{strategy_a} vs {strategy_b}"
                strategies.append(strategy_combo)
                
                # Get social welfare (total welfare)
                tw = row.get('Social Welfare', '')
                if tw:
                    try:
                        total_welfares.append(float(tw))
                    except ValueError:
                        pass
        
        # Calculate average total welfare
        avg_tw = sum(total_welfares) / len(total_welfares) if total_welfares else None
        
        # Find most common strategy
        if strategies:
            strategy_counts = Counter(strategies)
            most_common_strategy = strategy_counts.most_common(1)[0][0]
        else:
            most_common_strategy = None
        
        return avg_tw, most_common_strategy
    
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None


def get_p0_from_summary(folder_path):
    """
    Extract P0 parameter from summary.json file.
    Try to read from 1x1/summary.json first.
    """
    # Try to read from 1x1 folder first
    summary_path = folder_path / "1x1" / "summary.json"
    
    if not summary_path.exists():
        # If 1x1 doesn't exist, try other folders
        for n in range(2, 11):
            summary_path = folder_path / f"{n}x{n}" / "summary.json"
            if summary_path.exists():
                break
    
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
            return data.get('parameters', {}).get('P0', None)
    except Exception as e:
        print(f"Error reading P0 from {summary_path}: {e}")
        return None


def process_single_folder(folder_path):
    """
    Process a single run folder and extract all required metrics.
    """
    folder_name = folder_path.name
    
    # Parse parameters from folder name
    params = parse_folder_name(folder_name)
    if not params:
        return None
    
    # Analyze 3_best_ne_consumer_surplus.csv
    cs_csv_path = folder_path / "3_best_ne_consumer_surplus.csv"
    avg_cs, most_common_cs_strategy = analyze_csv_file(cs_csv_path)
    
    # Analyze 4_best_ne_total_welfare.csv (for total welfare)
    tw_csv_path = folder_path / "4_best_ne_total_welfare.csv"
    avg_tw, most_common_tw_strategy = analyze_total_welfare_csv(tw_csv_path)
    
    # Analyze 4_best_ne_total_welfare.csv (for producer surplus)
    # We can use the same file since it contains Total Profit data
    avg_ps, most_common_ps_strategy = analyze_producer_surplus_csv(tw_csv_path)
    
    # Combine all results
    result = {
        'run_id': params['run_id'],
        'gamma': params['gamma'],
        'beta': params['beta'],
        'alpha': params['alpha'],
        's0': params['s0'],
        'P0': None,  # Will be populated from source data
        'avg_consumer_surplus': avg_cs,
        'most_common_strategy_consumer_surplus': most_common_cs_strategy,
        'avg_producer_surplus': avg_ps,
        'most_common_strategy_producer_surplus': most_common_ps_strategy,
        'avg_total_welfare': avg_tw,
        'most_common_strategy_total_welfare': most_common_tw_strategy
    }
    
    return result


def aggregate_all_results(batch_results_path, output_csv_path, source_data_path=None):
    """
    Aggregate results from all processed folders into a single CSV.
    
    Args:
        batch_results_path: Path containing the analysis results
        output_csv_path: Output CSV file path
        source_data_path: Path to original source data for extracting P0
    """
    batch_results_path = Path(batch_results_path)
    if source_data_path:
        source_data_path = Path(source_data_path)
    
    # Get all run folders (excluding the log file)
    run_folders = sorted([d for d in batch_results_path.iterdir() 
                         if d.is_dir() and d.name.startswith("run_")])
    
    total_folders = len(run_folders)
    print(f"Found {total_folders} folders to aggregate")
    
    # Open output CSV file
    output_csv_path = Path(output_csv_path)
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'run_id', 'gamma', 'beta', 'alpha', 's0', 'P0',
            'avg_consumer_surplus', 'most_common_strategy_consumer_surplus',
            'avg_producer_surplus', 'most_common_strategy_producer_surplus',
            'avg_total_welfare', 'most_common_strategy_total_welfare'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        start_time = time.time()
        processed = 0
        
        for idx, folder in enumerate(run_folders):
            # Progress update every 1000 folders
            if idx % 1000 == 0 and idx > 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (total_folders - idx) / rate
                print(f"Progress: {idx}/{total_folders} ({100*idx/total_folders:.1f}%) "
                      f"- Rate: {rate:.1f} folders/sec - ETA: {remaining/60:.1f} min")
            
            # Process folder
            result = process_single_folder(folder)
            
            # Get P0 from source data if available
            if result and source_data_path:
                folder_name = folder.name
                source_folder = source_data_path / folder_name
                if source_folder.exists():
                    p0 = get_p0_from_summary(source_folder)
                    result['P0'] = p0
            
            if result:
                writer.writerow(result)
                processed += 1
            
            # Flush every 5000 rows
            if idx % 5000 == 0:
                csvfile.flush()
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Aggregation complete!")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Folders processed: {processed}/{total_folders}")
        print(f"Rate: {processed/elapsed:.2f} folders/sec")
        print(f"Output saved to: {output_csv_path}")
        print(f"{'='*80}")


if __name__ == "__main__":
    batch_results_path = "/home/ab823254/data/tiered_pricing_network_economics/result_analysis/batch_results"
    source_data_path = "/home/ab823254/data/raw/tieredPricing"
    output_csv_path = "/home/ab823254/data/tiered_pricing_network_economics/result_analysis/aggregated_results.csv"
    
    print("="*80)
    print("Aggregating Results from All Runs")
    print("="*80)
    print(f"Batch results path: {batch_results_path}")
    print(f"Source data path: {source_data_path}")
    print(f"Output CSV: {output_csv_path}")
    print()
    
    aggregate_all_results(batch_results_path, output_csv_path, source_data_path)

