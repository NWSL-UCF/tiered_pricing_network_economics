"""File I/O utilities for JSON and CSV operations."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(filepath, index=False)


def parse_payoff_matrix_csv(csv_path: str) -> Tuple[List[str], Dict]:
    """Parse payoff matrix CSV and extract strategy names and profits."""
    df = pd.read_csv(csv_path, index_col=0)
    strategies = df.columns.tolist()
    payoff_matrix = {}
    
    for i, row_name in enumerate(df.index):
        for j, col_name in enumerate(df.columns):
            cell_value = df.iloc[i, j]
            if isinstance(cell_value, str) and cell_value.startswith('('):
                clean_value = cell_value.replace('np.float64(', '').replace(')', '')
                clean_value = clean_value.replace('(', '').replace(')', '')
                parts = clean_value.split(', ')
                profit_a = float(parts[0])
                profit_b = float(parts[1])
                payoff_matrix[(row_name, col_name)] = (profit_a, profit_b)
            else:
                payoff_matrix[(row_name, col_name)] = (0.0, 0.0)
    
    return strategies, payoff_matrix


def calculate_welfare_matrix(strategies: List[str], payoff_matrix: Dict, 
                           analysis_results: Dict, alpha: float) -> Dict:
    """Calculate welfare metrics for all payoff matrix cells."""
    welfare_matrix = {}
    
    for strategy_a in strategies:
        welfare_matrix[strategy_a] = {}
        for strategy_b in strategies:
            profit_a, profit_b = payoff_matrix[(strategy_a, strategy_b)]
            price_a = analysis_results['strategies'][strategy_a]['prices']
            price_b = analysis_results['strategies'][strategy_b]['prices']
            
            # Calculate welfare using the welfare analyzer
            welfare_analyzer = analysis_results['welfare_analyzer']
            welfare = welfare_analyzer.calculate_social_welfare(price_a, price_b, alpha)
            
            welfare_matrix[strategy_a][strategy_b] = {
                'social_welfare': float(welfare['total_welfare']),
                'consumer_surplus': float(welfare['consumer_surplus']),
                'producer_profit': float(welfare['producer_surplus']),
                'profit_A': float(profit_a),
                'profit_B': float(profit_b)
            }
    
    return welfare_matrix


def calculate_strategy_details(analysis_results: Dict) -> Dict:
    """Calculate detailed cost and pricing information for each strategy."""
    strategy_details = {}
    
    for strategy_name, strategy_info in analysis_results['strategies'].items():
        df_strategy = strategy_info['data']
        prices = strategy_info['prices']
        
        tier_details = {}
        for tier_id in df_strategy['tier'].unique():
            tier_data = df_strategy[df_strategy['tier'] == tier_id]
            tier_name = tier_data['tier_name'].iloc[0] if 'tier_name' in tier_data.columns else f"Tier {tier_id}"
            
            unit_cost = tier_data['cost_per_tb'].mean()
            total_cost = (tier_data['cost_per_tb'] * tier_data['demand_tb']).sum()
            unit_price = tier_data['price'].iloc[0]
            total_price = (tier_data['price'] * tier_data['demand_tb']).sum()
            
            tier_details[f"tier_{tier_id}"] = {
                'tier_name': tier_name,
                'unit_cost': float(unit_cost),
                'total_cost': float(total_cost),
                'unit_price': float(unit_price),
                'total_price': float(total_price),
                'distances': {
                    'min_distance': float(tier_data['distance'].min()),
                    'max_distance': float(tier_data['distance'].max()),
                    'avg_distance': float(tier_data['distance'].mean())
                },
                'volume': {
                    'total_volume_tb': float(tier_data['demand_tb'].sum()),
                    'num_flows': int(len(tier_data))
                }
            }
        
        overall_unit_cost = df_strategy['cost_per_tb'].mean()
        overall_total_cost = (df_strategy['cost_per_tb'] * df_strategy['demand_tb']).sum()
        overall_unit_price = df_strategy['price'].mean()
        overall_total_price = (df_strategy['price'] * df_strategy['demand_tb']).sum()
        
        strategy_details[strategy_name] = {
            'n_tiers': strategy_info['n_tiers'],
            'overall_metrics': {
                'unit_cost': float(overall_unit_cost),
                'total_cost': float(overall_total_cost),
                'unit_price': float(overall_unit_price),
                'total_price': float(overall_total_price)
            },
            'tiers': tier_details
        }
    
    return strategy_details


def create_summary_json(
    parameters: Dict,
    nash_equilibria: List[Tuple],
    welfare_analysis: Dict,
    strategies: Dict,
    output_dir: Path
) -> None:
    """Create comprehensive summary JSON file."""
    summary = {
        'parameters': parameters,
        'strategies': {
            name: {
                'n_tiers': info['n_tiers'],
                'avg_price': float(info['prices'].mean()),
                'min_price': float(info['prices'].min()),
                'max_price': float(info['prices'].max())
            }
            for name, info in strategies.items()
        },
        'nash_equilibria': [
            {
                'strategy_A': nA,
                'strategy_B': nB,
                'profit_A': float(pA),
                'profit_B': float(pB),
                'total_profit': float(pA + pB)
            }
            for nA, nB, pA, pB in nash_equilibria
        ],
        'welfare': {
            'max_total_welfare': {
                'strategy': (welfare_analysis['max_total_welfare']['strategy_A'],
                           welfare_analysis['max_total_welfare']['strategy_B']),
                'consumer_surplus': float(welfare_analysis['max_total_welfare']['consumer_surplus']),
                'producer_surplus': float(welfare_analysis['max_total_welfare']['producer_surplus']),
                'total_welfare': float(welfare_analysis['max_total_welfare']['total_welfare']),
                'efficiency': float(welfare_analysis['max_total_welfare']['efficiency_ratio'])
            },
            'max_consumer_surplus': {
                'strategy': (welfare_analysis['max_consumer_surplus']['strategy_A'],
                           welfare_analysis['max_consumer_surplus']['strategy_B']),
                'consumer_surplus': float(welfare_analysis['max_consumer_surplus']['consumer_surplus']),
                'producer_surplus': float(welfare_analysis['max_consumer_surplus']['producer_surplus']),
                'total_welfare': float(welfare_analysis['max_consumer_surplus']['total_welfare'])
            },
            'max_producer_surplus': {
                'strategy': (welfare_analysis['max_producer_surplus']['strategy_A'],
                           welfare_analysis['max_producer_surplus']['strategy_B']),
                'consumer_surplus': float(welfare_analysis['max_producer_surplus']['consumer_surplus']),
                'producer_surplus': float(welfare_analysis['max_producer_surplus']['producer_surplus']),
                'total_welfare': float(welfare_analysis['max_producer_surplus']['total_welfare'])
            }
        }
    }
    
    summary_path = output_dir / 'summary.json'
    save_json(summary, str(summary_path))
