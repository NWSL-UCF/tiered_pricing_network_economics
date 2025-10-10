"""File I/O utilities for JSON and CSV operations."""

import json
import pandas as pd
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
