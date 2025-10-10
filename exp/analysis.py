#!/usr/bin/env python3
"""
Script to analyze results from parameter sweep
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np


def load_summary_data(base_dir: str) -> pd.DataFrame:
    """
    Load and combine summary data from all successful runs.
    
    Args:
        base_dir: Base directory containing all run results
        
    Returns:
        DataFrame with combined results
    """
    base_path = Path(base_dir)
    results = []
    
    # Find all summary.json files
    for summary_file in base_path.rglob('summary.json'):
        if summary_file.name == 'parameter_sweep_summary.json':
            continue  # Skip the main summary file
            
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            result = {
                'run_id': data['metadata']['run_id'],
                'P0': data['parameters']['P0'],
                'gamma': data['parameters']['gamma'],
                'beta': data['parameters']['beta'],
                'alpha': data['parameters']['alpha'],
                's0': data['parameters']['s0'],
                'nash_count': data['nash_equilibria']['count'],
                'avg_consumer_surplus': data['nash_equilibria']['averages']['consumer_surplus'],
                'avg_producer_surplus': data['nash_equilibria']['averages']['producer_surplus'],
                'avg_total_welfare': data['nash_equilibria']['averages']['total_welfare'],
                'max_welfare': data['optimal_outcomes']['max_total_welfare']['total_welfare'],
                'welfare_gap': data['welfare_comparison']['ne_vs_optimal']['welfare_gap']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
    
    return pd.DataFrame(results)


def create_analysis_plots(df: pd.DataFrame, output_dir: str):
    """
    Create analysis plots for the parameter sweep results.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Nash Equilibrium count by parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Nash Equilibrium Count by Parameters', fontsize=16, fontweight='bold')
    
    params = ['P0', 'gamma', 'beta', 'alpha', 's0']
    for i, param in enumerate(params):
        ax = axes[i//3, i%3]
        df.groupby(param)['nash_count'].mean().plot(kind='bar', ax=ax)
        ax.set_title(f'Nash Count vs {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Average Nash Count')
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_path / 'nash_count_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Welfare analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Welfare Analysis by Parameters', fontsize=16, fontweight='bold')
    
    # Consumer surplus vs s0
    df.plot.scatter(x='s0', y='avg_consumer_surplus', ax=axes[0, 0], alpha=0.6)
    axes[0, 0].set_title('Consumer Surplus vs Competition Intensity (s0)')
    axes[0, 0].set_xlabel('s0')
    axes[0, 0].set_ylabel('Average Consumer Surplus')
    
    # Producer surplus vs s0
    df.plot.scatter(x='s0', y='avg_producer_surplus', ax=axes[0, 1], alpha=0.6)
    axes[0, 1].set_title('Producer Surplus vs Competition Intensity (s0)')
    axes[0, 1].set_xlabel('s0')
    axes[0, 1].set_ylabel('Average Producer Surplus')
    
    # Total welfare vs alpha
    df.plot.scatter(x='alpha', y='avg_total_welfare', ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title('Total Welfare vs Price Elasticity (α)')
    axes[1, 0].set_xlabel('α')
    axes[1, 0].set_ylabel('Average Total Welfare')
    
    # Welfare gap vs s0
    df.plot.scatter(x='s0', y='welfare_gap', ax=axes[1, 1], alpha=0.6)
    axes[1, 1].set_title('Welfare Gap vs Competition Intensity (s0)')
    axes[1, 1].set_xlabel('s0')
    axes[1, 1].set_ylabel('Welfare Gap')
    
    plt.tight_layout()
    plt.savefig(output_path / 'welfare_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Parameter correlation heatmap
    numeric_cols = ['P0', 'gamma', 'beta', 'alpha', 's0', 'nash_count', 
                   'avg_consumer_surplus', 'avg_producer_surplus', 'avg_total_welfare']
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to analyze parameter sweep results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing parameter sweep results')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory to save analysis plots')
    
    args = parser.parse_args()
    
    print("Loading parameter sweep results...")
    df = load_summary_data(args.results_dir)
    
    print(f"Loaded {len(df)} successful runs")
    print(f"Parameter ranges:")
    for col in ['P0', 'gamma', 'beta', 'alpha', 's0']:
        print(f"  {col}: {df[col].min():.3f} - {df[col].max():.3f}")
    
    print("Creating analysis plots...")
    create_analysis_plots(df, args.output_dir)
    
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    
    # Save combined results to CSV
    output_path = Path(args.output_dir)
    df.to_csv(output_path / 'combined_results.csv', index=False)
    print(f"Combined results saved to: {output_path / 'combined_results.csv'}")


if __name__ == "__main__":
    main()