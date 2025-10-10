#!/usr/bin/env python3
"""
Generate PDF visualization of pricing tiers for each strategy
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_tier_prices(data_path, tier_config_path, P0, gamma, beta, alpha, s0):
    """Calculate tier prices for a given parameter configuration."""
    
    # Load data
    df = pd.read_csv(data_path)
    df = df[df['distance'] > 0].reset_index(drop=True)
    df['demand_tb'] = df['demand'] / 1e6  # Convert Mb to TB
    
    # Calculate costs
    df['cost_per_tb'] = gamma * df['distance'] + beta
    
    # Calculate valuations
    df['v'] = P0 * (df['demand_tb'] ** (1 / alpha))
    
    # Load tier configurations
    with open(tier_config_path, 'r') as f:
        tier_config = json.load(f)
    
    # Calculate prices for each tier strategy
    results = []
    
    for n_tiers in range(1, 8):
        tier_key = f"{n_tiers}_tier"
        strategy_name = tier_config[tier_key]['strategy_name']
        tiers_info = tier_config[tier_key]['tiers']
        
        # Assign tiers based on distance percentiles
        if n_tiers == 1:
            df_temp = df.copy()
            df_temp['tier'] = 0
            df_temp['tier_name'] = 'Universal'
        else:
            df_temp = df.copy()
            df_temp['tier'] = pd.qcut(df_temp['distance'], q=n_tiers, labels=False, duplicates='drop')
            df_temp['tier'] = df_temp['tier'].astype(int)
            tier_name_map = {tier['tier_id']: tier['name'] for tier in tiers_info}
            df_temp['tier_name'] = df_temp['tier'].map(tier_name_map)
        
        # Calculate tier prices
        for tier_id in sorted(df_temp['tier'].unique()):
            sub = df_temp[df_temp['tier'] == tier_id]
            
            # Bundle valuation calculation
            max_v = sub['v'].max()
            weights = np.exp(alpha * (sub['v'] - max_v))
            
            # Weighted average cost
            c_b = (sub['cost_per_tb'] * weights).sum() / weights.sum()
            
            # Price = cost + markup
            markup = 1 / (alpha * s0)
            p_b = c_b + markup
            
            # Distance range
            dist_min = sub['distance'].min()
            dist_max = sub['distance'].max()
            dist_mean = sub['distance'].mean()
            volume = sub['demand_tb'].sum()
            flow_count = len(sub)
            
            tier_name = sub['tier_name'].iloc[0]
            
            results.append({
                'Strategy': strategy_name,
                'n_tiers': n_tiers,
                'Tier_ID': tier_id,
                'Tier': tier_name,
                'Distance_Min': dist_min,
                'Distance_Max': dist_max,
                'Distance_Avg': dist_mean,
                'Price': p_b,
                'Cost': c_b,
                'Volume': volume,
                'Flow_Count': flow_count
            })
    
    return pd.DataFrame(results)


def create_pricing_visualization(pricing_df, parameters, output_path):
    """Create comprehensive PDF visualization of pricing strategies."""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # Main title
    param_str = f"P0=${parameters['P0']}, γ={parameters['gamma']}, β={parameters['beta']}, α={parameters['alpha']}, s0={parameters['s0']}"
    fig.suptitle(f'Transit ISP Pricing Strategies\n{param_str}', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Create 7 subplots (one for each strategy)
    strategies = ['Flat', '2-Tier', '3-Tier', '4-Tier', '5-Tier', '6-Tier', '7-Tier']
    
    for idx, strategy in enumerate(strategies):
        ax = plt.subplot(4, 2, idx + 1)
        
        strategy_data = pricing_df[pricing_df['Strategy'] == strategy].sort_values('Tier_ID')
        
        if strategy_data.empty:
            continue
        
        # Create bar chart showing prices
        x_pos = np.arange(len(strategy_data))
        prices = strategy_data['Price'].values
        costs = strategy_data['Cost'].values
        tier_names = strategy_data['Tier'].values
        
        # Plot bars
        bars1 = ax.bar(x_pos - 0.2, prices, 0.4, label='Price/TB', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x_pos + 0.2, costs, 0.4, label='Cost/TB', color='#e74c3c', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Tier', fontsize=12, fontweight='bold')
        ax.set_ylabel('$/TB', fontsize=12, fontweight='bold')
        ax.set_title(f'{strategy}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.2f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Add distance range annotations
        for i, (_, row) in enumerate(strategy_data.iterrows()):
            dist_text = f"{row['Distance_Min']:.0f}-{row['Distance_Max']:.0f} mi"
            ax.text(i, -0.05 * ax.get_ylim()[1], dist_text, 
                   ha='center', va='top', fontsize=7, style='italic',
                   transform=ax.get_xaxis_transform())
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Pricing visualization saved to: {output_path}")
    plt.close()


def create_pricing_table_pdf(pricing_df, parameters, output_path):
    """Create a detailed table-based PDF."""
    
    fig, axes = plt.subplots(7, 1, figsize=(16, 20))
    fig.suptitle(f'Detailed Pricing Tables\nP0=${parameters["P0"]}, γ={parameters["gamma"]}, β={parameters["beta"]}, α={parameters["alpha"]}, s0={parameters["s0"]}',
                 fontsize=16, fontweight='bold')
    
    strategies = ['Flat', '2-Tier', '3-Tier', '4-Tier', '5-Tier', '6-Tier', '7-Tier']
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        ax.axis('tight')
        ax.axis('off')
        
        strategy_data = pricing_df[pricing_df['Strategy'] == strategy].sort_values('Tier_ID')
        
        if strategy_data.empty:
            continue
        
        # Prepare table data
        table_data = []
        for _, row in strategy_data.iterrows():
            table_data.append([
                row['Tier'],
                f"{row['Distance_Min']:.0f}",
                f"{row['Distance_Max']:.0f}",
                f"{row['Distance_Avg']:.0f}",
                f"${row['Cost']:.2f}",
                f"${row['Price']:.2f}",
                f"{row['Volume']:,.0f}",
                f"{row['Flow_Count']:,}"
            ])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Tier Name', 'Min Dist\n(mi)', 'Max Dist\n(mi)', 'Avg Dist\n(mi)', 
                                  'Cost\n($/TB)', 'Price\n($/TB)', 'Volume\n(TB)', 'Flows'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(8):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(8):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        # Add title
        ax.text(0.5, 1.05, f'{strategy} Pricing', 
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Pricing table saved to: {output_path}")
    plt.close()


def create_comparison_chart(pricing_df, parameters, output_path):
    """Create comparison chart across all strategies."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Pricing Strategy Comparison\nP0=${parameters["P0"]}, γ={parameters["gamma"]}, β={parameters["beta"]}, α={parameters["alpha"]}, s0={parameters["s0"]}',
                 fontsize=16, fontweight='bold')
    
    # 1. Average price by strategy
    ax = axes[0, 0]
    avg_prices = pricing_df.groupby('Strategy')['Price'].mean()
    strategies_order = ['Flat', '2-Tier', '3-Tier', '4-Tier', '5-Tier', '6-Tier', '7-Tier']
    avg_prices = avg_prices.reindex(strategies_order)
    bars = ax.bar(range(len(avg_prices)), avg_prices.values, color='#3498db', alpha=0.7)
    ax.set_xticks(range(len(avg_prices)))
    ax.set_xticklabels(avg_prices.index, rotation=45, ha='right')
    ax.set_ylabel('Average Price ($/TB)', fontweight='bold')
    ax.set_title('Average Price per Strategy', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Price range by strategy
    ax = axes[0, 1]
    for strategy in strategies_order:
        data = pricing_df[pricing_df['Strategy'] == strategy]
        if not data.empty:
            prices = data['Price'].values
            ax.scatter([strategy] * len(prices), prices, alpha=0.6, s=100)
    ax.set_ylabel('Price ($/TB)', fontweight='bold')
    ax.set_title('Price Distribution by Strategy', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Number of tiers vs markup
    ax = axes[1, 0]
    markup_by_strategy = []
    for strategy in strategies_order:
        data = pricing_df[pricing_df['Strategy'] == strategy]
        if not data.empty:
            avg_markup = (data['Price'] - data['Cost']).mean()
            markup_by_strategy.append(avg_markup)
        else:
            markup_by_strategy.append(0)
    
    ax.plot(range(len(strategies_order)), markup_by_strategy, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xticks(range(len(strategies_order)))
    ax.set_xticklabels(strategies_order, rotation=45, ha='right')
    ax.set_ylabel('Average Markup ($/TB)', fontweight='bold')
    ax.set_title('Markup by Pricing Complexity', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Price vs Distance relationship
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, 7))
    for i, strategy in enumerate(strategies_order):
        data = pricing_df[pricing_df['Strategy'] == strategy].sort_values('Distance_Avg')
        if not data.empty:
            ax.plot(data['Distance_Avg'], data['Price'], marker='o', label=strategy, color=colors[i], linewidth=2)
    
    ax.set_xlabel('Average Distance (miles)', fontweight='bold')
    ax.set_ylabel('Price ($/TB)', fontweight='bold')
    ax.set_title('Price vs Distance by Strategy', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Comparison chart saved to: {output_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Paths
    data_path = "netflow_grouped_by_src_dst.csv"
    tier_config_path = "tier_strategies.json"
    
    # Consumer-optimal parameters
    parameters = {
        "P0": 4.0,
        "gamma": 0.025,
        "beta": 0.3,
        "alpha": 1.0,
        "s0": 0.05
    }
    
    print("="*80)
    print("PRICING STRATEGY PDF GENERATION")
    print("="*80)
    print(f"\nParameters (Consumer-Optimal):")
    for key, val in parameters.items():
        print(f"  {key}: {val}")
    print("="*80)
    
    # Calculate prices
    print("\nCalculating tier prices...")
    pricing_df = calculate_tier_prices(data_path, tier_config_path, **parameters)
    
    # Generate visualizations
    print("\nGenerating PDF visualizations...")
    create_pricing_visualization(pricing_df, parameters, 'pricing_strategies_charts.pdf')
    create_pricing_table_pdf(pricing_df, parameters, 'pricing_strategies_tables.pdf')
    create_comparison_chart(pricing_df, parameters, 'pricing_strategies_comparison.pdf')
    
    # Also save CSV
    pricing_df.to_csv('pricing_strategies_data.csv', index=False)
    print(f"✅ Data saved to: pricing_strategies_data.csv")
    
    print("\n" + "="*80)
    print("ALL FILES GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. pricing_strategies_charts.pdf      - Bar charts for each strategy")
    print("  2. pricing_strategies_tables.pdf      - Detailed tables for each strategy")
    print("  3. pricing_strategies_comparison.pdf  - Cross-strategy comparisons")
    print("  4. pricing_strategies_data.csv        - Raw data")
    print("="*80)