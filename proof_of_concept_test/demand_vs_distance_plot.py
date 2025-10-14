#!/usr/bin/env python3
"""
Simple script to plot demand distribution vs distance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_demand_vs_distance():
    """Plot demand distribution against distance"""
    
    # Load data
    df = pd.read_csv("netflow_grouped_by_src_dst.csv")
    
    # Remove zero distances and convert miles to km for better interpretation
    df_clean = df[df['distance'] > 0].copy()
    df_clean['distance_km'] = df_clean['distance'] * 1.609344  # Convert miles to km
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Demand vs Distance Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Demand vs Distance
    axes[0, 0].scatter(df_clean['distance_km'], df_clean['demand'], alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Distance (km)')
    axes[0, 0].set_ylabel('Demand (Mb)')
    axes[0, 0].set_title('Demand vs Distance (Scatter Plot)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log-scale scatter plot
    axes[0, 1].scatter(df_clean['distance_km'], df_clean['demand'], alpha=0.5, s=10)
    axes[0, 1].set_xlabel('Distance (km)')
    axes[0, 1].set_ylabel('Demand (Mb)')
    axes[0, 1].set_title('Demand vs Distance (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Binned analysis: Average demand by distance ranges
    distance_bins = [0, 500, 1000, 2000, 5000, 10000, 20000]
    df_clean['distance_bin'] = pd.cut(df_clean['distance_km'], bins=distance_bins)
    
    bin_stats = df_clean.groupby('distance_bin').agg({
        'demand': ['mean', 'median', 'count']
    }).round(2)
    
    bin_centers = [(distance_bins[i] + distance_bins[i+1])/2 for i in range(len(distance_bins)-1)]
    mean_demands = bin_stats['demand']['mean'].values
    
    axes[1, 0].bar(range(len(bin_centers)), mean_demands, alpha=0.7)
    axes[1, 0].set_xlabel('Distance Range')
    axes[1, 0].set_ylabel('Average Demand (Mb)')
    axes[1, 0].set_title('Average Demand by Distance Range')
    axes[1, 0].set_xticks(range(len(bin_centers)))
    axes[1, 0].set_xticklabels([f'{distance_bins[i]}-{distance_bins[i+1]}' for i in range(len(distance_bins)-1)], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mean_demands):
        if not np.isnan(v):
            axes[1, 0].text(i, v + max(mean_demands)*0.01, f'{v:.1f}', ha='center', va='bottom')
    
    # 4. Hexbin plot for density
    hb = axes[1, 1].hexbin(df_clean['distance_km'], df_clean['demand'], gridsize=30, cmap='Blues', mincnt=1)
    axes[1, 1].set_xlabel('Distance (km)')
    axes[1, 1].set_ylabel('Demand (Mb)')
    axes[1, 1].set_title('Demand vs Distance (Density Plot)')
    plt.colorbar(hb, ax=axes[1, 1], label='Count')
    
    plt.tight_layout()
    plt.savefig('figure/demand_vs_distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("DEMAND vs DISTANCE SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\nOverall Statistics:")
    print(f"Total IP pairs: {len(df_clean):,}")
    print(f"Distance range: {df_clean['distance_km'].min():.1f} - {df_clean['distance_km'].max():.1f} km")
    print(f"Demand range: {df_clean['demand'].min():.1f} - {df_clean['demand'].max():.1f} Mb")
    
    print(f"\nCorrelation Analysis:")
    correlation = df_clean['distance_km'].corr(df_clean['demand'])
    print(f"Pearson correlation (Distance vs Demand): {correlation:.4f}")
    
    print(f"\nDemand by Distance Ranges:")
    print("-" * 50)
    for i, (bin_range, stats) in enumerate(bin_stats.iterrows()):
        count = int(stats['demand']['count'])
        mean_demand = stats['demand']['mean']
        median_demand = stats['demand']['median']
        if count > 0:
            print(f"{bin_range}: {count:6,} pairs | Mean: {mean_demand:8.1f} Mb | Median: {median_demand:6.1f} Mb")

if __name__ == "__main__":
    plot_demand_vs_distance()
