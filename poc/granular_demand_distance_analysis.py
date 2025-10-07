#!/usr/bin/env python3
"""
Granular analysis of demand distribution vs distance with fine-grained bins
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def granular_demand_distance_analysis():
    """Perform granular analysis of demand vs distance"""
    
    # Load data
    df = pd.read_csv("netflow_grouped_by_src_dst.csv")
    
    # Clean data and convert to km
    df_clean = df[df['distance'] > 0].copy()
    df_clean['distance_km'] = df_clean['distance'] * 1.609344
    
    print("🔍 GRANULAR DEMAND vs DISTANCE ANALYSIS")
    print("=" * 70)
    
    # Create very fine-grained distance bins (every 100km up to 2000km, then larger bins)
    fine_bins = list(range(0, 2000, 100)) + [2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000]
    df_clean['fine_distance_bin'] = pd.cut(df_clean['distance_km'], bins=fine_bins)
    
    # Calculate detailed statistics for each bin
    bin_analysis = df_clean.groupby('fine_distance_bin', observed=True).agg({
        'demand': ['count', 'mean', 'median', 'std', 'min', 'max', 'sum'],
        'flow_count': ['mean', 'median', 'sum'],
        'distance_km': 'mean'
    }).round(2)
    
    bin_analysis.columns = ['_'.join(col).strip() for col in bin_analysis.columns]
    bin_analysis = bin_analysis.reset_index()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Fine-grained bar chart of average demand
    ax1 = plt.subplot(3, 3, 1)
    valid_bins = bin_analysis[bin_analysis['demand_count'] > 0]
    x_pos = range(len(valid_bins))
    
    bars = ax1.bar(x_pos, valid_bins['demand_mean'], alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('Average Demand by Fine Distance Bins', fontweight='bold')
    ax1.set_xlabel('Distance Bins')
    ax1.set_ylabel('Average Demand (Mb)')
    ax1.set_xticks(x_pos[::3])  # Show every 3rd label to avoid crowding
    ax1.set_xticklabels([str(valid_bins.iloc[i]['fine_distance_bin']) for i in range(0, len(valid_bins), 3)], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Number of connections per distance bin
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(x_pos, valid_bins['demand_count'], alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax2.set_title('Number of IP Pairs by Distance', fontweight='bold')
    ax2.set_xlabel('Distance Bins')
    ax2.set_ylabel('Number of IP Pairs')
    ax2.set_xticks(x_pos[::3])
    ax2.set_xticklabels([str(valid_bins.iloc[i]['fine_distance_bin']) for i in range(0, len(valid_bins), 3)], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total demand per distance bin
    ax3 = plt.subplot(3, 3, 3)
    ax3.bar(x_pos, valid_bins['demand_sum'], alpha=0.7, color='orange', edgecolor='darkorange')
    ax3.set_title('Total Demand by Distance', fontweight='bold')
    ax3.set_xlabel('Distance Bins')
    ax3.set_ylabel('Total Demand (Mb)')
    ax3.set_xticks(x_pos[::3])
    ax3.set_xticklabels([str(valid_bins.iloc[i]['fine_distance_bin']) for i in range(0, len(valid_bins), 3)], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Demand variability (standard deviation)
    ax4 = plt.subplot(3, 3, 4)
    ax4.bar(x_pos, valid_bins['demand_std'], alpha=0.7, color='red', edgecolor='darkred')
    ax4.set_title('Demand Variability (Std Dev)', fontweight='bold')
    ax4.set_xlabel('Distance Bins')
    ax4.set_ylabel('Standard Deviation (Mb)')
    ax4.set_xticks(x_pos[::3])
    ax4.set_xticklabels([str(valid_bins.iloc[i]['fine_distance_bin']) for i in range(0, len(valid_bins), 3)], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Median vs Mean demand comparison
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(x_pos, valid_bins['demand_mean'], 'o-', label='Mean', linewidth=2, markersize=4)
    ax5.plot(x_pos, valid_bins['demand_median'], 's-', label='Median', linewidth=2, markersize=4)
    ax5.set_title('Mean vs Median Demand', fontweight='bold')
    ax5.set_xlabel('Distance Bins')
    ax5.set_ylabel('Demand (Mb)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(x_pos[::5])
    
    # 6. Average flow count per distance
    ax6 = plt.subplot(3, 3, 6)
    ax6.bar(x_pos, valid_bins['flow_count_mean'], alpha=0.7, color='purple', edgecolor='darkviolet')
    ax6.set_title('Average Flow Count by Distance', fontweight='bold')
    ax6.set_xlabel('Distance Bins')
    ax6.set_ylabel('Average Flow Count')
    ax6.set_xticks(x_pos[::3])
    ax6.set_xticklabels([str(valid_bins.iloc[i]['fine_distance_bin']) for i in range(0, len(valid_bins), 3)], rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # 7. Scatter plot with trend line
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(df_clean['distance_km'], df_clean['demand'], alpha=0.3, s=5)
    
    # Add trend line
    z = np.polyfit(df_clean['distance_km'], df_clean['demand'], 1)
    p = np.poly1d(z)
    ax7.plot(df_clean['distance_km'], p(df_clean['distance_km']), "r--", alpha=0.8, linewidth=2)
    
    ax7.set_title('Demand vs Distance with Trend', fontweight='bold')
    ax7.set_xlabel('Distance (km)')
    ax7.set_ylabel('Demand (Mb)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Box plot for selected distance ranges
    ax8 = plt.subplot(3, 3, 8)
    
    # Create broader categories for box plot
    broad_bins = [0, 500, 1000, 2000, 5000, 10000, 20000]
    df_clean['broad_bin'] = pd.cut(df_clean['distance_km'], bins=broad_bins, 
                                   labels=['0-500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '10000+'])
    
    box_data = [df_clean[df_clean['broad_bin'] == cat]['demand'].values for cat in df_clean['broad_bin'].cat.categories if len(df_clean[df_clean['broad_bin'] == cat]) > 0]
    box_labels = [cat for cat in df_clean['broad_bin'].cat.categories if len(df_clean[df_clean['broad_bin'] == cat]) > 0]
    
    ax8.boxplot(box_data, labels=box_labels)
    ax8.set_title('Demand Distribution by Distance Range', fontweight='bold')
    ax8.set_xlabel('Distance Range (km)')
    ax8.set_ylabel('Demand (Mb)')
    ax8.set_yscale('log')
    plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')
    ax8.grid(True, alpha=0.3)
    
    # 9. Cumulative demand distribution
    ax9 = plt.subplot(3, 3, 9)
    sorted_distances = valid_bins['distance_km_mean'].values
    cumulative_demand = np.cumsum(valid_bins['demand_sum'].values)
    total_demand = cumulative_demand[-1]
    cumulative_percentage = (cumulative_demand / total_demand) * 100
    
    ax9.plot(sorted_distances, cumulative_percentage, 'o-', linewidth=2, markersize=4)
    ax9.set_title('Cumulative Demand Distribution', fontweight='bold')
    ax9.set_xlabel('Distance (km)')
    ax9.set_ylabel('Cumulative Demand (%)')
    ax9.grid(True, alpha=0.3)
    ax9.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50%')
    ax9.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80%')
    ax9.legend()
    
    plt.tight_layout(pad=2.0)
    plt.savefig('figure/granular_demand_distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\n📊 DETAILED STATISTICS BY DISTANCE BINS:")
    print("-" * 70)
    print(f"{'Distance Range':<15} {'Count':<8} {'Mean':<10} {'Median':<8} {'Std':<10} {'Total':<12} {'Avg Flows':<10}")
    print("-" * 70)
    
    for _, row in valid_bins.iterrows():
        if row['demand_count'] >= 5:  # Only show bins with at least 5 data points
            dist_range = str(row['fine_distance_bin'])[:13]
            count = int(row['demand_count'])
            mean_demand = row['demand_mean']
            median_demand = row['demand_median']
            std_demand = row['demand_std']
            total_demand = row['demand_sum']
            avg_flows = row['flow_count_mean']
            
            print(f"{dist_range:<15} {count:<8} {mean_demand:<10.1f} {median_demand:<8.1f} {std_demand:<10.1f} {total_demand:<12.1f} {avg_flows:<10.1f}")
    
    # Calculate and print key insights
    print(f"\n🎯 KEY INSIGHTS:")
    print("-" * 70)
    
    # Find peak demand distance ranges
    top_3_mean = valid_bins.nlargest(3, 'demand_mean')[['fine_distance_bin', 'demand_mean', 'demand_count']]
    print(f"\nTop 3 distance ranges by AVERAGE demand:")
    for _, row in top_3_mean.iterrows():
        print(f"  • {row['fine_distance_bin']}: {row['demand_mean']:.1f} Mb avg ({int(row['demand_count'])} pairs)")
    
    # Find highest total demand ranges
    top_3_total = valid_bins.nlargest(3, 'demand_sum')[['fine_distance_bin', 'demand_sum', 'demand_count']]
    print(f"\nTop 3 distance ranges by TOTAL demand:")
    for _, row in top_3_total.iterrows():
        print(f"  • {row['fine_distance_bin']}: {row['demand_sum']:,.1f} Mb total ({int(row['demand_count'])} pairs)")
    
    # Find most active distance ranges (by number of pairs)
    top_3_count = valid_bins.nlargest(3, 'demand_count')[['fine_distance_bin', 'demand_count', 'demand_mean']]
    print(f"\nTop 3 most active distance ranges (by number of IP pairs):")
    for _, row in top_3_count.iterrows():
        print(f"  • {row['fine_distance_bin']}: {int(row['demand_count'])} pairs (avg {row['demand_mean']:.1f} Mb)")
    
    # Calculate distance-based percentiles
    print(f"\n📈 DISTANCE-BASED DEMAND PERCENTILES:")
    print("-" * 70)
    for percentile in [50, 75, 90, 95, 99]:
        threshold = np.percentile(df_clean['demand'], percentile)
        high_demand_pairs = df_clean[df_clean['demand'] >= threshold]
        avg_distance = high_demand_pairs['distance_km'].mean()
        print(f"Top {100-percentile:2d}% demand pairs (≥{threshold:.1f} Mb): avg distance {avg_distance:.1f} km")

if __name__ == "__main__":
    granular_demand_distance_analysis()
