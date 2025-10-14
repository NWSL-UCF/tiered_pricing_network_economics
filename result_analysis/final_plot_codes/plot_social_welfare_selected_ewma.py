import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Base directory
base_dir = "/home/ab823254/data/tiered_pricing_network_economics/different_alpha"

# Selected alpha values to plot
selected_alphas = [0.77, 1.27, 1.77, 2.27, 2.93]

# EWMA parameters
ewma_span = 3  # Smoothing parameter (higher = more smoothing)

# Get all run folders
run_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("run_")])

# Dictionary to store data for each run
all_data = {}

for run_folder in run_folders:
    # Extract alpha value from folder name
    # Format: run_XXXXX_g0.0010_b0.30_aX.XX_s0.35
    parts = run_folder.split("_")
    alpha_str = [p for p in parts if p.startswith("a")][0]
    alpha = float(alpha_str[1:])
    
    # Only process selected alpha values
    if alpha not in selected_alphas:
        continue
    
    # Path to welfare_matrix.json and summary.json
    welfare_path = os.path.join(base_dir, run_folder, "10x10", "welfare_matrix.json")
    summary_path = os.path.join(base_dir, run_folder, "10x10", "summary.json")
    
    # Read welfare matrix
    with open(welfare_path, 'r') as f:
        welfare_data = json.load(f)
    
    # Read summary for Nash equilibria
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    # Extract diagonal social welfare values
    strategies = ["Flat", "2-Tier", "3-Tier", "4-Tier", "5-Tier", 
                  "6-Tier", "7-Tier", "8-Tier", "9-Tier", "10-Tier"]
    
    tiers = []
    social_welfare = []
    
    for i, strategy in enumerate(strategies, 1):
        if strategy in welfare_data["welfare_matrix"]:
            if strategy in welfare_data["welfare_matrix"][strategy]:
                sw = welfare_data["welfare_matrix"][strategy][strategy]["social_welfare"]
                tiers.append(i)
                social_welfare.append(sw)
    
    # Find Nash equilibria on the diagonal
    nash_indices = []
    nash_equilibria = summary_data.get("nash_equilibria", [])
    
    for ne in nash_equilibria:
        if ne["strategy_A"] == ne["strategy_B"]:
            # Extract tier number from strategy name
            if ne["strategy_A"] == "Flat":
                tier_num = 1
            else:
                tier_num = int(ne["strategy_A"].split("-")[0])
            nash_indices.append(tier_num)
    
    all_data[alpha] = {
        'tiers': tiers,
        'social_welfare': social_welfare,
        'nash_indices': nash_indices
    }

# Plot the results
plt.figure(figsize=(12, 8))

# Color palette for the lines
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Normalize social welfare for each alpha separately and plot
for i, alpha in enumerate(sorted(all_data.keys())):
    data = all_data[alpha]
    tiers = data['tiers']
    sw = np.array(data['social_welfare'])
    
    # Normalize social welfare to [0, 1] range for this alpha
    sw_min = sw.min()
    sw_max = sw.max()
    if sw_max - sw_min > 0:
        sw_normalized = (sw - sw_min) / (sw_max - sw_min)
    else:
        sw_normalized = np.zeros_like(sw)
    
    # Apply EWMA smoothing
    sw_series = pd.Series(sw_normalized)
    sw_ewma = sw_series.ewm(span=ewma_span, adjust=False).mean().values
    
    # Plot original data with lower alpha (more transparent)
    plt.plot(tiers, sw_normalized, marker='o', label=f'α = {alpha:.2f} (original)', 
             alpha=0.3, linewidth=1, markersize=5, color=colors[i], linestyle='--')
    
    # Plot EWMA smoothed line
    plt.plot(tiers, sw_ewma, label=f'α = {alpha:.2f} (EWMA)', 
             alpha=0.9, linewidth=3, color=colors[i])
    
    # Mark Nash equilibria with stars on the EWMA line
    nash_indices = data['nash_indices']
    for nash_tier in nash_indices:
        if nash_tier in tiers:
            idx = tiers.index(nash_tier)
            plt.scatter(nash_tier, sw_ewma[idx], marker='*', s=600, 
                       edgecolors='red', linewidths=2.5, zorder=5, facecolors=colors[i])

# Formatting
plt.xlabel('Number of Tiers', fontsize=16, fontweight='bold')
plt.ylabel('Normalized Social Welfare', fontsize=16, fontweight='bold')
plt.title('Social Welfare vs Number of Tiers (EWMA Smoothed)\nNash Equilibria marked with ★', 
         fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
plt.legend(fontsize=10, frameon=True, shadow=True, loc='best', ncol=2)
plt.xticks(range(1, 11), ['1x1', '2x2', '3x3', '4x4', '5x5', '6x6', '7x7', '8x8', '9x9', '10x10'], fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save the plot
output_path = os.path.join(base_dir, 'social_welfare_vs_tiers_selected_ewma.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also create a summary of Nash equilibria
print(f"\nEWMA Smoothing Applied (span={ewma_span})")
print("\nNash Equilibria Summary (Selected Alpha Values):")
print("-" * 60)
for alpha in sorted(all_data.keys()):
    nash_indices = all_data[alpha]['nash_indices']
    if nash_indices:
        nash_str = ", ".join([f"{n}x{n}" for n in nash_indices])
        print(f"α = {alpha:.2f}: {nash_str}")
    else:
        print(f"α = {alpha:.2f}: No diagonal Nash equilibria")

plt.show()

