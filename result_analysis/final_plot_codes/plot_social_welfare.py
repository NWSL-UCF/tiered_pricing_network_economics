import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Base directory
base_dir = "/home/ab823254/data/tiered_pricing_network_economics/different_alpha"

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
plt.figure(figsize=(14, 10))

# Normalize social welfare for each alpha separately and plot
for alpha in sorted(all_data.keys()):
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
    
    # Plot line
    plt.plot(tiers, sw_normalized, marker='o', label=f'α={alpha:.2f}', alpha=0.7, linewidth=2)
    
    # Mark Nash equilibria with stars
    nash_indices = data['nash_indices']
    for nash_tier in nash_indices:
        if nash_tier in tiers:
            idx = tiers.index(nash_tier)
            plt.scatter(nash_tier, sw_normalized[idx], marker='*', s=500, 
                       edgecolors='red', linewidths=2, zorder=5)

# Formatting
plt.xlabel('Number of Tiers', fontsize=14, fontweight='bold')
plt.ylabel('Normalized Social Welfare', fontsize=14, fontweight='bold')
plt.title('Social Welfare vs Number of Tiers (Diagonal Values)\nNash Equilibria marked with ★', 
         fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
plt.xticks(range(1, 11), ['1x1', '2x2', '3x3', '4x4', '5x5', '6x6', '7x7', '8x8', '9x9', '10x10'])
plt.tight_layout()

# Save the plot
output_path = os.path.join(base_dir, 'social_welfare_vs_tiers.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also create a summary of Nash equilibria
print("\nNash Equilibria Summary:")
print("-" * 60)
for alpha in sorted(all_data.keys()):
    nash_indices = all_data[alpha]['nash_indices']
    if nash_indices:
        nash_str = ", ".join([f"{n}x{n}" for n in nash_indices])
        print(f"α={alpha:.2f}: {nash_str}")
    else:
        print(f"α={alpha:.2f}: No diagonal Nash equilibria")

plt.show()

