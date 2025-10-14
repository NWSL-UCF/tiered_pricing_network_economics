import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Base directory
base_dir = "/home/ab823254/data/tiered_pricing_network_economics/different_s0"

# Get all run folders and sort them
run_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("run_")])

# Select every 3rd folder (1st, 4th, 7th, 10th, ...)
selected_folders = [run_folders[i] for i in range(0, len(run_folders), 3)]

print(f"Selected {len(selected_folders)} folders:")
for folder in selected_folders:
    print(f"  {folder}")

# Prepare data for all selected folders
heatmap_data = []

for run_folder in selected_folders:
    # Extract s0 value from folder name
    parts = run_folder.split("_")
    s0_str = [p for p in parts if p.startswith("s")][0]
    s0 = float(s0_str[1:])
    
    # Path to welfare_matrix.json and summary.json
    welfare_path = os.path.join(base_dir, run_folder, "10x10", "welfare_matrix.json")
    summary_path = os.path.join(base_dir, run_folder, "10x10", "summary.json")
    
    # Read welfare matrix
    with open(welfare_path, 'r') as f:
        welfare_data = json.load(f)
    
    # Read summary for Nash equilibria
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    # Extract producer surplus matrix
    strategies = ["Flat", "2-Tier", "3-Tier", "4-Tier", "5-Tier", 
                  "6-Tier", "7-Tier", "8-Tier", "9-Tier", "10-Tier"]
    
    matrix = np.zeros((10, 10))
    for i, strat_A in enumerate(strategies):
        for j, strat_B in enumerate(strategies):
            if strat_A in welfare_data["welfare_matrix"] and strat_B in welfare_data["welfare_matrix"][strat_A]:
                matrix[i, j] = welfare_data["welfare_matrix"][strat_A][strat_B]["producer_profit"]
    
    # Find Nash equilibria positions
    nash_positions = []
    nash_equilibria = summary_data.get("nash_equilibria", [])
    
    for ne in nash_equilibria:
        # Get indices for both strategies
        if ne["strategy_A"] == "Flat":
            idx_A = 0
        else:
            idx_A = int(ne["strategy_A"].split("-")[0]) - 1
        
        if ne["strategy_B"] == "Flat":
            idx_B = 0
        else:
            idx_B = int(ne["strategy_B"].split("-")[0]) - 1
        
        nash_positions.append((idx_A, idx_B))
    
    heatmap_data.append({
        's0': s0,
        'matrix': matrix,
        'nash_positions': nash_positions
    })

# Create figure with subplots
n_plots = len(heatmap_data)
fig, axes = plt.subplots(1, n_plots, figsize=(13, 2.0))

# Strategy labels
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# Plot each heatmap with its own color scale
for idx, (ax, data) in enumerate(zip(axes, heatmap_data)):
    vmin = data['matrix'].min()
    vmax = data['matrix'].max()
    
    im = ax.imshow(data['matrix'], cmap='plasma', aspect='equal', vmin=vmin, vmax=vmax)
    
    # Add Nash equilibria stars
    for (i, j) in data['nash_positions']:
        ax.plot(j, i, marker='*', color='red', markersize=5, 
               markeredgecolor='red', markeredgewidth=1, zorder=5)
    
    ax.set_title(f's₀={data["s0"]:.2f}', fontsize=10, pad=5)
    
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    
    ax.set_xticks(np.arange(10) - 0.5, minor=True)
    ax.set_yticks(np.arange(10) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                        fraction=0.08, pad=0.25, aspect=10)
    cbar.ax.tick_params(labelsize=6)

plt.tight_layout()

output_path_png = os.path.join(base_dir, 'producer_surplus_heatmaps_s0_individual.png')
output_path_pdf = os.path.join(base_dir, 'producer_surplus_heatmaps_s0_individual.pdf')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"\nPlots saved to:")
print(f"  PNG: {output_path_png}")
print(f"  PDF: {output_path_pdf}")

print("\nProducer Surplus ranges for each s₀:")
print("-" * 60)
for data in heatmap_data:
    vmin = data['matrix'].min()
    vmax = data['matrix'].max()
    print(f"s₀={data['s0']:.2f}: min={vmin:.2f}, max={vmax:.2f}, range={vmax-vmin:.2f}")

plt.show()

