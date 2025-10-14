import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Path to the specific run
run_path = "/home/ab823254/data/tiered_pricing_network_economics/run_00309_g0.0010_b0.30_a1.77_s0.32/10x10"

# Read welfare matrix for data
welfare_path = os.path.join(run_path, "welfare_matrix.json")
summary_path = os.path.join(run_path, "summary.json")

with open(welfare_path, 'r') as f:
    welfare_data = json.load(f)

with open(summary_path, 'r') as f:
    summary_data = json.load(f)

# Extract consumer surplus matrix and payoffs
strategies = ["Flat", "2-Tier", "3-Tier", "4-Tier", "5-Tier", 
              "6-Tier", "7-Tier", "8-Tier", "9-Tier", "10-Tier"]

consumer_surplus_matrix = np.zeros((10, 10))
payoff_matrix = []  # Store individual payoffs for display

for i, strat_A in enumerate(strategies):
    row = []
    for j, strat_B in enumerate(strategies):
        if strat_A in welfare_data["welfare_matrix"] and strat_B in welfare_data["welfare_matrix"][strat_A]:
            consumer_surplus = welfare_data["welfare_matrix"][strat_A][strat_B]["consumer_surplus"]
            profit_A = welfare_data["welfare_matrix"][strat_A][strat_B]["profit_A"]
            profit_B = welfare_data["welfare_matrix"][strat_A][strat_B]["profit_B"]
            consumer_surplus_matrix[i, j] = consumer_surplus
            row.append((profit_A, profit_B))
    payoff_matrix.append(row)

# Find Nash equilibria
nash_positions = []
nash_equilibria = summary_data.get("nash_equilibria", [])

for ne in nash_equilibria:
    if ne["strategy_A"] == "Flat":
        idx_A = 0
    else:
        idx_A = int(ne["strategy_A"].split("-")[0]) - 1
    
    if ne["strategy_B"] == "Flat":
        idx_B = 0
    else:
        idx_B = int(ne["strategy_B"].split("-")[0]) - 1
    
    nash_positions.append((idx_A, idx_B))

# Create figure
fig, ax = plt.subplots(figsize=(10, 9))

# Plot heatmap with color based on consumer surplus
vmin = consumer_surplus_matrix.min()
vmax = consumer_surplus_matrix.max()
im = ax.imshow(consumer_surplus_matrix, cmap='plasma', aspect='equal', vmin=vmin, vmax=vmax)

# Strategy labels
labels = ['Flat', '2-Tier', '3-Tier', '4-Tier', '5-Tier', 
          '6-Tier', '7-Tier', '8-Tier', '9-Tier', '10-Tier']

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(labels, fontsize=14, rotation=45, ha='right')
ax.set_yticklabels(labels, fontsize=14)

# Add grid
ax.set_xticks(np.arange(10) - 0.5, minor=True)
ax.set_yticks(np.arange(10) - 0.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
ax.tick_params(which='minor', size=0)

# Add text annotations with individual payoffs
for i in range(10):
    for j in range(10):
        profit_A, profit_B = payoff_matrix[i][j]
        
        # Choose text color based on background brightness
        if consumer_surplus_matrix[i, j] < (vmin + vmax) / 2:
            text_color = 'white'
        else:
            text_color = 'black'
        
        # Display profit_A and profit_B in two rows
        ax.text(j, i-0.15, f'{profit_A:.2f}', ha='center', va='center', 
               color=text_color, fontsize=12, fontweight='bold')
        ax.text(j, i+0.15, f'{profit_B:.2f}', ha='center', va='center', 
               color=text_color, fontsize=12, fontweight='bold')

# Mark Nash equilibria with stars in top right corner
for (i, j) in nash_positions:
    ax.plot(j+0.35, i-0.35, marker='*', color='red', markersize=15, 
           markeredgecolor='red', markeredgewidth=2, zorder=5)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Consumer Surplus', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()

# Save output
output_path_png = os.path.join(run_path, 'consumer_surplus_matrix.png')
output_path_pdf = os.path.join(run_path, 'consumer_surplus_matrix.pdf')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.savefig(output_path_pdf, bbox_inches='tight')

print(f"Consumer Surplus Matrix saved to:")
print(f"  PNG: {output_path_png}")
print(f"  PDF: {output_path_pdf}")
print(f"\nConsumer Surplus range: min={vmin:.2f}, max={vmax:.2f}")
print(f"Nash Equilibria positions: {nash_positions}")

plt.show()

