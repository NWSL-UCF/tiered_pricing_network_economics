import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re

# Define the base directory
base_dir = Path("/home/ab823254/data/tiered_pricing_network_economics/optimal")

# Dictionary to map tier number to strategy name
tier_names = {
    1: "Flat",
    2: "2-Tier",
    3: "3-Tier",
    4: "4-Tier",
    5: "5-Tier",
    6: "6-Tier",
    7: "7-Tier",
    8: "8-Tier",
    9: "9-Tier",
    10: "10-Tier"
}

# Load payoff matrix to identify Nash Equilibria
payoff_csv = base_dir / "10x10" / "payoff_matrix.csv"
df = pd.read_csv(payoff_csv, index_col=0)

# Parse the payoff matrix
payoff_matrix = {}
strategies = list(df.columns)

for row_strat in strategies:
    payoff_matrix[row_strat] = {}
    for col_strat in strategies:
        cell = df.loc[row_strat, col_strat]
        # Parse the tuple string - extract numbers inside the parentheses, excluding "float64"
        # Pattern: find two numbers separated by comma inside parentheses
        match = re.search(r'\((?:np\.float64\()?([+-]?\d+\.?\d*)\)?,\s*(?:np\.float64\()?([+-]?\d+\.?\d*)\)', cell)
        if match:
            payoff_A = float(match.group(1))
            payoff_B = float(match.group(2))
            payoff_matrix[row_strat][col_strat] = (payoff_A, payoff_B)

# Identify Nash Equilibria on the diagonal
nash_equilibria = []

for i, strat in enumerate(strategies):
    tier_num = i + 1
    payoff_at_diagonal = payoff_matrix[strat][strat]
    payoff_A, payoff_B = payoff_at_diagonal
    
    # For Nash Equilibrium at (strat, strat):
    # Check if Player A wants to deviate: compare with all other strategies A could play when B plays strat
    # This is the column (fixed B strategy)
    can_A_improve = False
    for other_strat in strategies:
        if other_strat != strat:
            payoff_if_A_deviates = payoff_matrix[other_strat][strat][0]
            if payoff_if_A_deviates > payoff_A + 0.01:  # A can improve by deviating
                can_A_improve = True
                break
    
    # Check if Player B wants to deviate: compare with all other strategies B could play when A plays strat
    # This is the row (fixed A strategy)
    can_B_improve = False
    for other_strat in strategies:
        if other_strat != strat:
            payoff_if_B_deviates = payoff_matrix[strat][other_strat][1]
            if payoff_if_B_deviates > payoff_B + 0.01:  # B can improve by deviating
                can_B_improve = True
                break
    
    is_ne = not can_A_improve and not can_B_improve
    
    if is_ne:
        nash_equilibria.append(tier_num)
        print(f"{tier_num}x{tier_num} ({strat} vs {strat}): Nash Equilibrium")
        print(f"  Payoffs: A={payoff_A:.2f}, B={payoff_B:.2f}")
    else:
        print(f"{tier_num}x{tier_num} ({strat} vs {strat}): NOT a Nash Equilibrium")
        if can_A_improve:
            print(f"  Player A can improve by deviating")
        if can_B_improve:
            print(f"  Player B can improve by deviating")

print(f"\nNash Equilibria at tiers: {nash_equilibria}")

# Extract consumer surplus and producer surplus for each tier configuration
tiers = []
consumer_surpluses = []
producer_surpluses = []

for tier_num in range(1, 11):
    folder_name = f"{tier_num}x{tier_num}"
    welfare_file = base_dir / folder_name / "welfare_matrix.json"
    
    if welfare_file.exists():
        with open(welfare_file, 'r') as f:
            data = json.load(f)
        
        # Get the diagonal entry (both players use the same strategy)
        strategy_name = tier_names[tier_num]
        consumer_surplus = data["welfare_matrix"][strategy_name][strategy_name]["consumer_surplus"]
        producer_surplus = data["welfare_matrix"][strategy_name][strategy_name]["producer_profit"]
        
        tiers.append(tier_num)
        consumer_surpluses.append(consumer_surplus)
        producer_surpluses.append(producer_surplus)

# Create the plot with larger, legible fonts and white background
fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')
ax.set_facecolor('white')

# Plot consumer and producer surplus with improved styling
ax.plot(tiers, consumer_surpluses, marker='o', linewidth=3, markersize=12, 
        color='#1E88E5', label='Consumer Surplus', alpha=0.9, zorder=2,
        markeredgecolor='white', markeredgewidth=2)
ax.plot(tiers, producer_surpluses, marker='s', linewidth=3, markersize=12, 
        color='#E53935', label='Transit ISPs Total Profit', alpha=0.9, zorder=2,
        markeredgecolor='white', markeredgewidth=2)

# Mark Nash Equilibria with larger, more prominent markers
for ne_tier in nash_equilibria:
    idx = ne_tier - 1
    # Add a star marker for NE with glow effect
    ax.scatter(ne_tier, consumer_surpluses[idx], marker='*', s=600, 
              color='#FFD700', edgecolors='#FF8C00', linewidths=2.5, zorder=4, 
              label='Nash Equilibrium' if ne_tier == nash_equilibria[0] else '')
    ax.scatter(ne_tier, producer_surpluses[idx], marker='*', s=600, 
              color='#FFD700', edgecolors='#FF8C00', linewidths=2.5, zorder=4)

# Set labels with larger fonts (no title)
ax.set_xlabel('Number of Tiers', fontsize=22, color='#2C3E50')
ax.set_ylabel('Surplus Value', fontsize=22, color='#2C3E50')

# Improve grid with subtle styling
ax.grid(True, alpha=0.25, linestyle='-', linewidth=1, color='#BDC3C7')
ax.set_axisbelow(True)
ax.set_xticks(tiers)
ax.set_xticklabels([f'{t}x{t}' for t in tiers])
ax.tick_params(axis='both', which='major', labelsize=18, colors='#34495E', width=1.5, length=6)

# Add legend with improved styling
legend = ax.legend(fontsize=18, loc='center right', framealpha=0.95, 
                   edgecolor='#34495E', fancybox=True, shadow=True, 
                   borderpad=1, labelspacing=0.8)
legend.get_frame().set_linewidth(1.5)

# Add horizontal line at y=0 for reference with better styling
ax.axhline(y=0, color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

# Add subtle spines styling
for spine in ax.spines.values():
    spine.set_edgecolor('#34495E')
    spine.set_linewidth(1.5)

plt.tight_layout()

# Save the plot
output_path = base_dir / "10x10" / "surplus_with_ne_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Also save as PDF
output_path_pdf = base_dir / "10x10" / "surplus_with_ne_plot.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Plot also saved as: {output_path_pdf}")

plt.show()

# Save the data with NE indicator to CSV
import csv
csv_output = base_dir / "10x10" / "surplus_with_ne_data.csv"
with open(csv_output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Number_of_Tiers', 'Consumer_Surplus', 'Producer_Surplus', 'Is_Nash_Equilibrium'])
    for tier, cs, ps in zip(tiers, consumer_surpluses, producer_surpluses):
        is_ne = 'Yes' if tier in nash_equilibria else 'No'
        writer.writerow([tier, cs, ps, is_ne])
print(f"Data saved to: {csv_output}")

