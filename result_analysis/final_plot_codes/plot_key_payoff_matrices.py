import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import re
from pathlib import Path

# Define the base directory
base_dir = Path("/home/ab823254/data/tiered_pricing_network_economics/optimal")

# Function to parse payoff values from CSV
def parse_payoff_cell(cell):
    """Extract payoff values from the CSV cell string"""
    match = re.search(r'\((?:np\.float64\()?([+-]?\d+\.?\d*)\)?,\s*(?:np\.float64\()?([+-]?\d+\.?\d*)\)', cell)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# Function to identify Nash Equilibria
def find_nash_equilibria(payoff_matrix, strategies):
    """Find Nash Equilibria in the payoff matrix"""
    nash_eq = []
    
    for i, strat in enumerate(strategies):
        payoff_A, payoff_B = payoff_matrix[strat][strat]
        
        # Check if either player can improve by deviating
        can_A_improve = False
        can_B_improve = False
        
        for other_strat in strategies:
            if other_strat != strat:
                if payoff_matrix[other_strat][strat][0] > payoff_A + 0.01:
                    can_A_improve = True
                if payoff_matrix[strat][other_strat][1] > payoff_B + 0.01:
                    can_B_improve = True
        
        if not can_A_improve and not can_B_improve:
            nash_eq.append(i)
    
    return nash_eq

# Function to plot a single payoff matrix
def plot_matrix(ax, size, base_dir):
    folder_name = f"{size}x{size}"
    csv_file = base_dir / folder_name / "payoff_matrix.csv"
    
    df = pd.read_csv(csv_file, index_col=0)
    strategies = list(df.columns)
    
    # Parse the payoff matrix
    payoff_matrix = {}
    for row_strat in strategies:
        payoff_matrix[row_strat] = {}
        for col_strat in strategies:
            cell = df.loc[row_strat, col_strat]
            payoff_A, payoff_B = parse_payoff_cell(cell)
            payoff_matrix[row_strat][col_strat] = (payoff_A, payoff_B)
    
    nash_eq = find_nash_equilibria(payoff_matrix, strategies)
    
    # Create a color-coded heatmap based on total payoffs
    n = len(strategies)
    total_payoffs = np.zeros((n, n))
    
    for i, row_strat in enumerate(strategies):
        for j, col_strat in enumerate(strategies):
            payoff_A, payoff_B = payoff_matrix[row_strat][col_strat]
            total_payoffs[i, j] = payoff_A + payoff_B
    
    # Plot heatmap
    im = ax.imshow(total_payoffs, cmap='RdYlGn', aspect='auto', alpha=0.3)
    
    # Add text annotations with payoffs
    fontsize = 11 if size <= 3 else 8
    
    for i, row_strat in enumerate(strategies):
        for j, col_strat in enumerate(strategies):
            payoff_A, payoff_B = payoff_matrix[row_strat][col_strat]
            
            # Check if this is a Nash Equilibrium (diagonal only)
            is_ne = (i == j) and (i in nash_eq)
            
            # Format text
            text = f'{payoff_A:.1f}\n{payoff_B:.1f}'
            
            # Color and weight based on NE status
            if is_ne:
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=fontsize, fontweight='bold', color='red')
                # Add marker for Nash Equilibrium
                circle = plt.Circle((j, i), 0.35, color='gold', fill=False, linewidth=2.5)
                ax.add_patch(circle)
            else:
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=fontsize, color='black')
    
    # Set ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    
    # Simplified labels for readability
    if size == 1:
        labels = ['Flat']
    elif size == 2:
        labels = ['Flat', '2-Tier']
    else:
        labels = ['Flat'] + [f'{i}-Tier' for i in range(2, size+1)]
    
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add title
    ax.set_title(f'{size}×{size} Game', fontsize=13, fontweight='bold', pad=10)
    
    # Add axis labels
    ax.set_ylabel('ISP A Strategy', fontsize=11)
    ax.set_xlabel('ISP B Strategy', fontsize=11)
    
    # Style
    ax.tick_params(length=0)
    
    # Add grid
    for i in range(n+1):
        ax.axhline(i-0.5, color='gray', linewidth=0.8)
        ax.axvline(i-0.5, color='gray', linewidth=0.8)
    
    return nash_eq

# Create figure with only key matrices: 1x1, 2x2, 3x3, 10x10
fig = plt.figure(figsize=(7.5, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

# Plot key matrices
sizes = [1, 2, 3, 10]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for size, (row, col) in zip(sizes, positions):
    ax = fig.add_subplot(gs[row, col])
    nash_eq = plot_matrix(ax, size, base_dir)
    print(f"{size}×{size} Nash Equilibria: {nash_eq}")

# Add legend
legend_elements = [
    mpatches.Patch(color='gold', label='Nash Equilibrium'),
    mpatches.Patch(facecolor='lightgreen', alpha=0.3, label='Higher Total Payoff'),
    mpatches.Patch(facecolor='lightcoral', alpha=0.3, label='Lower Total Payoff')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
          fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_path = base_dir / "key_payoff_matrices_ieee.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

output_path_pdf = base_dir / "key_payoff_matrices_ieee.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure also saved as: {output_path_pdf}")

plt.show()




