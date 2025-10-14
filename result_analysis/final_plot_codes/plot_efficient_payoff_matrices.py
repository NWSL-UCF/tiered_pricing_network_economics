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
    
    # No background coloring - clean white cells
    n = len(strategies)
    
    # Set square aspect ratio for cells
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotations with payoffs - consistent font size across all matrices
    fontsize = 8
    
    for i, row_strat in enumerate(strategies):
        for j, col_strat in enumerate(strategies):
            payoff_A, payoff_B = payoff_matrix[row_strat][col_strat]
            
            # Check if this is a Nash Equilibrium (diagonal only)
            is_ne = (i == j) and (i in nash_eq)
            
            # Format text
            text = f'{payoff_A:.1f}\n{payoff_B:.1f}'
            
            # Place text - bold red for NE cells, regular black for others
            if is_ne:
                # Bold red text for Nash Equilibrium cells
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=fontsize, fontweight='bold', color='red', zorder=2)
            else:
                # Regular text for non-NE cells
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=fontsize, color='black')
    
    # Set ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    
    # Simplified labels for readability
    if size == 1:
        labels = ['F']
    elif size == 2:
        labels = ['F', '2T']
    elif size <= 5:
        labels = ['F'] + [f'{i}T' for i in range(2, size+1)]
    else:
        # For larger matrices, use even more compact labels
        labels = ['F'] + [str(i) for i in range(2, size+1)]
    
    label_fontsize = 8
    ax.set_xticklabels(labels, fontsize=label_fontsize, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    
    # Add title
    title_fontsize = 9
    ax.set_title(f'{size}×{size} Game', fontsize=title_fontsize, fontweight='bold', pad=6)
    
    # Style - clean table appearance
    ax.tick_params(length=0)
    
    # Add simple grid lines for table style
    for i in range(n+1):
        ax.axhline(i-0.5, color='black', linewidth=1)
        ax.axvline(i-0.5, color='black', linewidth=1)
    
    # Set white background
    ax.set_facecolor('white')
    
    # Set limits to ensure square cells
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(n-0.5, -0.5)
    
    return nash_eq

# Create compact figure with uniform cell sizes
# Top row: 3 matrices evenly spaced, Bottom row: 2 matrices evenly spaced
fig = plt.figure(figsize=(9, 5.5), facecolor='white')

# Create a precise grid system - more compact vertical spacing
# Each matrix cell = 2 grid units for uniform sizing
gs = GridSpec(24, 36, figure=fig, hspace=0.15, wspace=0.15)

# Top row: 1x1, 2x2, 3x3 - evenly distributed with equal spacing
# Total width = 36 columns
# Matrices: 2 + 4 + 6 = 12 columns
# Spaces: (36-12)/4 = 6 columns per gap (left, between1-2, between2-3, right)
ax1 = fig.add_subplot(gs[1:3, 6:8])          # 1x1: start at col 6
ax2 = fig.add_subplot(gs[1:5, 14:18])        # 2x2: start at col 14 (6+2+6)
ax3 = fig.add_subplot(gs[1:7, 24:30])        # 3x3: start at col 24 (6+2+6+4+6)

# Bottom row: 4x4 and 5x5 - evenly distributed with equal spacing
# Matrices: 8 + 10 = 18 columns
# Spaces: (36-18)/3 = 6 columns per gap (left, between, right)
ax4 = fig.add_subplot(gs[9:17, 6:14])       # 4x4: start at col 6
ax5 = fig.add_subplot(gs[9:19, 20:30])      # 5x5: start at col 20 (6+8+6)

# Plot matrices
axes = [ax1, ax2, ax3, ax4, ax5]
sizes = [1, 2, 3, 4, 5]

print("Nash Equilibria:")
for ax, size in zip(axes, sizes):
    nash_eq = plot_matrix(ax, size, base_dir)
    ne_indices = [f"{i}" for i in nash_eq]
    print(f"  {size}×{size}: positions {ne_indices}")

# Add legend at the top
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='red', label='Nash Equilibrium (in red)')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=1, 
          fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_path = base_dir / "efficient_payoff_matrices_ieee.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

output_path_pdf = base_dir / "efficient_payoff_matrices_ieee.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Figure also saved as: {output_path_pdf}")

plt.show()

