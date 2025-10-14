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

# Function to plot and save a single payoff matrix
def plot_and_save_matrix(size, base_dir, output_dir):
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
    
    # Create figure - uniform cell size across all matrices
    # Each cell should be approximately 0.8 inches
    cell_size = 0.8
    fig_size = 1.5 + size * cell_size  # Add margin for labels
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), facecolor='white')
    
    # No background coloring - clean white cells
    n = len(strategies)
    
    # Set square aspect ratio for cells
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotations with payoffs - larger font size to fill cells
    fontsize = 16  # Fixed font size for all matrices - larger to occupy cell space
    
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
    else:
        labels = ['F'] + [f'{i}T' for i in range(2, size+1)]
    
    # Uniform label font size
    label_fontsize = 13
    ax.set_xticklabels(labels, fontsize=label_fontsize, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    
    # Add title - uniform size
    title_fontsize = 15
    ax.set_title(f'{size}×{size} Game', fontsize=title_fontsize, fontweight='bold', pad=10)
    
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
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f"payoff_matrix_{size}x{size}.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()
    
    return nash_eq

# Create output directory
output_dir = base_dir
sizes = [1, 2, 3, 4, 5, 6]

print("Generating individual payoff matrix PDFs...")
print("\nNash Equilibria:")
for size in sizes:
    nash_eq = plot_and_save_matrix(size, base_dir, output_dir)
    ne_indices = [f"{i}" for i in nash_eq]
    print(f"  {size}×{size}: positions {ne_indices}")

print(f"\nAll PDF files saved to: {output_dir}")

