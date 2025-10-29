import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import json
import argparse
import os
from matplotlib.colors import LinearSegmentedColormap

def parse_payoff_tuple(tuple_str):
    """Parse payoff tuple string and return both values"""
    import re
    numbers = re.findall(r'np\.float64\(([-\d.]+)\)', tuple_str)
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))
    elif len(numbers) == 1:
        return (float(numbers[0]), 0.0)
    return (0.0, 0.0)

def find_nash_equilibria(row_matrix, col_matrix):
    """Find Nash equilibria in the payoff matrix"""
    nash_equilibria = []
    rows, cols = row_matrix.shape
    
    for i in range(rows):
        for j in range(cols):
            # Check if (i,j) is a Nash equilibrium
            row_payoff = row_matrix[i, j]
            col_payoff = col_matrix[i, j]
            
            # Check if row player has no incentive to deviate
            row_deviates = False
            for k in range(rows):
                if k != i and row_matrix[k, j] > row_payoff:
                    row_deviates = True
                    break
            
            # Check if column player has no incentive to deviate
            col_deviates = False
            for l in range(cols):
                if l != j and col_matrix[i, l] > col_payoff:
                    col_deviates = True
                    break
            
            # If neither player wants to deviate, it's a Nash equilibrium
            if not row_deviates and not col_deviates:
                nash_equilibria.append((i, j))
    
    return nash_equilibria

def load_consumer_surplus_data(json_file_path, csv_file_path):
    """Load consumer surplus data from JSON and payoff matrix from CSV"""
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        welfare_data = json.load(f)
    
    # Read the CSV file to get matrix structure
    df = pd.read_csv(csv_file_path, index_col=0)
    
    # Create matrices to store consumer surplus and payoff values
    matrix_size = len(df.columns)
    consumer_surplus_matrix = np.zeros((matrix_size, matrix_size))
    row_player_matrix = np.zeros((matrix_size, matrix_size))
    column_player_matrix = np.zeros((matrix_size, matrix_size))
    
    # Get welfare matrix data
    welfare_matrix = welfare_data['welfare_matrix']
    
    # Extract consumer surplus and payoff data
    for i, row_label in enumerate(df.index):
        for j, col_label in enumerate(df.columns):
            # Get consumer surplus from JSON
            if row_label in welfare_matrix and col_label in welfare_matrix[row_label]:
                consumer_surplus_matrix[i, j] = welfare_matrix[row_label][col_label]['consumer_surplus']
            
            # Get payoff data from CSV for Nash equilibrium calculation
            tuple_str = df.iloc[i, j]
            row_payoff, col_payoff = parse_payoff_tuple(tuple_str)
            row_player_matrix[i, j] = row_payoff
            column_player_matrix[i, j] = col_payoff
    
    return consumer_surplus_matrix, row_player_matrix, column_player_matrix

def create_consumer_surplus_heatmap(json_file_path, csv_file_path, output_path=None, color_range=None, hide_numbers=False):
    """Create a heatmap from consumer surplus data"""
    import numpy as np
    import matplotlib.colors as mcolors
    
    # Load data
    consumer_surplus_matrix, row_player_matrix, column_player_matrix = load_consumer_surplus_data(json_file_path, csv_file_path)
    matrix_size = consumer_surplus_matrix.shape[0]
    
    # Define custom colormap with 200 equidistant bins from white to blue
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Define RGB values for 100-color palette: white -> blue -> green
    color_palette = [
        (1.0, 1.0, 1.0),    # Pure white (lowest)
        (0.99, 0.99, 1.0),  # Very light blue 1
        (0.98, 0.98, 1.0),  # Very light blue 2
        (0.97, 0.97, 1.0),  # Very light blue 3
        (0.96, 0.96, 1.0),  # Very light blue 4
        (0.95, 0.95, 1.0),  # Very light blue 5
        (0.94, 0.94, 1.0),  # Very light blue 6
        (0.93, 0.93, 1.0),  # Very light blue 7
        (0.92, 0.92, 1.0),  # Very light blue 8
        (0.91, 0.91, 1.0),  # Very light blue 9
        (0.9, 0.9, 1.0),    # Very light blue 10
        (0.89, 0.89, 1.0),  # Light blue 1
        (0.88, 0.88, 1.0),  # Light blue 2
        (0.87, 0.87, 1.0),  # Light blue 3
        (0.86, 0.86, 1.0),  # Light blue 4
        (0.85, 0.85, 1.0),  # Light blue 5
        (0.84, 0.84, 1.0),  # Light blue 6
        (0.83, 0.83, 1.0),  # Light blue 7
        (0.82, 0.82, 1.0),  # Light blue 8
        (0.81, 0.81, 1.0),  # Light blue 9
        (0.8, 0.8, 1.0),    # Light blue 10
        (0.79, 0.79, 1.0),  # Light blue 11
        (0.78, 0.78, 1.0),  # Light blue 12
        (0.77, 0.77, 1.0),  # Light blue 13
        (0.76, 0.76, 1.0),  # Light blue 14
        (0.75, 0.75, 1.0),  # Light blue 15
        (0.74, 0.74, 1.0),  # Light blue 16
        (0.73, 0.73, 1.0),  # Light blue 17
        (0.72, 0.72, 1.0),  # Light blue 18
        (0.71, 0.71, 1.0),  # Light blue 19
        (0.7, 0.7, 1.0),    # Light blue 20
        (0.69, 0.69, 1.0),  # Light blue 21
        (0.68, 0.68, 1.0),  # Light blue 22
        (0.67, 0.67, 1.0),  # Light blue 23
        (0.66, 0.66, 1.0),  # Light blue 24
        (0.65, 0.65, 1.0),  # Light blue 25
        (0.64, 0.64, 1.0),  # Light blue 26
        (0.63, 0.63, 1.0),  # Light blue 27
        (0.62, 0.62, 1.0),  # Light blue 28
        (0.61, 0.61, 1.0),  # Light blue 29
        (0.6, 0.6, 1.0),    # Light blue 30
        (0.59, 0.59, 1.0),  # Medium blue 1
        (0.58, 0.58, 1.0),  # Medium blue 2
        (0.57, 0.57, 1.0),  # Medium blue 3
        (0.56, 0.56, 1.0),  # Medium blue 4
        (0.55, 0.55, 1.0),  # Medium blue 5
        (0.54, 0.54, 1.0),  # Medium blue 6
        (0.53, 0.53, 1.0),  # Medium blue 7
        (0.52, 0.52, 1.0),  # Medium blue 8
        (0.51, 0.51, 1.0),  # Medium blue 9
        (0.5, 0.5, 1.0),    # Medium blue 10
        (0.49, 0.49, 1.0),  # Medium blue 11
        (0.48, 0.48, 1.0),  # Medium blue 12
        (0.47, 0.47, 1.0),  # Medium blue 13
        (0.46, 0.46, 1.0),  # Medium blue 14
        (0.45, 0.45, 1.0),  # Medium blue 15
        (0.44, 0.44, 1.0),  # Medium blue 16
        (0.43, 0.43, 1.0),  # Medium blue 17
        (0.42, 0.42, 1.0),  # Medium blue 18
        (0.41, 0.41, 1.0),  # Medium blue 19
        (0.4, 0.4, 1.0),    # Medium blue 20
        (0.39, 0.39, 1.0),  # Medium blue 21
        (0.38, 0.38, 1.0),  # Medium blue 22
        (0.37, 0.37, 1.0),  # Medium blue 23
        (0.36, 0.36, 1.0),  # Medium blue 24
        (0.35, 0.35, 1.0),  # Medium blue 25
        (0.34, 0.34, 1.0),  # Medium blue 26
        (0.33, 0.33, 1.0),  # Medium blue 27
        (0.32, 0.32, 1.0),  # Medium blue 28
        (0.31, 0.31, 1.0),  # Medium blue 29
        (0.3, 0.3, 1.0),    # Medium blue 30
        (0.29, 0.29, 1.0),  # Medium blue 31
        (0.28, 0.28, 1.0),  # Medium blue 32
        (0.27, 0.27, 1.0),  # Medium blue 33
        (0.26, 0.26, 1.0),  # Medium blue 34
        (0.25, 0.25, 1.0),  # Medium blue 35
        (0.24, 0.24, 1.0),  # Medium blue 36
        (0.23, 0.23, 1.0),  # Medium blue 37
        (0.22, 0.22, 1.0),  # Medium blue 38
        (0.21, 0.21, 1.0),  # Medium blue 39
        (0.2, 0.2, 1.0),    # Medium blue 40
        (0.19, 0.19, 1.0),  # Medium blue 41
        (0.18, 0.18, 1.0),  # Medium blue 42
        (0.17, 0.17, 1.0),  # Medium blue 43
        (0.16, 0.16, 1.0),  # Medium blue 44
        (0.15, 0.15, 1.0),  # Medium blue 45
        (0.14, 0.14, 1.0),  # Medium blue 46
        (0.13, 0.13, 1.0),  # Medium blue 47
        (0.12, 0.12, 1.0),  # Medium blue 48
        (0.11, 0.11, 1.0),  # Medium blue 49
        (0.1, 0.1, 1.0),    # Medium blue 50
        (0.09, 0.09, 1.0),  # Medium blue 51
        (0.08, 0.08, 1.0),  # Medium blue 52
        (0.07, 0.07, 1.0),  # Medium blue 53
        (0.06, 0.06, 1.0),  # Medium blue 54
        (0.05, 0.05, 1.0),  # Medium blue 55
        (0.04, 0.04, 1.0),  # Medium blue 56
        (0.03, 0.03, 1.0),  # Medium blue 57
        (0.02, 0.02, 1.0),  # Medium blue 58
        (0.01, 0.01, 1.0),  # Medium blue 59
        (0.0, 0.0, 1.0),    # Pure blue
    ]
    # Create 200 equidistant bins
    n_bins = 1000
    colors = []
    
    for i in range(n_bins):
        # Map to color palette segments
        ratio = i / (n_bins - 1)  # 0 to 1
        
        # Determine which color segment we're in
        segment = ratio * (len(color_palette) - 1)
        segment_index = int(segment)
        segment_fraction = segment - segment_index
        
        # Clamp segment_index to valid range
        segment_index = min(segment_index, len(color_palette) - 2)
        
        # Interpolate between adjacent colors
        color1 = color_palette[segment_index]
        color2 = color_palette[segment_index + 1]
        
        red = color1[0] + segment_fraction * (color2[0] - color1[0])
        green = color1[1] + segment_fraction * (color2[1] - color1[1])
        blue = color1[2] + segment_fraction * (color2[2] - color1[2])
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(red * 255), int(green * 255), int(blue * 255)
        )
        colors.append(hex_color)
    
    # Create colormap with exactly 200 bins
    cmap = mcolors.LinearSegmentedColormap.from_list('200_bin_white_to_blue', colors, N=n_bins)
    
    # Create the heatmap using exact colors from color matrix
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Create color matrix with exact colors
    color_matrix = np.empty((matrix_size, matrix_size), dtype=object)
    
    if color_range is not None:
        vmin, vmax = color_range
    else:
        vmin, vmax = consumer_surplus_matrix.min(), consumer_surplus_matrix.max()
    
    # Map each cell consumer surplus to its corresponding color
    for i in range(matrix_size):
        for j in range(matrix_size):
            cell_value = consumer_surplus_matrix[i, j]
            
            # Normalize value to 0-1 range
            norm_val = (cell_value - vmin) / (vmax - vmin)
            
            # Map to bin index (0 to 199)
            bin_index = int(norm_val * (n_bins - 1))
            bin_index = max(0, min(bin_index, n_bins - 1))  # Clamp to valid range
            
            # Get the color for this bin
            color = colors[bin_index]
            
            # Store in color matrix
            color_matrix[i, j] = color
    
    # Create colored rectangles for each cell using exact colors
    for i in range(matrix_size):
        for j in range(matrix_size):
            color = color_matrix[i, j]
            
            # Create rectangle with the exact color
            rect = matplotlib.patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                               facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
    
    # Set tick labels using sequential numbers (1, 2, 3, ...)
    ax.set_xticks(range(matrix_size))
    ax.set_yticks(range(matrix_size))
    ax.set_xticklabels([str(i+1) for i in range(matrix_size)], fontsize=28)
    ax.set_yticklabels([str(i+1) for i in range(matrix_size)], fontsize=28)
    
    # Set axis limits and aspect
    ax.set_xlim(-0.5, matrix_size - 0.5)
    ax.set_ylim(-0.5, matrix_size - 0.5)
    ax.set_aspect('equal')
    
    # Find and mark Nash equilibria (based on payoff matrix)
    nash_equilibria = find_nash_equilibria(row_player_matrix, column_player_matrix)
    
    # Add stars for Nash equilibria (as background layer)
    for i, j in nash_equilibria:
        ax.text(j, i, '★', ha='center', va='center', fontsize=60, color='red', alpha=1.0)
    
    # Add consumer surplus values on top of stars (only if not hidden)
    if not hide_numbers:
        for i in range(matrix_size):
            for j in range(matrix_size):
                consumer_surplus_val = consumer_surplus_matrix[i, j]
                
                # Calculate text color based on bin position (40% threshold)
                cell_value = consumer_surplus_matrix[i, j]
                
                if color_range is not None:
                    vmin, vmax = color_range
                    norm_val = (cell_value - vmin) / (vmax - vmin)
                else:
                    norm_val = (cell_value - consumer_surplus_matrix.min()) / (consumer_surplus_matrix.max() - consumer_surplus_matrix.min())
                
                # Map to bin index (0 to 199)
                bin_index = int(norm_val * (n_bins - 1))
                bin_index = max(0, min(bin_index, n_bins - 1))
                
                # Text color decision based on bin position:
                # - If data lies in below 40% of bins (0-79): black text
                # - Otherwise (80-199): white text
                text_color = 'white' if bin_index >= 80 else 'black'
                
                ax.text(j, i, f'{consumer_surplus_val:.1f}', 
                       ha='center', va='center', fontsize=28, color=text_color)
    
    # Set output path
    if output_path is None:
        output_path = 'consumer_surplus_heatmap.pdf'
    
    # Save the heatmap
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Consumer surplus heatmap saved as {output_path}")

def create_consumer_surplus_colorbar(json_file_path, csv_file_path, output_path=None, color_range=None):
    """Create a separate colorbar image for consumer surplus"""
    import numpy as np
    import matplotlib.colors as mcolors
    
    # Load data
    consumer_surplus_matrix, _, _ = load_consumer_surplus_data(json_file_path, csv_file_path)
    
    # Define custom colormap with 200 equidistant bins from white to blue
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Define RGB values for 100-color palette: white -> blue -> green with balanced distribution
    color_palette = [
        (1.0, 1.0, 1.0),    # Pure white (lowest)
        (0.99, 0.99, 1.0),  # Very light blue 1
        (0.98, 0.98, 1.0),  # Very light blue 2
        (0.97, 0.97, 1.0),  # Very light blue 3
        (0.96, 0.96, 1.0),  # Very light blue 4
        (0.95, 0.95, 1.0),  # Very light blue 5
        (0.94, 0.94, 1.0),  # Very light blue 6
        (0.93, 0.93, 1.0),  # Very light blue 7
        (0.92, 0.92, 1.0),  # Very light blue 8
        (0.91, 0.91, 1.0),  # Very light blue 9
        (0.9, 0.9, 1.0),    # Very light blue 10
        (0.89, 0.89, 1.0),  # Light blue 1
        (0.88, 0.88, 1.0),  # Light blue 2
        (0.87, 0.87, 1.0),  # Light blue 3
        (0.86, 0.86, 1.0),  # Light blue 4
        (0.85, 0.85, 1.0),  # Light blue 5
        (0.84, 0.84, 1.0),  # Light blue 6
        (0.83, 0.83, 1.0),  # Light blue 7
        (0.82, 0.82, 1.0),  # Light blue 8
        (0.81, 0.81, 1.0),  # Light blue 9
        (0.8, 0.8, 1.0),    # Light blue 10
        (0.79, 0.79, 1.0),  # Light blue 11
        (0.78, 0.78, 1.0),  # Light blue 12
        (0.77, 0.77, 1.0),  # Light blue 13
        (0.76, 0.76, 1.0),  # Light blue 14
        (0.75, 0.75, 1.0),  # Light blue 15
        (0.74, 0.74, 1.0),  # Light blue 16
        (0.73, 0.73, 1.0),  # Light blue 17
        (0.72, 0.72, 1.0),  # Light blue 18
        (0.71, 0.71, 1.0),  # Light blue 19
        (0.7, 0.7, 1.0),    # Light blue 20
        (0.69, 0.69, 1.0),  # Light blue 21
        (0.68, 0.68, 1.0),  # Light blue 22
        (0.67, 0.67, 1.0),  # Light blue 23
        (0.66, 0.66, 1.0),  # Light blue 24
        (0.65, 0.65, 1.0),  # Light blue 25
        (0.64, 0.64, 1.0),  # Light blue 26
        (0.63, 0.63, 1.0),  # Light blue 27
        (0.62, 0.62, 1.0),  # Light blue 28
        (0.61, 0.61, 1.0),  # Light blue 29
        (0.6, 0.6, 1.0),    # Light blue 30
        (0.59, 0.59, 1.0),  # Medium blue 1
        (0.58, 0.58, 1.0),  # Medium blue 2
        (0.57, 0.57, 1.0),  # Medium blue 3
        (0.56, 0.56, 1.0),  # Medium blue 4
        (0.55, 0.55, 1.0),  # Medium blue 5
        (0.54, 0.54, 1.0),  # Medium blue 6
        (0.53, 0.53, 1.0),  # Medium blue 7
        (0.52, 0.52, 1.0),  # Medium blue 8
        (0.51, 0.51, 1.0),  # Medium blue 9
        (0.5, 0.5, 1.0),    # Medium blue 10
        (0.49, 0.49, 1.0),  # Medium blue 11
        (0.48, 0.48, 1.0),  # Medium blue 12
        (0.47, 0.47, 1.0),  # Medium blue 13
        (0.46, 0.46, 1.0),  # Medium blue 14
        (0.45, 0.45, 1.0),  # Medium blue 15
        (0.44, 0.44, 1.0),  # Medium blue 16
        (0.43, 0.43, 1.0),  # Medium blue 17
        (0.42, 0.42, 1.0),  # Medium blue 18
        (0.41, 0.41, 1.0),  # Medium blue 19
        (0.4, 0.4, 1.0),    # Medium blue 20
        (0.39, 0.39, 1.0),  # Medium blue 21
        (0.38, 0.38, 1.0),  # Medium blue 22
        (0.37, 0.37, 1.0),  # Medium blue 23
        (0.36, 0.36, 1.0),  # Medium blue 24
        (0.35, 0.35, 1.0),  # Medium blue 25
        (0.34, 0.34, 1.0),  # Medium blue 26
        (0.33, 0.33, 1.0),  # Medium blue 27
        (0.32, 0.32, 1.0),  # Medium blue 28
        (0.31, 0.31, 1.0),  # Medium blue 29
        (0.3, 0.3, 1.0),    # Medium blue 30
        (0.29, 0.29, 1.0),  # Medium blue 31
        (0.28, 0.28, 1.0),  # Medium blue 32
        (0.27, 0.27, 1.0),  # Medium blue 33
        (0.26, 0.26, 1.0),  # Medium blue 34
        (0.25, 0.25, 1.0),  # Medium blue 35
        (0.24, 0.24, 1.0),  # Medium blue 36
        (0.23, 0.23, 1.0),  # Medium blue 37
        (0.22, 0.22, 1.0),  # Medium blue 38
        (0.21, 0.21, 1.0),  # Medium blue 39
        (0.2, 0.2, 1.0),    # Medium blue 40
        (0.19, 0.19, 1.0),  # Medium blue 41
        (0.18, 0.18, 1.0),  # Medium blue 42
        (0.17, 0.17, 1.0),  # Medium blue 43
        (0.16, 0.16, 1.0),  # Medium blue 44
        (0.15, 0.15, 1.0),  # Medium blue 45
        (0.14, 0.14, 1.0),  # Medium blue 46
        (0.13, 0.13, 1.0),  # Medium blue 47
        (0.12, 0.12, 1.0),  # Medium blue 48
        (0.11, 0.11, 1.0),  # Medium blue 49
        (0.1, 0.1, 1.0),    # Medium blue 50
        (0.09, 0.09, 1.0),  # Medium blue 51
        (0.08, 0.08, 1.0),  # Medium blue 52
        (0.07, 0.07, 1.0),  # Medium blue 53
        (0.06, 0.06, 1.0),  # Medium blue 54
        (0.05, 0.05, 1.0),  # Medium blue 55
        (0.04, 0.04, 1.0),  # Medium blue 56
        (0.03, 0.03, 1.0),  # Medium blue 57
        (0.02, 0.02, 1.0),  # Medium blue 58
        (0.01, 0.01, 1.0),  # Medium blue 59
        (0.0, 0.0, 1.0),    # Pure blue
    ]
    
    # Create 1000 equidistant bins for fine granularity
    n_bins = 1000
    colors = []
    
    for i in range(n_bins):
        # Map to color palette segments
        ratio = i / (n_bins - 1)  # 0 to 1
        
        # Determine which color segment we're in
        segment = ratio * (len(color_palette) - 1)
        segment_index = int(segment)
        segment_fraction = segment - segment_index
        
        # Clamp segment_index to valid range
        segment_index = min(segment_index, len(color_palette) - 2)
        
        # Interpolate between adjacent colors
        color1 = color_palette[segment_index]
        color2 = color_palette[segment_index + 1]
        
        red = color1[0] + segment_fraction * (color2[0] - color1[0])
        green = color1[1] + segment_fraction * (color2[1] - color1[1])
        blue = color1[2] + segment_fraction * (color2[2] - color1[2])
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(red * 255), int(green * 255), int(blue * 255)
        )
        colors.append(hex_color)
    
    # Create colormap with exactly 200 bins
    cmap = mcolors.LinearSegmentedColormap.from_list('200_bin_white_to_blue', colors, N=n_bins)
    
    # Create a separate figure for the horizontal colorbar
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Set color range if specified - use exact range as specified
    if color_range is not None:
        vmin, vmax = color_range
        im = ax.imshow(consumer_surplus_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(consumer_surplus_matrix, cmap=cmap, aspect='auto')
    
    # Add horizontal colorbar without title
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=14)
    
    # Set colorbar to show numerical ticks at different levels
    if color_range is not None:
        vmin, vmax = color_range
        # Create 5 evenly spaced ticks across the range
        num_ticks = 5
        tick_values = [vmin + i * (vmax - vmin) / (num_ticks - 1) for i in range(num_ticks)]
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.1f}' for val in tick_values])
    
    # Remove the main plot
    ax.remove()
    
    # Set output path
    if output_path is None:
        output_path = 'consumer_surplus_colorbar.pdf'
    
    # Save the colorbar
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Consumer surplus colorbar saved as {output_path}")

def create_consumer_surplus_matrix(json_file_path, csv_file_path, color_range=None):
    """Create a matrix where each cell contains the RGB color values for consumer surplus"""
    import numpy as np
    import matplotlib.colors as mcolors
    
    # Load data
    consumer_surplus_matrix, _, _ = load_consumer_surplus_data(json_file_path, csv_file_path)
    matrix_size = consumer_surplus_matrix.shape[0]
    
    # Define RGB values for white (lower) and blue (higher)
    white_rgb = (1.0, 1.0, 1.0)  # Pure white
    blue_rgb = (0.0, 0.0, 0.5)   # Dark blue
    
    # Create 200 equidistant bins
    n_bins = 300
    colors = []
    
    for i in range(n_bins):
        # Interpolate between white and blue RGB values
        ratio = i / (n_bins - 1)  # 0 to 1
        
        red = white_rgb[0] + ratio * (blue_rgb[0] - white_rgb[0])
        green = white_rgb[1] + ratio * (blue_rgb[1] - white_rgb[1])
        blue = white_rgb[2] + ratio * (blue_rgb[2] - white_rgb[2])
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(red * 255), int(green * 255), int(blue * 255)
        )
        colors.append(hex_color)
    
    # Create color matrix
    color_matrix = np.empty((matrix_size, matrix_size), dtype=object)
    
    if color_range is not None:
        vmin, vmax = color_range
    else:
        vmin, vmax = consumer_surplus_matrix.min(), consumer_surplus_matrix.max()
    
    # Map each cell consumer surplus to its corresponding color
    for i in range(matrix_size):
        for j in range(matrix_size):
            cell_value = consumer_surplus_matrix[i, j]
            
            # Normalize value to 0-1 range
            norm_val = (cell_value - vmin) / (vmax - vmin)
            
            # Map to bin index (0 to 199)
            bin_index = int(norm_val * (n_bins - 1))
            bin_index = max(0, min(bin_index, n_bins - 1))  # Clamp to valid range
            
            # Get the color for this bin
            color = colors[bin_index]
            
            # Store in color matrix
            color_matrix[i, j] = color
    
    return color_matrix, consumer_surplus_matrix

def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Generate consumer surplus heatmap and colorbar')
    parser.add_argument('json_file', help='Path to the JSON file containing welfare matrix')
    parser.add_argument('csv_file', help='Path to the CSV file containing payoff matrix')
    parser.add_argument('--heatmap-output', '-o', help='Output path for heatmap (default: consumer_surplus_heatmap.pdf)')
    parser.add_argument('--colorbar-output', '-c', help='Output path for colorbar (default: consumer_surplus_colorbar.pdf)')
    parser.add_argument('--color-min', type=float, help='Minimum value for color range')
    parser.add_argument('--color-max', type=float, help='Maximum value for color range')
    parser.add_argument('--heatmap-only', action='store_true', help='Generate only heatmap')
    parser.add_argument('--colorbar-only', action='store_true', help='Generate only colorbar')
    parser.add_argument('--color-matrix', action='store_true', help='Generate color matrix instead of plots')
    parser.add_argument('--hide-numbers', action='store_true', help='Hide numbers inside cells, show only colors')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file '{args.json_file}' not found!")
        return
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found!")
        return
    
    # Prepare color range
    color_range = None
    if args.color_min is not None and args.color_max is not None:
        color_range = (args.color_min, args.color_max)
    elif args.color_min is not None or args.color_max is not None:
        print("Warning: Both --color-min and --color-max must be specified for custom color range")
    
    # Generate color matrix if requested
    if args.color_matrix:
        color_matrix, consumer_surplus_matrix = create_consumer_surplus_matrix(args.json_file, args.csv_file, color_range)
        
        print("Consumer Surplus Color Matrix (RGB hex values for each cell):")
        print("=" * 60)
        for i in range(color_matrix.shape[0]):
            row_str = ""
            for j in range(color_matrix.shape[1]):
                row_str += f"{color_matrix[i, j]:>8} "
            print(f"Row {i+1:2d}: {row_str}")
        
        print("\nConsumer Surplus Matrix (values used for color mapping):")
        print("=" * 60)
        for i in range(consumer_surplus_matrix.shape[0]):
            row_str = ""
            for j in range(consumer_surplus_matrix.shape[1]):
                row_str += f"{consumer_surplus_matrix[i, j]:8.2f} "
            print(f"Row {i+1:2d}: {row_str}")
        
        return
    
    # Generate heatmap
    if not args.colorbar_only:
        create_consumer_surplus_heatmap(args.json_file, args.csv_file, args.heatmap_output, color_range, args.hide_numbers)
    
    # Generate colorbar
    if not args.heatmap_only:
        create_consumer_surplus_colorbar(args.json_file, args.csv_file, args.colorbar_output, color_range)

if __name__ == "__main__":
    main()
