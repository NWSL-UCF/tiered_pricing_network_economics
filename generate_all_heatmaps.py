#!/usr/bin/env python3
"""
Script to generate heatmaps for all folders specified in selected_folders.json.

Use --format pdf (default) or --format png for output files.
Use --grid-linewidth to set white cell dividers and the black outer frame (same width).
With --hide-axis-labels (used in this batch script), heatmaps use no figure padding.
"""

import argparse
import json
import os
import subprocess
import sys

def load_selected_folders(json_path):
    """Load the selected folders from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_heatmaps_for_category(base_path, folders, category_name, output_format, grid_linewidth):
    """Generate heatmaps for a specific category (different_alpha or different_s0).

    output_format: 'pdf' or 'png' (file extension used for all outputs).
    grid_linewidth: white cell divider and black outer border width (matplotlib points).
    """
    ext = "." + output_format.lstrip(".")
    print(f"\n{'='*60}")
    print(f"Processing {category_name.upper()} folders")
    print(f"{'='*60}")
    
    # Define color ranges based on category
    if category_name == "different_alpha":
        payoff_color_range = ("-12", "30")      # -12 to 30 for payoff_sum
        consumer_color_range = ("-55", "1")     # -55 to 1 for consumer surplus
    elif category_name == "different_s0":
        payoff_color_range = ("-11", "42")      # -11 to 42 for payoff_sum
        consumer_color_range = ("-45", "-5")   # -45 to -5 for consumer surplus
    else:
        payoff_color_range = ("-55", "45")     # Default fallback
        consumer_color_range = ("-55", "45")   # Default fallback
    
    print(f"🎨 Color ranges:")
    print(f"   📊 Payoff sum: {payoff_color_range[0]} to {payoff_color_range[1]}")
    print(f"   📈 Consumer surplus: {consumer_color_range[0]} to {consumer_color_range[1]}")
    print(f"   📄 Output format: {ext}")
    print(f"   ┃ Grid / border linewidth: {grid_linewidth}")
    
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] Processing: {folder}")
        
        # Construct paths
        folder_path = os.path.join(base_path, folder, "10x10")
        payoff_csv = os.path.join(folder_path, "payoff_matrix.csv")
        welfare_json = os.path.join(folder_path, "welfare_matrix.json")
        
        # Check if files exist
        if not os.path.exists(payoff_csv):
            print(f"  ❌ Missing: {payoff_csv}")
            continue
        if not os.path.exists(welfare_json):
            print(f"  ❌ Missing: {welfare_json}")
            continue
        
        # Create output directory
        output_dir = f"generated_heatmaps/{category_name}/{folder}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate payoff heatmap
        payoff_output = os.path.join(output_dir, f"payoff_heatmap{ext}")
        payoff_colorbar_output = os.path.join(output_dir, f"payoff_colorbar{ext}")
        
        print(f"  📊 Generating payoff heatmap...")
        try:
            cmd = [
                sys.executable, "generate_payoff_heatmap.py",
                payoff_csv,
                "--heatmap-output", payoff_output,
                "--colorbar-output", payoff_colorbar_output,
                "--color-min", payoff_color_range[0],
                "--color-max", payoff_color_range[1],
                "--hide-numbers",
                "--hide-axis-labels",
                "--grid-linewidth", str(grid_linewidth),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    ✅ Payoff heatmap: {payoff_output}")
                print(f"    ✅ Payoff colorbar: {payoff_colorbar_output}")
            else:
                print(f"    ❌ Payoff heatmap failed: {result.stderr}")
        except Exception as e:
            print(f"    ❌ Payoff heatmap error: {e}")
        
        # Generate consumer surplus heatmap
        consumer_output = os.path.join(output_dir, f"consumer_surplus_heatmap{ext}")
        consumer_colorbar_output = os.path.join(output_dir, f"consumer_surplus_colorbar{ext}")
        
        print(f"  📈 Generating consumer surplus heatmap...")
        try:
            cmd = [
                sys.executable, "generate_consumer_surplus_heatmap.py",
                welfare_json,
                payoff_csv,
                "--heatmap-output", consumer_output,
                "--colorbar-output", consumer_colorbar_output,
                "--color-min", consumer_color_range[0],
                "--color-max", consumer_color_range[1],
                "--hide-numbers",
                "--hide-axis-labels",
                "--grid-linewidth", str(grid_linewidth),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    ✅ Consumer surplus heatmap: {consumer_output}")
                print(f"    ✅ Consumer surplus colorbar: {consumer_colorbar_output}")
            else:
                print(f"    ❌ Consumer surplus heatmap failed: {result.stderr}")
        except Exception as e:
            print(f"    ❌ Consumer surplus heatmap error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate payoff and consumer-surplus heatmaps for folders in selected_folders.json"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("pdf", "png"),
        default="pdf",
        help="Output format for heatmaps and colorbars (default: pdf).",
    )
    parser.add_argument(
        "--grid-linewidth",
        type=float,
        default=0.5,
        help="Width of white lines between cells and black outer border (default: 0.5).",
    )
    args = parser.parse_args()
    output_format = args.format
    grid_linewidth = args.grid_linewidth

    print("🚀 Starting batch heatmap generation")
    print("="*60)
    print(f"📄 Output format: .{output_format}")
    print(f"┃ Grid / border linewidth: {grid_linewidth}")

    # Load selected folders
    json_path = "selected_folders.json"
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found!")
        return
    
    selected_folders = load_selected_folders(json_path)
    
    # Process different_alpha folders
    alpha_base_path = "notable_results/different_alpha"
    if "different_alpha" in selected_folders:
        generate_heatmaps_for_category(
            alpha_base_path,
            selected_folders["different_alpha"],
            "different_alpha",
            output_format,
            grid_linewidth,
        )

    # Process different_s0 folders
    s0_base_path = "notable_results/different_s0"
    if "different_s0" in selected_folders:
        generate_heatmaps_for_category(
            s0_base_path,
            selected_folders["different_s0"],
            "different_s0",
            output_format,
            grid_linewidth,
        )

    print(f"\n{'='*60}")
    print("🎉 Batch heatmap generation completed!")
    print(f"{'='*60}")
    
    # Summary
    total_folders = len(selected_folders.get("different_alpha", [])) + len(selected_folders.get("different_s0", []))
    print(f"📊 Total folders processed: {total_folders}")
    print(f"📁 Output directory: generated_heatmaps/")
    print(f"🎨 Color ranges:")
    print(f"   📊 Different Alpha - Payoff sum: -12 to 30, Consumer surplus: -55 to 1")
    print(f"   📈 Different S0 - Payoff sum: -11 to 42, Consumer surplus: -45 to -5")
    print(f"🔢 Numbers: Hidden")
    print(f"📐 Axis tick labels: Hidden")
    print(f"📄 Output format: .{output_format}")
    print(f"┃ Grid / border linewidth: {grid_linewidth}")

if __name__ == "__main__":
    main()
