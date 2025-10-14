import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
        print(f"{tier_num}x{tier_num}: {strategy_name} vs {strategy_name}")
        print(f"  Consumer Surplus: {consumer_surplus:.4f}")
        print(f"  Producer Surplus: {producer_surplus:.4f}")
    else:
        print(f"Warning: {welfare_file} not found")

# Calculate raw welfare (direct sum: consumer + producer)
raw_welfare = [cs + ps for cs, ps in zip(consumer_surpluses, producer_surpluses)]

print(f"\nStatistics:")
print(f"Consumer Surplus - Min: {min(consumer_surpluses):.4f}, Max: {max(consumer_surpluses):.4f}")
print(f"Producer Surplus - Min: {min(producer_surpluses):.4f}, Max: {max(producer_surpluses):.4f}")
print(f"Welfare - Min: {min(raw_welfare):.4f}, Max: {max(raw_welfare):.4f}")

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Plot 1: Consumer Surplus
ax1.plot(tiers, consumer_surpluses, marker='o', linewidth=2.5, markersize=8, 
         color='#2E86AB', alpha=0.8)
ax1.set_xlabel('Number of Tiers', fontsize=12)
ax1.set_ylabel('Consumer Surplus', fontsize=12)
ax1.set_title('Consumer Surplus vs Number of Tiers (Raw Values)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(tiers)

# Plot 2: Producer Surplus
ax2.plot(tiers, producer_surpluses, marker='s', linewidth=2.5, markersize=8, 
         color='#A23B72', alpha=0.8)
ax2.set_xlabel('Number of Tiers', fontsize=12)
ax2.set_ylabel('Producer Surplus', fontsize=12)
ax2.set_title('Producer Surplus vs Number of Tiers (Raw Values)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(tiers)

# Plot 3: Welfare
ax3.plot(tiers, raw_welfare, marker='D', linewidth=2.5, markersize=8, 
         color='#F18F01', alpha=0.8)
ax3.set_xlabel('Number of Tiers', fontsize=12)
ax3.set_ylabel('Welfare (CS + PS)', fontsize=12)
ax3.set_title('Welfare vs Number of Tiers (Raw Values)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(tiers)

plt.tight_layout()

# Save the plot
output_path = base_dir / "10x10" / "raw_surplus_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Also save as PDF
output_path_pdf = base_dir / "10x10" / "raw_surplus_plot.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Plot also saved as: {output_path_pdf}")

# Create a combined plot as well
fig2, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.plot(tiers, consumer_surpluses, marker='o', linewidth=2.5, markersize=8, 
        color='#2E86AB', label='Consumer Surplus', alpha=0.8)
ax.plot(tiers, producer_surpluses, marker='s', linewidth=2.5, markersize=8, 
        color='#A23B72', label='Producer Surplus', alpha=0.8)
ax.plot(tiers, raw_welfare, marker='D', linewidth=2.5, markersize=8, 
        color='#F18F01', label='Welfare', alpha=0.8)

ax.set_xlabel('Number of Tiers', fontsize=13)
ax.set_ylabel('Surplus Value', fontsize=13)
ax.set_title('Consumer Surplus, Producer Surplus, and Welfare vs Number of Tiers (Raw Values)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(tiers)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()

# Save combined plot
output_path_combined = base_dir / "10x10" / "raw_surplus_combined_plot.png"
plt.savefig(output_path_combined, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to: {output_path_combined}")

output_path_combined_pdf = base_dir / "10x10" / "raw_surplus_combined_plot.pdf"
plt.savefig(output_path_combined_pdf, bbox_inches='tight')
print(f"Combined plot also saved as: {output_path_combined_pdf}")

plt.show()

# Save the data to a CSV file for reference
import csv
csv_output = base_dir / "10x10" / "raw_surplus_data.csv"
with open(csv_output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Number_of_Tiers', 'Consumer_Surplus', 'Producer_Surplus', 'Welfare'])
    for tier, cs, ps, w in zip(tiers, consumer_surpluses, producer_surpluses, raw_welfare):
        writer.writerow([tier, cs, ps, w])
print(f"Data saved to: {csv_output}")

