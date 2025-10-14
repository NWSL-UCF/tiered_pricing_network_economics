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
        print(f"{tier_num}x{tier_num}: {strategy_name} vs {strategy_name} -> Consumer Surplus: {consumer_surplus:.4f}, Producer Surplus: {producer_surplus:.4f}")
    else:
        print(f"Warning: {welfare_file} not found")

# Normalize consumer surplus
min_cs = min(consumer_surpluses)
max_cs = max(consumer_surpluses)
print(f"\nConsumer Surplus - Min: {min_cs:.4f}, Max: {max_cs:.4f}")

normalized_consumer_surpluses = [(cs - min_cs) / (max_cs - min_cs) for cs in consumer_surpluses]

# Normalize producer surplus
min_ps = min(producer_surpluses)
max_ps = max(producer_surpluses)
print(f"Producer Surplus - Min: {min_ps:.4f}, Max: {max_ps:.4f}")

normalized_producer_surpluses = [(ps - min_ps) / (max_ps - min_ps) for ps in producer_surpluses]

# Calculate normalized welfare as the average of normalized consumer and producer surplus
normalized_welfare = [0.5 * norm_cs + 0.5 * norm_ps 
                      for norm_cs, norm_ps in zip(normalized_consumer_surpluses, normalized_producer_surpluses)]
print(f"\nNormalized Welfare calculated for each tier configuration")

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(tiers, normalized_consumer_surpluses, marker='o', linewidth=2.5, markersize=8, 
         color='#2E86AB', label='Consumer Surplus', alpha=0.8)
plt.plot(tiers, normalized_producer_surpluses, marker='s', linewidth=2.5, markersize=8, 
         color='#A23B72', label='Producer Surplus', alpha=0.8)
plt.plot(tiers, normalized_welfare, marker='D', linewidth=2.5, markersize=8, 
         color='#F18F01', label='Normalized Welfare', alpha=0.8)

plt.xlabel('Number of Tiers', fontsize=13)
plt.ylabel('Normalized Surplus', fontsize=13)
plt.title('Consumer Surplus, Producer Surplus, and Normalized Welfare vs Number of Tiers', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(tiers)
plt.ylim(-0.05, 1.1)
plt.legend(fontsize=11, loc='best')

plt.tight_layout()

# Save the plot
output_path = base_dir / "10x10" / "consumer_surplus_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Also save as PDF
output_path_pdf = base_dir / "10x10" / "consumer_surplus_plot.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Plot also saved as: {output_path_pdf}")

plt.show()

# Save the data to a CSV file for reference
import csv
csv_output = base_dir / "10x10" / "consumer_producer_surplus_data.csv"
with open(csv_output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Number_of_Tiers', 'Consumer_Surplus', 'Normalized_Consumer_Surplus', 
                     'Producer_Surplus', 'Normalized_Producer_Surplus', 'Normalized_Welfare'])
    for tier, cs, norm_cs, ps, norm_ps, norm_w in zip(tiers, consumer_surpluses, normalized_consumer_surpluses, 
                                                        producer_surpluses, normalized_producer_surpluses, normalized_welfare):
        writer.writerow([tier, cs, norm_cs, ps, norm_ps, norm_w])
print(f"Data saved to: {csv_output}")

