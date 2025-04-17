import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../netflow_grouped_by_src_dst.csv")

# Filter out invalid distances
df = df[df['distance'] > 0]

# Assign tier based on distance
df['tier'] = pd.cut(
    df['distance'],
    bins=[0, 500, 2000, float('inf')],
    labels=['Metro', 'Regional', 'Intercontinental']
)

# Group by tier
grouped = df.groupby('tier').agg(
    num_flows=('distance', 'count'),
    avg_demand=('demand', 'mean')
).reset_index()

# Plotting
fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar for number of flows
ax1.bar(grouped['tier'], grouped['num_flows'], color='skyblue', label='Number of Flows')
ax1.set_ylabel('Number of Flows')
ax1.set_xlabel('Tier')
ax1.set_title('Flows per Tier and Average Demand (Mbps)')

# Line for average demand
ax2 = ax1.twinx()
ax2.plot(grouped['tier'], grouped['avg_demand'], color='orange', marker='o', label='Avg Demand (Mbps)')
ax2.set_ylabel('Average Demand (Mbps)')

# Combine legends
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.6)
plt.savefig("../figure/flowcount_flowdemand_comparision_by_tiers_new.png")
