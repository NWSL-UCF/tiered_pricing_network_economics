import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("../netflow_grouped_by_src_dst.csv")
df = df[df['distance'] > 0]
df['flow'] = range(len(df))

# Parameters
gamma, beta = 0.005, 2
alpha, s0, P0 = 2.0, 0.2,10.0
df['cost'] = gamma * df['distance'] + beta
df['v'] = P0 * (df['demand'] ** (1 / alpha))

# Tier assignment
df['tier'] = pd.cut(df['distance'], bins=[0, 500, 2000, float('inf')],
                    labels=['Metro', 'Regional', 'Intercontinental'])

# Calculate tiered prices
tier_params = {}
for t in df['tier'].unique():
    sub = df[df['tier'] == t]
    max_v = sub['v'].max()
    weights = np.exp(alpha * (sub['v'] - max_v))
    v_b = max_v + np.log(weights.sum()) / alpha
    c_b = (sub['cost'] * weights).sum() / weights.sum()
    p_b = c_b + 1 / (alpha * s0)
    tier_params[t] = {'price': p_b, 'avg_cost': c_b}

# Prepare data for plotting
tiers = ['Metro', 'Regional', 'Intercontinental']
avg_costs = [tier_params[t]['avg_cost'] for t in tiers]
tiered_prices = [tier_params[t]['price'] for t in tiers]
flat_prices = [P0] * len(tiers)

x = np.arange(len(tiers))
width = 0.25

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, avg_costs, width, label='Average Cost', color='gray')
ax.bar(x, tiered_prices, width, label='Tiered Price', color='steelblue')
ax.bar(x + width, flat_prices, width, label='Flat Price ($10)', color='orange')

ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_ylabel('Price / Cost ($)')
ax.set_title('Flat vs Tiered Price vs Average Cost by Tier')
ax.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../figure/flat_vs_tiered_cost_graph.png", dpi=300)
plt.close()
