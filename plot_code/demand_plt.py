# Re-import necessary libraries after code state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("netflow_grouped_by_src_dst.csv")
df = df[df['distance'] > 0]
df['flow'] = range(len(df))

# Parameters
gamma, beta = 0.005, 2
alpha, s0, P0 = 2.0, 0.2, 5.0
df['cost'] = gamma * df['distance'] + beta
df['v'] = P0 * (df['demand'] ** (1 / alpha))

# Tier assignment based on distance
df['tier'] = pd.cut(df['distance'], bins=[0, 100, 1000, float('inf')], labels=['Metro', 'Regional', 'Intercontinental'])

# Function to calculate demand at different prices
def demand_at_price(df_tier, prices, alpha):
    demands = []
    for p in prices:
        q = (df_tier['v'] / p) ** alpha
        demands.append(q.sum())
    return demands

# Define price range
price_range = np.linspace(1, 150, 100)

# Plot demand curves for each tier
plt.figure(figsize=(10, 6))
for tier in df['tier'].dropna().unique():
    df_tier = df[df['tier'] == tier]
    demand_curve = demand_at_price(df_tier, price_range, alpha)
    plt.plot(price_range, demand_curve, label=f"{tier} Tier")


plt.title("Demand Curve by Tier (CED Model)")
plt.xlabel("Price ($)")
plt.ylabel("Total Demand (Mbps)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("demand_curve_by_tier.png", dpi=300)
plt.close()


# Redefine a smaller price range to zoom into the lower end of the curve
price_range_zoom = np.linspace(1, 20, 100)

# Plot zoomed-in demand curves for each tier
plt.figure(figsize=(10, 6))
for tier in df['tier'].dropna().unique():
    df_tier = df[df['tier'] == tier]
    demand_curve = demand_at_price(df_tier, price_range_zoom, alpha)
    plt.plot(price_range_zoom, demand_curve, label=f"{tier} Tier")

plt.title("Zoomed-In Demand Curve by Tier (Price up to $20)")
plt.xlabel("Price ($)")
plt.ylabel("Total Demand (Mbps)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("demand_curve_zoomed_up_to_20.png", dpi=300)
plt.close()

