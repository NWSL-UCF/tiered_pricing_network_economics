import numpy as np
import pandas as pd

# Step 1: Simulate flows
# np.random.seed(42)
# n = 10
# dist = np.random.randint(5, 3000, size=n)
# demand = np.random.randint(50, 1000, size=n)
df = pd.read_csv("netflow_grouped_by_src_dst.csv")
df = df[df['distance'] > 0]
df['flow'] = range(len(df))

print("=== Simulated Flows ===")
# print(df)

# Step 2: Define cost and valuation
gamma, beta = 0.01, 2
alpha, s0, P0 = 2.0, 0.2, 5.0
df['cost'] = gamma * df['distance'] + beta
df['v'] = P0 * (df['demand'] ** (1 / alpha))
print("\n=== Cost and Valuation ===")
print(df[['distance', 'demand', 'cost', 'v']])

total_flat_valuation = df['v'].sum()
print(f"\n=== Aggregated Valuation for Flat Price (P0 = ${P0}) ===")
print(f"Total Valuation: {total_flat_valuation:.2f}")
# Step 3: Estimate CED profit and bundle into 3 tiers
df['ced_profit'] = (df['v']**alpha / alpha) * (
    (alpha * df['cost'] / (alpha - 1) - df['cost']) /
    (alpha * df['cost'] / (alpha - 1))**alpha
)

# df = df.sort_values('ced_profit', ascending=False).reset_index(drop=True)
# df['tier'] = pd.qcut(df.index, 3, labels=[0, 1, 2])
df['tier'] = pd.cut(df['distance'], 
                    bins=[0, 500, 2000, float('inf')], 
                    labels=[0, 1, 2])
print(df.groupby('tier')['v'].sum())
print("\n=== Tier Assignment ===")
print(df[['flow', 'distance', 'tier']])

# Step 4: Compute bundle price (stable)
tier_params = {}
for t in df['tier'].unique():
    sub = df[df['tier'] == t]
    max_v = sub['v'].max()
    weights = np.exp(alpha * (sub['v'] - max_v))
    v_b = max_v + np.log(weights.sum()) / alpha
    c_b = (sub['cost'] * weights).sum() / weights.sum()
    p_b = c_b + 1 / (alpha * s0)
    tier_params[t] = {'price': p_b}
    print(f"\nTier {t} — Price: {p_b:.2f}, Avg Cost: {c_b:.2f}, Bundle Valuation: {v_b:.2f}")

df['p_tiered'] = df['tier'].map(lambda t: tier_params[t]['price']).astype(float)
df['p_single'] = float(P0)

# Step 5: Profit function
def compute_profits(pA, pB):
    pa = pA.to_numpy(dtype=float)
    pb = pB.to_numpy(dtype=float)
    # print(pa, pb)
    shareA = 1.0 / (1.0 + np.exp(alpha * (pa - pb)))
    print(shareA)
    demands = df['demand'].to_numpy(dtype=float)
    costs = df['cost'].to_numpy(dtype=float)
    profitA = np.sum(demands * shareA * (pa - costs))
    profitB = np.sum(demands * (1.0 - shareA) * (pb - costs))
    return profitA, profitB

# Step 6: Build strategy matrix
strategies = {'Single': df['p_single'], 'Tiered': df['p_tiered']}
payoff = pd.DataFrame(index=strategies.keys(), columns=strategies.keys(), dtype=object)
for A_name, pA in strategies.items():
    for B_name, pB in strategies.items():
        piA, piB = compute_profits(pA, pB)
        payoff.loc[A_name, B_name] = (round(piA, 1), round(piB, 1))

print("\n=== Duopoly Payoff Matrix ===")
print(payoff)
