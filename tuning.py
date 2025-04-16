import numpy as np
import pandas as pd

# Load and clean dataset
df = pd.read_csv("netflow_grouped_by_src_dst.csv")
df = df[(df['distance'] > 0) & (df['demand'] > 0)].copy()
df['flow'] = range(len(df))

# Function to simulate and return duopoly payoff for a given (alpha, P0, gamma, beta)
def simulate_duopoly(alpha, P0, gamma, beta, s0=0.2, tiers=3):
    df_local = df.copy()
    df_local['cost'] = gamma * df_local['distance'] + beta
    df_local['v'] = P0 * (df_local['demand'] ** (1 / alpha))

    # Calculate profit
    df_local['ced_profit'] = (df_local['v']**alpha / alpha) * (
        (alpha * df_local['cost'] / (alpha - 1) - df_local['cost']) /
        (alpha * df_local['cost'] / (alpha - 1))**alpha
    )
    df_local = df_local.dropna(subset=['ced_profit'])
    df_local = df_local.sort_values('ced_profit', ascending=False).reset_index(drop=True)
    df_local['tier'] = pd.qcut(df_local.index, tiers, labels=range(tiers))

    tier_params = {}
    for t in df_local['tier'].unique():
        sub = df_local[df_local['tier'] == t]
        max_v = sub['v'].max()
        weights = np.exp(alpha * (sub['v'] - max_v))
        v_b = max_v + np.log(weights.sum()) / alpha
        c_b = (sub['cost'] * weights).sum() / weights.sum()
        p_b = c_b + 1 / (alpha * s0)
        tier_params[t] = {'price': p_b}

    df_local['p_tiered'] = df_local['tier'].map(lambda t: tier_params[t]['price']).astype(float)
    df_local['p_single'] = float(P0)

    def compute_profits(pA, pB):
        pa = pA.to_numpy(dtype=float)
        pb = pB.to_numpy(dtype=float)
        shareA = 1.0 / (1.0 + np.exp(alpha * (pa - pb)))
        demands = df_local['demand'].to_numpy(dtype=float)
        costs = df_local['cost'].to_numpy(dtype=float)
        profitA = np.sum(demands * shareA * (pa - costs))
        profitB = np.sum(demands * (1.0 - shareA) * (pb - costs))
        return round(profitA, 1), round(profitB, 1)

    strategies = {'Single': df_local['p_single'], 'Tiered': df_local['p_tiered']}
    result = {}
    for A_name, pA in strategies.items():
        for B_name, pB in strategies.items():
            piA, piB = compute_profits(pA, pB)
            result[(A_name, B_name)] = (piA, piB)

    return {
        "alpha": alpha,
        "P0": P0,
        "gamma": gamma,
        "beta": beta,
        "Single,Single": result[("Single", "Single")],
        "Single,Tiered": result[("Single", "Tiered")],
        "Tiered,Single": result[("Tiered", "Single")],
        "Tiered,Tiered": result[("Tiered", "Tiered")]
    }

# Parameter grid
alphas = [1.5, 2.0, 3.0]
P0s = [15, 20, 25]
gammas = [0.005, 0.01]
betas = [1, 5]

results = []
for alpha in alphas:
    for P0 in P0s:
        for gamma in gammas:
            for beta in betas:
                result = simulate_duopoly(alpha, P0, gamma, beta)
                results.append(result)

results_df = pd.DataFrame(results)
print(results_df)