"""Pricing model for transit ISP tiered pricing strategies."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging


class PricingModel:
    
    def __init__(self, df: pd.DataFrame, tier_config: dict, logger: logging.Logger):
        self.df = df
        self.tier_config = tier_config
        self.logger = logger
        self.P0 = None
    
    def calculate_costs(self, gamma: float, beta: float) -> None:
        """Calculate cost per TB: cost = gamma * distance + beta"""
        self.df['cost_per_tb'] = gamma * self.df['distance'] + beta
    
    def calculate_valuations(self, alpha: float, s0: float) -> None:
        """Calculate valuations with endogenous P0 = avg_cost + 1/(alpha*s0)"""
        avg_cost = self.df['cost_per_tb'].mean()
        markup = 1 / (alpha * s0)
        self.P0 = avg_cost + markup
        self.df['v'] = self.P0 * (self.df['demand_tb'] ** (1 / alpha))
    
    def assign_tiers(self, n_tiers: int) -> pd.DataFrame:
        """Assign flows to tiers based on distance ranges."""
        df_copy = self.df.copy()
        
        if n_tiers == 1:
            df_copy['tier'] = 0
            df_copy['tier_name'] = 'Universal'
            return df_copy
        
        tier_key = f"{n_tiers}_tier"
        tiers_info = self.tier_config[tier_key]['tiers']
        
        df_copy['tier'] = pd.cut(
            df_copy['distance'], bins=n_tiers, labels=False
        ).astype(int)
        
        tier_name_map = {tier['tier_id']: tier['name'] for tier in tiers_info}
        df_copy['tier_name'] = df_copy['tier'].map(tier_name_map)
        
        return df_copy
    
    def calculate_tier_prices(self, df_tiers: pd.DataFrame, alpha: float, s0: float) -> Dict[int, float]:
        """Calculate optimal price per tier: p = weighted_cost + markup"""
        tier_prices = {}
        
        for tier_id in df_tiers['tier'].unique():
            sub = df_tiers[df_tiers['tier'] == tier_id]
            max_v = sub['v'].max()
            weights = np.exp(alpha * (sub['v'] - max_v))
            c_b = (sub['cost_per_tb'] * weights).sum() / weights.sum()
            tier_prices[tier_id] = c_b + 1 / (alpha * s0)
        
        return tier_prices
    
    def create_pricing_strategy(
        self, n_tiers: int, gamma: float, beta: float, alpha: float, s0: float
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create complete pricing strategy with tier assignments and prices."""
        df_strategy = self.assign_tiers(n_tiers)
        tier_prices = self.calculate_tier_prices(df_strategy, alpha, s0)
        df_strategy['price'] = df_strategy['tier'].map(tier_prices).astype(float)
        return df_strategy, df_strategy['price'].to_numpy(dtype=float)
    
    def log_strategy_summary(self, strategy_name: str, df_strategy: pd.DataFrame, n_tiers: int) -> None:
        """Log pricing strategy summary."""
        self.logger.info(f"{strategy_name}:")
        
        if n_tiers == 1:
            self.logger.info(f"  Universal: ${df_strategy['price'].iloc[0]:.2f}/TB")
        else:
            tier_summary = df_strategy.groupby(['tier', 'tier_name'], observed=True).agg({
                'distance': ['min', 'max'], 'price': 'first', 'demand_tb': 'sum'
            }).round(2)
            
            for (tier_id, tier_name) in tier_summary.index:
                dist_min = tier_summary.loc[(tier_id, tier_name), ('distance', 'min')]
                dist_max = tier_summary.loc[(tier_id, tier_name), ('distance', 'max')]
                price = tier_summary.loc[(tier_id, tier_name), ('price', 'first')]
                volume = tier_summary.loc[(tier_id, tier_name), ('demand_tb', 'sum')]
                self.logger.info(
                    f"  {tier_name:>16}: {dist_min:6.0f}-{dist_max:6.0f} mi → "
                    f"${price:6.2f}/TB ({volume:8.2f} TB)"
                )
        self.logger.info("")

