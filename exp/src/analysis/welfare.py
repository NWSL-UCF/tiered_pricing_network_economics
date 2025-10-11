"""Welfare analysis for calculating consumer surplus, producer surplus, and efficiency metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging


class WelfareAnalyzer:
    
    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.df = df
        self.logger = logger
    
    def calculate_consumer_surplus(self, price_A: np.ndarray, price_B: np.ndarray, alpha: float) -> float:
        """Calculate consumer surplus: sum of (valuation - expected price paid)."""
        share_A = 1.0 / (1.0 + np.exp(alpha * (price_A - price_B)))
        expected_price = share_A * price_A + (1.0 - share_A) * price_B
        cs_per_flow = self.df['v'].values - expected_price
        return np.sum(self.df['demand_tb'].values * cs_per_flow)
    
    def calculate_producer_surplus(self, price_A: np.ndarray, price_B: np.ndarray, alpha: float) -> float:
        """Calculate producer surplus: sum of ISP profits."""
        share_A = 1.0 / (1.0 + np.exp(alpha * (price_A - price_B)))
        demands = self.df['demand_tb'].values
        costs = self.df['cost_per_tb'].values
        profit_A = np.sum(demands * share_A * (price_A - costs))
        profit_B = np.sum(demands * (1.0 - share_A) * (price_B - costs))
        return profit_A + profit_B
    
    def calculate_social_welfare(self, price_A: np.ndarray, price_B: np.ndarray, alpha: float) -> Dict[str, float]:
        """Calculate comprehensive welfare components."""
        consumer_surplus = self.calculate_consumer_surplus(price_A, price_B, alpha)
        producer_surplus = self.calculate_producer_surplus(price_A, price_B, alpha)
        total_welfare = consumer_surplus + producer_surplus
        max_possible_welfare = np.sum(self.df['v'].values * self.df['demand_tb'].values)
        
        return {
            'consumer_surplus': consumer_surplus,
            'producer_surplus': producer_surplus, 
            'total_welfare': total_welfare,
            'max_possible_welfare': max_possible_welfare,
            'efficiency_ratio': total_welfare / max_possible_welfare if max_possible_welfare > 0 else 0,
            'deadweight_loss': max_possible_welfare - total_welfare
        }
    
    def analyze_comprehensive_welfare(self, payoff_matrix: pd.DataFrame, strategies: Dict, alpha: float) -> Dict:
        """Analyze welfare across all strategy combinations."""
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE SOCIAL WELFARE ANALYSIS")
        self.logger.info("="*80)
        
        welfare_results = []
        for name_A in payoff_matrix.index:
            for name_B in payoff_matrix.columns:
                price_A = strategies[name_A]['prices']
                price_B = strategies[name_B]['prices']
                welfare = self.calculate_social_welfare(price_A, price_B, alpha)
                profit_A, profit_B = payoff_matrix.loc[name_A, name_B]
                
                welfare_results.append({
                    'strategy_A': name_A, 'strategy_B': name_B,
                    'consumer_surplus': welfare['consumer_surplus'],
                    'producer_surplus': welfare['producer_surplus'],
                    'total_welfare': welfare['total_welfare'],
                    'efficiency_ratio': welfare['efficiency_ratio'],
                    'deadweight_loss': welfare['deadweight_loss'],
                    'profit_A': profit_A, 'profit_B': profit_B
                })
        
        max_total_welfare = max(welfare_results, key=lambda x: x['total_welfare'])
        max_consumer_surplus = max(welfare_results, key=lambda x: x['consumer_surplus'])
        max_producer_surplus = max(welfare_results, key=lambda x: x['producer_surplus'])
        max_efficiency = max(welfare_results, key=lambda x: x['efficiency_ratio'])
        
        self._log_optimal_outcomes(max_total_welfare, max_consumer_surplus, max_producer_surplus, max_efficiency)
        
        return {
            'welfare_results': welfare_results,
            'max_total_welfare': max_total_welfare,
            'max_consumer_surplus': max_consumer_surplus,
            'max_producer_surplus': max_producer_surplus,
            'max_efficiency': max_efficiency
        }
    
    def analyze_nash_welfare(self, nash_equilibria: List[Tuple], welfare_results: List[Dict], max_total_welfare: Dict) -> None:
        """Analyze welfare at Nash Equilibria."""
        self.logger.info(f"\nNASH EQUILIBRIUM WELFARE ANALYSIS:")
        
        for idx, (name_A, name_B, profit_A, profit_B) in enumerate(nash_equilibria, 1):
            ne_welfare = next((w for w in welfare_results 
                             if w['strategy_A'] == name_A and w['strategy_B'] == name_B), None)
            
            if ne_welfare:
                self.logger.info(f"\n   NE {idx}: ({name_A}, {name_B})")
                self.logger.info(f"   Consumer Surplus: ${ne_welfare['consumer_surplus']:>14,.2f}")
                self.logger.info(f"   Producer Surplus: ${ne_welfare['producer_surplus']:>14,.2f}")
                self.logger.info(f"   Total Welfare: ${ne_welfare['total_welfare']:>14,.2f}")
                self.logger.info(f"   Efficiency: {ne_welfare['efficiency_ratio']:.1%}")
                welfare_gap = max_total_welfare['total_welfare'] - ne_welfare['total_welfare']
                self.logger.info(f"   Welfare Gap vs Max: ${welfare_gap:>14,.2f}")
    
    def _log_optimal_outcomes(self, max_total_welfare: Dict, max_consumer_surplus: Dict, 
                              max_producer_surplus: Dict, max_efficiency: Dict) -> None:
        """Log optimal welfare outcomes."""
        self.logger.info(f"\nMAXIMUM TOTAL SOCIAL WELFARE:")
        self.logger.info(f"   Strategy: ({max_total_welfare['strategy_A']}, {max_total_welfare['strategy_B']})")
        self.logger.info(f"   Consumer Surplus: ${max_total_welfare['consumer_surplus']:>14,.2f}")
        self.logger.info(f"   Producer Surplus: ${max_total_welfare['producer_surplus']:>14,.2f}")
        self.logger.info(f"   Total Welfare: ${max_total_welfare['total_welfare']:>14,.2f}")
        self.logger.info(f"   Efficiency: {max_total_welfare['efficiency_ratio']:.1%}")
        
        self.logger.info(f"\nMAXIMUM CONSUMER SURPLUS:")
        self.logger.info(f"   Strategy: ({max_consumer_surplus['strategy_A']}, {max_consumer_surplus['strategy_B']})")
        self.logger.info(f"   Consumer Surplus: ${max_consumer_surplus['consumer_surplus']:>14,.2f}")
        self.logger.info(f"   Producer Surplus: ${max_consumer_surplus['producer_surplus']:>14,.2f}")
        self.logger.info(f"   Total Welfare: ${max_consumer_surplus['total_welfare']:>14,.2f}")
        
        self.logger.info(f"\nMAXIMUM PRODUCER SURPLUS:")
        self.logger.info(f"   Strategy: ({max_producer_surplus['strategy_A']}, {max_producer_surplus['strategy_B']})")
        self.logger.info(f"   Consumer Surplus: ${max_producer_surplus['consumer_surplus']:>14,.2f}")
        self.logger.info(f"   Producer Surplus: ${max_producer_surplus['producer_surplus']:>14,.2f}")
        self.logger.info(f"   Total Welfare: ${max_producer_surplus['total_welfare']:>14,.2f}")
        
        self.logger.info(f"\nMAXIMUM EFFICIENCY:")
        self.logger.info(f"   Strategy: ({max_efficiency['strategy_A']}, {max_efficiency['strategy_B']})")
        self.logger.info(f"   Efficiency: {max_efficiency['efficiency_ratio']:.1%}")
        self.logger.info(f"   Deadweight Loss: ${max_efficiency['deadweight_loss']:>14,.2f}")
