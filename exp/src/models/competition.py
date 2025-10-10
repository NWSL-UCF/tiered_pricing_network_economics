"""Game-theoretic competition analysis for ISP pricing strategies."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging


class CompetitionModel:
    
    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.df = df
        self.logger = logger
    
    def compute_profits(self, price_A: np.ndarray, price_B: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Compute ISP profits using logit market share model."""
        share_A = 1.0 / (1.0 + np.exp(alpha * (price_A - price_B)))
        demands = self.df['demand_tb'].values
        costs = self.df['cost_per_tb'].values
        profit_A = np.sum(demands * share_A * (price_A - costs))
        profit_B = np.sum(demands * (1.0 - share_A) * (price_B - costs))
        return profit_A, profit_B
    
    def build_payoff_matrix(self, strategies: Dict, alpha: float) -> pd.DataFrame:
        """Build complete payoff matrix for all strategy combinations."""
        strategy_names = list(strategies.keys())
        payoff_matrix = pd.DataFrame(index=strategy_names, columns=strategy_names, dtype=object)
        
        for name_A in strategy_names:
            for name_B in strategy_names:
                price_A = strategies[name_A]['prices']
                price_B = strategies[name_B]['prices']
                profit_A, profit_B = self.compute_profits(price_A, price_B, alpha)
                payoff_matrix.loc[name_A, name_B] = (round(profit_A, 2), round(profit_B, 2))
        
        return payoff_matrix
    
    def find_nash_equilibria(self, payoff_matrix: pd.DataFrame) -> List[Tuple]:
        """Find all pure strategy Nash Equilibria."""
        nash_equilibria = []
        strategy_names = payoff_matrix.index.tolist()
        
        for name_A in strategy_names:
            for name_B in strategy_names:
                profit_A, profit_B = payoff_matrix.loc[name_A, name_B]
                is_nash = True
                
                for alt_A in strategy_names:
                    if payoff_matrix.loc[alt_A, name_B][0] > profit_A:
                        is_nash = False
                        break
                
                if is_nash:
                    for alt_B in strategy_names:
                        if payoff_matrix.loc[name_A, alt_B][1] > profit_B:
                            is_nash = False
                            break
                
                if is_nash:
                    nash_equilibria.append((name_A, name_B, profit_A, profit_B))
        
        return nash_equilibria
    
    def analyze_best_responses(self, payoff_matrix: pd.DataFrame) -> Dict:
        """Find best response for each player given opponent's strategy."""
        strategy_names = payoff_matrix.index.tolist()
        best_responses = {'ISP_A': {}, 'ISP_B': {}}
        
        for strat_B in strategy_names:
            profits = [(strat_A, payoff_matrix.loc[strat_A, strat_B][0]) for strat_A in strategy_names]
            best_A, profit_A = max(profits, key=lambda x: x[1])
            best_responses['ISP_A'][strat_B] = (best_A, profit_A)
        
        for strat_A in strategy_names:
            profits = [(strat_B, payoff_matrix.loc[strat_A, strat_B][1]) for strat_B in strategy_names]
            best_B, profit_B = max(profits, key=lambda x: x[1])
            best_responses['ISP_B'][strat_A] = (best_B, profit_B)
        
        return best_responses
    
    def log_payoff_matrix(self, payoff_matrix: pd.DataFrame) -> None:
        """Log payoff matrix."""
        n = len(payoff_matrix)
        self.logger.info("\n" + "="*80)
        self.logger.info(f"COMPLETE PAYOFF MATRIX ({n}x{n})")
        self.logger.info("="*80)
        self.logger.info("Format: (ISP A profit, ISP B profit) in dollars")
        self.logger.info("Rows = ISP A's strategy, Columns = ISP B's strategy\n")
        self.logger.info(payoff_matrix.to_string())
    
    def log_nash_equilibria(self, nash_equilibria: List[Tuple]) -> None:
        """Log Nash Equilibria."""
        self.logger.info("\n" + "="*80)
        self.logger.info("NASH EQUILIBRIUM ANALYSIS")
        self.logger.info("="*80)
        
        if nash_equilibria:
            self.logger.info(f"\nFound {len(nash_equilibria)} Nash Equilibrium/Equilibria:\n")
            for idx, (name_A, name_B, profit_A, profit_B) in enumerate(nash_equilibria, 1):
                self.logger.info(f"{idx}. ({name_A}, {name_B}):")
                self.logger.info(f"   ISP A profit: ${profit_A:>14,.2f}")
                self.logger.info(f"   ISP B profit: ${profit_B:>14,.2f}")
                self.logger.info(f"   Total welfare: ${profit_A + profit_B:>14,.2f}\n")
        else:
            self.logger.warning("\nNo pure strategy Nash Equilibrium found!")
    
    def log_best_responses(self, best_responses: Dict) -> None:
        """Log best responses."""
        self.logger.info("="*80)
        self.logger.info("BEST RESPONSE ANALYSIS")
        self.logger.info("="*80)
        
        self.logger.info("\nBest Responses for ISP A (given B's strategy):")
        for strat_B, (best_A, profit_A) in best_responses['ISP_A'].items():
            self.logger.info(f"  If B plays {strat_B:>8} → A's best: {best_A:>8} (profit: ${profit_A:>12,.2f})")
        
        self.logger.info("\nBest Responses for ISP B (given A's strategy):")
        for strat_A, (best_B, profit_B) in best_responses['ISP_B'].items():
            self.logger.info(f"  If A plays {strat_A:>8} → B's best: {best_B:>8} (profit: ${profit_B:>12,.2f})")
