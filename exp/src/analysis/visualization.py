"""Visualization for payoff matrices and competition results."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging


class PayoffMatrixVisualizer:
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def visualize_payoff_matrix(self, payoff_matrix: pd.DataFrame, nash_equilibria: List[Tuple],
                                parameters: Dict, output_path: str = "payoff_matrix.pdf") -> None:
        """Create payoff matrix heatmap with Nash Equilibria highlighted."""
        strategy_names = payoff_matrix.index.tolist()
        n = len(strategy_names)
        
        profit_A = np.zeros((n, n))
        profit_B = np.zeros((n, n))
        total_welfare = np.zeros((n, n))
        
        for i, name_A in enumerate(strategy_names):
            for j, name_B in enumerate(strategy_names):
                pA, pB = payoff_matrix.loc[name_A, name_B]
                profit_A[i, j] = pA
                profit_B[i, j] = pB
                total_welfare[i, j] = pA + pB
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        im = ax.imshow(total_welfare, cmap='YlGn', aspect='auto', alpha=0.6)
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(strategy_names, rotation=0, ha='center', fontsize=14, fontweight='bold')
        ax.set_yticklabels(strategy_names, fontsize=14, fontweight='bold')
        ax.set_xlabel('ISP B Strategy', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel('ISP A Strategy', fontsize=18, fontweight='bold', labelpad=15)
        
        nash_coords = [(strategy_names.index(nA), strategy_names.index(nB)) for nA, nB, _, _ in nash_equilibria]
        
        ax.set_xticks(np.arange(n) - 0.5, minor=True)
        ax.set_yticks(np.arange(n) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
        
        self._add_cell_values(ax, n, profit_A, profit_B, nash_coords)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Total Social Welfare ($)', rotation=270, labelpad=25, fontsize=16, fontweight='bold')
        
        self._add_title(ax, parameters, nash_equilibria)
        self._add_legend(ax)
        
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        self.logger.info(f"Payoff matrix visualization saved to '{output_path}'")
        plt.close()
    
    def _add_cell_values(self, ax, n: int, profit_A: np.ndarray, profit_B: np.ndarray, nash_coords: List[Tuple]) -> None:
        """Add profit values to cells."""
        for i in range(n):
            for j in range(n):
                is_nash = (i, j) in nash_coords
                
                if is_nash:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor='gold', 
                                        alpha=0.4, edgecolor='red', linewidth=4)
                    ax.add_patch(rect)
                
                pA, pB = profit_A[i, j], profit_B[i, j]
                text_str = f"A: ${pA:.2f}\nB: ${pB:.2f}\n★ NE ★" if is_nash else f"A: ${pA:.2f}\nB: ${pB:.2f}"
                color, weight, size = ('darkred', 'bold', 12) if is_nash else ('black', 'normal', 11)
                
                ax.text(j, i, text_str, ha="center", va="center", color=color, 
                       fontsize=size, fontweight=weight,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.8))
    
    def _add_title(self, ax, parameters: Dict, nash_equilibria: List[Tuple]) -> None:
        """Add title with parameters."""
        max_tiers = parameters.get('max_tiers', 10)
        param_str = f"{max_tiers}x{max_tiers} Matrix | P0=${parameters['P0']:.2f}, γ={parameters['gamma']}, β={parameters['beta']}, α={parameters['alpha']}, s0={parameters['s0']}"
        title_text = f'Transit ISP Pricing Competition\n{param_str}'
        
        if nash_equilibria:
            ne_list = ", ".join([f"({nA}, {nB})" for nA, nB, _, _ in nash_equilibria])
            title_text += f"\nNash Equilibrium: {ne_list}"
        
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25)
    
    def _add_legend(self, ax) -> None:
        """Add legend."""
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='gold', alpha=0.4, edgecolor='red', linewidth=2, label='Nash Equilibrium'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=0, 
                      label='A = ISP A profit, B = ISP B profit')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
