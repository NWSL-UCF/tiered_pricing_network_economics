import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List
import itertools
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import logging
from datetime import datetime
import os
from pathlib import Path


class TransitISPCompetitionModel:
	"""
	Model for analyzing transit ISP pricing competition with multiple tier strategies.
	"""
	
	def __init__(self, data_path: str, tier_config_path: str, logger: logging.Logger = None):
		"""
		Initialize the model with network flow data and tier configurations.
		
		Args:
			data_path: Path to netflow CSV file
			tier_config_path: Path to tier strategy JSON configuration
			logger: Logger instance (optional)
		"""
		self.logger = logger or logging.getLogger(__name__)
		
		# Load data
		self.df = pd.read_csv(data_path)
		self.df = self.df[self.df['distance'] > 0].reset_index(drop=True)
		self.df['demand_tb'] = self.df['demand'] / 1e6  # Convert Mb to TB
		
		# Load tier configurations
		with open(tier_config_path, 'r') as f:
			self.tier_config = json.load(f)
		
		self.logger.info(f"Loaded {len(self.df)} network flows")
		self.logger.info(f"Total volume: {self.df['demand_tb'].sum():.2f} TB")
		self.logger.info(f"Distance range: {self.df['distance'].min():.0f} - {self.df['distance'].max():.0f} miles")
	
	
	def calculate_costs(self, gamma: float, beta: float) -> None:
		"""
		Calculate cost per TB for each flow based on distance.
		
		Args:
			gamma: Variable cost per mile per TB ($/mile/TB)
			beta: Fixed cost per TB ($/TB)
		"""
		self.df['cost_per_tb'] = gamma * self.df['distance'] + beta
		self.logger.debug(f"Calculated costs with γ={gamma}, β={beta}")
	
	
	def calculate_valuations(self, alpha: float, s0: float) -> None:
		"""
		Calculate customer valuations for each flow.
		
		Valuations are based on an endogenous flat pricing equilibrium:
		P0 = average_cost + markup
		
		Args:
			alpha: Price elasticity parameter
			s0: Competition intensity parameter
		"""
		# Calculate P0 endogenously from average cost structure
		avg_cost = self.df['cost_per_tb'].mean()
		markup = 1 / (alpha * s0)
		P0 = avg_cost + markup
		
		self.df['v'] = P0 * (self.df['demand_tb'] ** (1 / alpha))
		self.logger.debug(f"Calculated valuations with endogenous P0=${P0:.2f} (avg_cost=${avg_cost:.2f}, markup=${markup:.2f}), α={alpha}")
		
		# Store P0 for reference
		self.P0 = P0
	
	
	def assign_tiers(self, n_tiers: int) -> pd.DataFrame:
		"""
		Assign flows to tiers based on distance percentiles from config.
		
		Args:
			n_tiers: Number of tiers (1-7)
		
		Returns:
			DataFrame with tier assignments and names
		"""
		df_copy = self.df.copy()
		
		if n_tiers == 1:
			# Flat pricing - all flows in one tier
			df_copy['tier'] = 0
			df_copy['tier_name'] = 'Universal'
			return df_copy
		
		# Get tier configuration
		tier_key = f"{n_tiers}_tier"
		tiers_info = self.tier_config[tier_key]['tiers']
		
		# Create tier boundaries based on distance quantiles
		# Use labels=False to get numeric labels instead of categorical
		df_copy['tier'] = pd.qcut(
			df_copy['distance'], 
			q=n_tiers, 
			labels=False,
			duplicates='drop'
		)
		
		# Ensure tier is integer type
		df_copy['tier'] = df_copy['tier'].astype(int)
		
		# Map tier names
		tier_name_map = {tier['tier_id']: tier['name'] for tier in tiers_info}
		df_copy['tier_name'] = df_copy['tier'].map(tier_name_map)
		
		return df_copy
	
	
	def calculate_tier_prices(self, df_tiers: pd.DataFrame, alpha: float, s0: float) -> Dict[int, float]:
		"""
		Calculate optimal price for each tier using bundle pricing formula.
		
		Args:
			df_tiers: DataFrame with tier assignments
			alpha: Price elasticity parameter
			s0: Competition intensity parameter
		
		Returns:
			Dictionary mapping tier_id to price
		"""
		tier_prices = {}
		
		unique_tiers = df_tiers['tier'].unique()
		
		# Calculate price per tier (including flat pricing as single tier)
		for tier_id in unique_tiers:
			sub = df_tiers[df_tiers['tier'] == tier_id]
			
			# Bundle valuation calculation
			max_v = sub['v'].max()
			weights = np.exp(alpha * (sub['v'] - max_v))
			
			# Weighted average cost
			c_b = (sub['cost_per_tb'] * weights).sum() / weights.sum()
			
			# Price = cost + markup
			markup = 1 / (alpha * s0)
			p_b = c_b + markup
			
			tier_prices[tier_id] = p_b
		
		return tier_prices
	
	
	def create_pricing_strategy(self, n_tiers: int, gamma: float, 
	                            beta: float, alpha: float, s0: float) -> Tuple[pd.DataFrame, np.ndarray]:
		"""
		Create complete pricing strategy for given number of tiers.
		
		Args:
			n_tiers: Number of tiers (1-10)
			gamma, beta, alpha, s0: Model parameters
		
		Returns:
			Tuple of (DataFrame with tier info, price array for each flow)
		"""
		# Assign tiers
		df_strategy = self.assign_tiers(n_tiers)
		
		# Calculate tier prices
		tier_prices = self.calculate_tier_prices(df_strategy, alpha, s0)
		
		# Map prices to flows and convert to float
		df_strategy['price'] = df_strategy['tier'].map(tier_prices).astype(float)
		
		# Return DataFrame and prices as numpy array
		return df_strategy, df_strategy['price'].to_numpy(dtype=float)
	
	
	def compute_profits(self, price_A: np.ndarray, price_B: np.ndarray, 
	                   alpha: float) -> Tuple[float, float]:
		"""
		Compute profits for both ISPs using logit market share model.
		
		Args:
			price_A: Price array for ISP A
			price_B: Price array for ISP B
			alpha: Price elasticity parameter
		
		Returns:
			Tuple of (profit_A, profit_B)
		"""
		# Logit market share
		share_A = 1.0 / (1.0 + np.exp(alpha * (price_A - price_B)))
		
		# Get data arrays
		demands = self.df['demand_tb'].values
		costs = self.df['cost_per_tb'].values
		
		# Calculate profits
		profit_A = np.sum(demands * share_A * (price_A - costs))
		profit_B = np.sum(demands * (1.0 - share_A) * (price_B - costs))
		
		return profit_A, profit_B
	
	
	def calculate_consumer_surplus(self, price_A: np.ndarray, price_B: np.ndarray, 
	                              alpha: float) -> float:
		"""
		Calculate consumer surplus for all flows.
		
		Consumer Surplus = Sum of (Valuation - Price Paid) for each flow
		
		Args:
			price_A: Price array for ISP A
			price_B: Price array for ISP B  
			alpha: Price elasticity parameter
			
		Returns:
			Total consumer surplus
		"""
		# Market shares
		share_A = 1.0 / (1.0 + np.exp(alpha * (price_A - price_B)))
		share_B = 1.0 - share_A
		
		# Expected price paid by consumers (weighted average)
		expected_price = share_A * price_A + share_B * price_B
		
		# Consumer surplus per flow = Valuation - Expected Price
		cs_per_flow = self.df['v'].values - expected_price
		
		# Total consumer surplus (weighted by demand volume)
		total_cs = np.sum(self.df['demand_tb'].values * cs_per_flow)
		
		return total_cs


	def calculate_producer_surplus(self, price_A: np.ndarray, price_B: np.ndarray, 
	                               alpha: float) -> float:
		"""
		Calculate total producer surplus (ISP profits).
		
		Args:
			price_A: Price array for ISP A
			price_B: Price array for ISP B
			alpha: Price elasticity parameter
			
		Returns:
			Total producer surplus
		"""
		profit_A, profit_B = self.compute_profits(price_A, price_B, alpha)
		return profit_A + profit_B


	def calculate_social_welfare(self, price_A: np.ndarray, price_B: np.ndarray, 
	                            alpha: float) -> Dict[str, float]:
		"""
		Calculate comprehensive social welfare components.
		
		Args:
			price_A: Price array for ISP A
			price_B: Price array for ISP B
			alpha: Price elasticity parameter
			
		Returns:
			Dictionary with welfare components
		"""
		# Calculate components
		consumer_surplus = self.calculate_consumer_surplus(price_A, price_B, alpha)
		producer_surplus = self.calculate_producer_surplus(price_A, price_B, alpha)
		total_welfare = consumer_surplus + producer_surplus
		
		# Calculate efficiency metrics
		max_possible_welfare = np.sum(self.df['v'].values * self.df['demand_tb'].values)
		efficiency_ratio = total_welfare / max_possible_welfare if max_possible_welfare > 0 else 0
		deadweight_loss = max_possible_welfare - total_welfare
		
		return {
			'consumer_surplus': consumer_surplus,
			'producer_surplus': producer_surplus, 
			'total_welfare': total_welfare,
			'max_possible_welfare': max_possible_welfare,
			'efficiency_ratio': efficiency_ratio,
			'deadweight_loss': deadweight_loss
		}


	def analyze_comprehensive_welfare(self, payoff_matrix: pd.DataFrame, 
	                                strategies: Dict, alpha: float) -> Dict:
		"""
		Comprehensive social welfare analysis for all strategy combinations.
		
		Args:
			payoff_matrix: Payoff matrix DataFrame
			strategies: Dictionary of strategies
			alpha: Price elasticity parameter
			
		Returns:
			Dictionary with welfare analysis results
		"""
		self.logger.info("="*80)
		self.logger.info("COMPREHENSIVE SOCIAL WELFARE ANALYSIS")
		self.logger.info("="*80)
		
		welfare_results = []
		
		for name_A in payoff_matrix.index:
			for name_B in payoff_matrix.columns:
				price_A = strategies[name_A]['prices']
				price_B = strategies[name_B]['prices']
				
				# Calculate welfare components
				welfare = self.calculate_social_welfare(price_A, price_B, alpha)
				
				# Get profits for reference
				profit_A, profit_B = self.compute_profits(price_A, price_B, alpha)
				
				welfare_results.append({
					'strategy_A': name_A,
					'strategy_B': name_B,
					'consumer_surplus': welfare['consumer_surplus'],
					'producer_surplus': welfare['producer_surplus'],
					'total_welfare': welfare['total_welfare'],
					'efficiency_ratio': welfare['efficiency_ratio'],
					'deadweight_loss': welfare['deadweight_loss'],
					'profit_A': profit_A,
					'profit_B': profit_B
				})
		
		# Find optimal outcomes
		max_total_welfare = max(welfare_results, key=lambda x: x['total_welfare'])
		max_consumer_surplus = max(welfare_results, key=lambda x: x['consumer_surplus'])
		max_producer_surplus = max(welfare_results, key=lambda x: x['producer_surplus'])
		max_efficiency = max(welfare_results, key=lambda x: x['efficiency_ratio'])
		
		self.logger.info(f"\n🏆 MAXIMUM TOTAL SOCIAL WELFARE:")
		self.logger.info(f"   Strategy: ({max_total_welfare['strategy_A']}, {max_total_welfare['strategy_B']})")
		self.logger.info(f"   Consumer Surplus: ${max_total_welfare['consumer_surplus']:>14,.2f}")
		self.logger.info(f"   Producer Surplus: ${max_total_welfare['producer_surplus']:>14,.2f}")
		self.logger.info(f"   Total Welfare: ${max_total_welfare['total_welfare']:>14,.2f}")
		self.logger.info(f"   Efficiency: {max_total_welfare['efficiency_ratio']:.1%}")
		
		self.logger.info(f"\n👥 MAXIMUM CONSUMER SURPLUS:")
		self.logger.info(f"   Strategy: ({max_consumer_surplus['strategy_A']}, {max_consumer_surplus['strategy_B']})")
		self.logger.info(f"   Consumer Surplus: ${max_consumer_surplus['consumer_surplus']:>14,.2f}")
		self.logger.info(f"   Producer Surplus: ${max_consumer_surplus['producer_surplus']:>14,.2f}")
		self.logger.info(f"   Total Welfare: ${max_consumer_surplus['total_welfare']:>14,.2f}")
		
		self.logger.info(f"\n🏢 MAXIMUM PRODUCER SURPLUS:")
		self.logger.info(f"   Strategy: ({max_producer_surplus['strategy_A']}, {max_producer_surplus['strategy_B']})")
		self.logger.info(f"   Consumer Surplus: ${max_producer_surplus['consumer_surplus']:>14,.2f}")
		self.logger.info(f"   Producer Surplus: ${max_producer_surplus['producer_surplus']:>14,.2f}")
		self.logger.info(f"   Total Welfare: ${max_producer_surplus['total_welfare']:>14,.2f}")
		
		self.logger.info(f"\n⚡ MAXIMUM EFFICIENCY:")
		self.logger.info(f"   Strategy: ({max_efficiency['strategy_A']}, {max_efficiency['strategy_B']})")
		self.logger.info(f"   Efficiency: {max_efficiency['efficiency_ratio']:.1%}")
		self.logger.info(f"   Deadweight Loss: ${max_efficiency['deadweight_loss']:>14,.2f}")
		
		# Nash Equilibrium welfare analysis
		self.logger.info(f"\n🎯 NASH EQUILIBRIUM WELFARE ANALYSIS:")
		nash_equilibria = self.find_nash_equilibria(payoff_matrix)
		
		for idx, (name_A, name_B, profit_A, profit_B) in enumerate(nash_equilibria, 1):
			# Find welfare for this NE
			ne_welfare = next((w for w in welfare_results 
			                  if w['strategy_A'] == name_A and w['strategy_B'] == name_B), None)
			
			if ne_welfare:
				self.logger.info(f"\n   NE {idx}: ({name_A}, {name_B})")
				self.logger.info(f"   Consumer Surplus: ${ne_welfare['consumer_surplus']:>14,.2f}")
				self.logger.info(f"   Producer Surplus: ${ne_welfare['producer_surplus']:>14,.2f}")
				self.logger.info(f"   Total Welfare: ${ne_welfare['total_welfare']:>14,.2f}")
				self.logger.info(f"   Efficiency: {ne_welfare['efficiency_ratio']:.1%}")
				
				# Compare to maximum welfare
				welfare_gap = max_total_welfare['total_welfare'] - ne_welfare['total_welfare']
				self.logger.info(f"   Welfare Gap vs Max: ${welfare_gap:>14,.2f}")
		
		return {
			'welfare_results': welfare_results,
			'max_total_welfare': max_total_welfare,
			'max_consumer_surplus': max_consumer_surplus,
			'max_producer_surplus': max_producer_surplus,
			'max_efficiency': max_efficiency
		}
	
	
	def build_payoff_matrix(self, gamma: float, beta: float, 
	                       alpha: float, s0: float) -> Tuple[pd.DataFrame, Dict]:
		"""
		Build complete payoff matrix for all strategy combinations.
		
		Args:
			gamma, beta, alpha, s0: Model parameters
		
		Returns:
			Tuple of (payoff matrix DataFrame, strategies dictionary)
		"""
		self.logger.info("="*80)
		self.logger.info("BUILDING PAYOFF MATRIX")
		self.logger.info("="*80)
		
		# Calculate costs and valuations (same for all strategies)
		self.calculate_costs(gamma, beta)
		self.calculate_valuations(alpha, s0)
		
		# Log parameters including endogenous P0
		self.logger.info(f"Parameters: γ={gamma}, β={beta}, α={alpha}, s0={s0}")
		self.logger.info(f"Endogenous P0=${self.P0:.2f}/TB\n")
		
		# Generate all strategies (1-tier through 10-tier)
		strategy_names = []
		strategies = {}
		
		for n_tiers in range(1, 11):
			if n_tiers == 1:
				strategy_name = "Flat"
			else:
				strategy_name = f"{n_tiers}-Tier"
			
			strategy_names.append(strategy_name)
			
			# Create pricing strategy
			df_strategy, prices = self.create_pricing_strategy(
				n_tiers, gamma, beta, alpha, s0
			)
			strategies[strategy_name] = {
				'prices': prices,
				'data': df_strategy
			}
			
			# Print strategy summary
			self.logger.info(f"{strategy_name}:")
			if n_tiers == 1:
				self.logger.info(f"  Universal: ${prices[0]:.2f}/TB")
			else:
				# Use observed=True to avoid FutureWarning
				tier_summary = df_strategy.groupby(['tier', 'tier_name'], observed=True).agg({
					'distance': ['min', 'max', 'mean'],
					'price': 'first',
					'demand_tb': 'sum'
				}).round(2)
				
				for (tier_id, tier_name) in tier_summary.index:
					dist_min = tier_summary.loc[(tier_id, tier_name), ('distance', 'min')]
					dist_max = tier_summary.loc[(tier_id, tier_name), ('distance', 'max')]
					price = tier_summary.loc[(tier_id, tier_name), ('price', 'first')]
					volume = tier_summary.loc[(tier_id, tier_name), ('demand_tb', 'sum')]
					self.logger.info(f"  {tier_name:>16}: {dist_min:6.0f}-{dist_max:6.0f} mi → ${price:6.2f}/TB ({volume:8.2f} TB)")
			self.logger.info("")
		
		# Build payoff matrix
		self.logger.info("Computing all strategy combinations...")
		payoff_matrix = pd.DataFrame(
			index=strategy_names,
			columns=strategy_names,
			dtype=object
		)
		
		total_computations = len(strategy_names) ** 2
		current = 0
		
		for name_A in strategy_names:
			for name_B in strategy_names:
				current += 1
				
				price_A = strategies[name_A]['prices']
				price_B = strategies[name_B]['prices']
				
				profit_A, profit_B = self.compute_profits(price_A, price_B, alpha)
				
				payoff_matrix.loc[name_A, name_B] = (round(profit_A, 2), round(profit_B, 2))
				
				self.logger.debug(f"  [{current}/{total_computations}] ({name_A:>7}, {name_B:>7}) → "
				                  f"A: ${profit_A:>12,.0f}, B: ${profit_B:>12,.0f}")
		
		return payoff_matrix, strategies
	
	
	def find_nash_equilibria(self, payoff_matrix: pd.DataFrame) -> List[Tuple]:
		"""
		Find all pure strategy Nash Equilibria in the payoff matrix.
		
		Args:
			payoff_matrix: Payoff matrix DataFrame
		
		Returns:
			List of Nash Equilibria as (strategy_A, strategy_B, profit_A, profit_B)
		"""
		nash_equilibria = []
		strategy_names = payoff_matrix.index.tolist()
		
		for name_A in strategy_names:
			for name_B in strategy_names:
				profit_A, profit_B = payoff_matrix.loc[name_A, name_B]
				
				is_nash = True
				
				# Check if A wants to deviate
				for alt_A in strategy_names:
					alt_profit_A, _ = payoff_matrix.loc[alt_A, name_B]
					if alt_profit_A > profit_A:
						is_nash = False
						break
				
				# Check if B wants to deviate
				if is_nash:
					for alt_B in strategy_names:
						_, alt_profit_B = payoff_matrix.loc[name_A, alt_B]
						if alt_profit_B > profit_B:
							is_nash = False
							break
				
				if is_nash:
					nash_equilibria.append((name_A, name_B, profit_A, profit_B))
		
		return nash_equilibria
	
	
	def analyze_best_responses(self, payoff_matrix: pd.DataFrame) -> Dict:
		"""
		Find best response for each strategy.
		
		Args:
			payoff_matrix: Payoff matrix DataFrame
		
		Returns:
			Dictionary with best responses
		"""
		strategy_names = payoff_matrix.index.tolist()
		
		best_responses = {
			'ISP_A': {},
			'ISP_B': {}
		}
		
		# Best responses for ISP A (given B's strategy)
		for strat_B in strategy_names:
			profits = [(strat_A, payoff_matrix.loc[strat_A, strat_B][0]) 
			          for strat_A in strategy_names]
			best_A, profit_A = max(profits, key=lambda x: x[1])
			best_responses['ISP_A'][strat_B] = (best_A, profit_A)
		
		# Best responses for ISP B (given A's strategy)
		for strat_A in strategy_names:
			profits = [(strat_B, payoff_matrix.loc[strat_A, strat_B][1]) 
			          for strat_B in strategy_names]
			best_B, profit_B = max(profits, key=lambda x: x[1])
			best_responses['ISP_B'][strat_A] = (best_B, profit_B)
		
		return best_responses
	
	
	def visualize_payoff_matrix(self, payoff_matrix: pd.DataFrame, nash_equilibria: List[Tuple], 
	                            parameters: Dict, output_path: str = "figure/payoff_matrix.pdf"):
		"""
		Create visualization of the payoff matrix with both ISPs' profits in each cell.
		
		Args:
			payoff_matrix: Payoff matrix DataFrame
			nash_equilibria: List of Nash Equilibria
			parameters: Dictionary of model parameters
			output_path: Path to save the figure (PDF format)
		"""
		self.logger.info("Generating payoff matrix visualization...")
		
		# Extract profits for ISP A and ISP B
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
		
		# Create figure with single subplot - larger figure size for better readability
		fig, ax = plt.subplots(1, 1, figsize=(16, 14))
		
		# Use total welfare for background coloring
		im = ax.imshow(total_welfare, cmap='YlGn', aspect='auto', alpha=0.6)
		
		ax.set_xticks(range(n))
		ax.set_yticks(range(n))
		# Increased font sizes for better readability in papers
		ax.set_xticklabels(strategy_names, rotation=0, ha='center', fontsize=14, fontweight='bold')
		ax.set_yticklabels(strategy_names, fontsize=14, fontweight='bold')
		ax.set_xlabel('ISP B Strategy', fontsize=18, fontweight='bold', labelpad=15)
		ax.set_ylabel('ISP A Strategy', fontsize=18, fontweight='bold', labelpad=15)
		
		# Identify Nash Equilibria coordinates
		nash_coords = [(strategy_names.index(nA), strategy_names.index(nB)) 
		               for nA, nB, _, _ in nash_equilibria]
		
		# Add grid
		ax.set_xticks(np.arange(n) - 0.5, minor=True)
		ax.set_yticks(np.arange(n) - 0.5, minor=True)
		ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
		
		# Add values to cells - both ISPs side by side
		for i in range(n):
			for j in range(n):
				# Determine if this is a Nash Equilibrium
				is_nash = (i, j) in nash_coords
				
				# Cell background highlight for Nash Equilibrium
				if is_nash:
					rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
					                    fill=True, facecolor='gold', alpha=0.4,
					                    edgecolor='red', linewidth=4)
					ax.add_patch(rect)
				
				# Format profits
				pA = profit_A[i, j]
				pB = profit_B[i, j]
				
				# Create two-line text: ISP A on top, ISP B on bottom
				if is_nash:
					text_str = f"A: ${pA:.0f}\nB: ${pB:.0f}\n★ NE ★"
					color = 'darkred'
					weight = 'bold'
					size = 12  # Increased font size
				else:
					text_str = f"A: ${pA:.0f}\nB: ${pB:.0f}"
					color = 'black'
					weight = 'normal'
					size = 11  # Increased font size
				
				ax.text(j, i, text_str,
				       ha="center", va="center", 
				       color=color, fontsize=size, fontweight=weight,
				       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
				                edgecolor='none', alpha=0.8))
		
		# Add colorbar with larger font
		cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label('Total Social Welfare ($)', rotation=270, labelpad=25, fontsize=16, fontweight='bold')
		
		# Add title with parameters - larger font
		param_str = f"P0=${parameters['P0']}, γ={parameters['gamma']}, β={parameters['beta']}, α={parameters['alpha']}, s0={parameters['s0']}"
		title_text = f'Transit ISP Pricing Competition: Payoff Matrix\n{param_str}'
		
		if nash_equilibria:
			ne_list = ", ".join([f"({nA}, {nB})" for nA, nB, _, _ in nash_equilibria])
			title_text += f"\nNash Equilibrium: {ne_list}"
		
		ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25)
		
		# Add legend with larger font
		legend_elements = [
			plt.Rectangle((0, 0), 1, 1, fc='gold', alpha=0.4, edgecolor='red', linewidth=2, label='Nash Equilibrium'),
			plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=0, 
			          label='A = ISP A profit, B = ISP B profit')
		]
		ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=12)
		
		plt.tight_layout()
		
		# Save as PDF with high quality settings
		plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', 
		            facecolor='white', edgecolor='none')
		self.logger.info(f"✅ Payoff matrix visualization saved to '{output_path}'")
		plt.close()
	
	
	def run_analysis(self, gamma: float = 0.005, 
	                beta: float = 0.5, alpha: float = 2.0, s0: float = 0.2) -> Dict:
		"""
		Run complete competition analysis.
		
		Args:
			gamma: Variable cost per mile ($/mile/TB)
			beta: Fixed cost ($/TB)
			alpha: Price elasticity parameter
			s0: Competition intensity parameter
		
		Returns:
			Dictionary with all analysis results
		"""
		self.logger.info("Starting competition analysis...")
		
		# Build payoff matrix
		payoff_matrix, strategies = self.build_payoff_matrix(gamma, beta, alpha, s0)
		
		# Display payoff matrix
		self.logger.info("\n" + "="*80)
		self.logger.info("COMPLETE PAYOFF MATRIX (10x10)")
		self.logger.info("="*80)
		self.logger.info("Format: (ISP A profit, ISP B profit) in dollars")
		self.logger.info("Rows = ISP A's strategy, Columns = ISP B's strategy\n")
		self.logger.info(payoff_matrix.to_string())
		
		# Find Nash Equilibria
		self.logger.info("\n" + "="*80)
		self.logger.info("NASH EQUILIBRIUM ANALYSIS")
		self.logger.info("="*80)
		
		nash_equilibria = self.find_nash_equilibria(payoff_matrix)
		
		if nash_equilibria:
			self.logger.info(f"\nFound {len(nash_equilibria)} Nash Equilibrium/Equilibria:\n")
			for idx, (name_A, name_B, profit_A, profit_B) in enumerate(nash_equilibria, 1):
				self.logger.info(f"{idx}. ({name_A}, {name_B}):")
				self.logger.info(f"   ISP A profit: ${profit_A:>14,.2f}")
				self.logger.info(f"   ISP B profit: ${profit_B:>14,.2f}")
				self.logger.info(f"   Total welfare: ${profit_A + profit_B:>14,.2f}\n")
		else:
			self.logger.warning("\nNo pure strategy Nash Equilibrium found!")
		
		# Best response analysis
		self.logger.info("="*80)
		self.logger.info("BEST RESPONSE ANALYSIS")
		self.logger.info("="*80)
		
		best_responses = self.analyze_best_responses(payoff_matrix)
		
		self.logger.info("\nBest Responses for ISP A (given B's strategy):")
		for strat_B, (best_A, profit_A) in best_responses['ISP_A'].items():
			self.logger.info(f"  If B plays {strat_B:>8} → A's best: {best_A:>8} (profit: ${profit_A:>12,.2f})")
		
		self.logger.info("\nBest Responses for ISP B (given A's strategy):")
		for strat_A, (best_B, profit_B) in best_responses['ISP_B'].items():
			self.logger.info(f"  If A plays {strat_A:>8} → B's best: {best_B:>8} (profit: ${profit_B:>12,.2f})")
		
		# Comprehensive welfare analysis
		welfare_analysis = self.analyze_comprehensive_welfare(payoff_matrix, strategies, alpha)
		
		# Store parameters (including endogenous P0)
		parameters = {
			'P0': self.P0, 'gamma': gamma, 'beta': beta, 
			'alpha': alpha, 's0': s0
		}
		
		self.logger.info("Analysis completed successfully!")
		
		# Return results
		return {
			'payoff_matrix': payoff_matrix,
			'strategies': strategies,
			'nash_equilibria': nash_equilibria,
			'best_responses': best_responses,
			'welfare_analysis': welfare_analysis,
			'parameters': parameters
		}


def create_summary_json(results: Dict, base_path: str, run_id: str = None) -> Dict:
	"""
	Create comprehensive summary JSON with all important results.
	
	Args:
		results: Analysis results dictionary
		base_path: Base path for the run
		run_id: Optional run identifier
	
	Returns:
		Summary dictionary
	"""
	# Extract key information
	payoff_matrix = results['payoff_matrix']
	nash_equilibria = results['nash_equilibria']
	welfare_analysis = results['welfare_analysis']
	parameters = results['parameters']
	
	# Calculate Nash Equilibrium welfare statistics
	ne_welfare_stats = []
	ne_consumer_surplus = []
	ne_producer_surplus = []
	ne_total_welfare = []
	
	for name_A, name_B, profit_A, profit_B in nash_equilibria:
		# Find welfare for this NE
		ne_welfare = next((w for w in welfare_analysis['welfare_results'] 
		                  if w['strategy_A'] == name_A and w['strategy_B'] == name_B), None)
		
		if ne_welfare:
			ne_stats = {
				'strategy_A': name_A,
				'strategy_B': name_B,
				'profit_A': float(profit_A),
				'profit_B': float(profit_B),
				'consumer_surplus': float(ne_welfare['consumer_surplus']),
				'producer_surplus': float(ne_welfare['producer_surplus']),
				'total_welfare': float(ne_welfare['total_welfare']),
				'efficiency_ratio': float(ne_welfare['efficiency_ratio']),
				'deadweight_loss': float(ne_welfare['deadweight_loss'])
			}
			ne_welfare_stats.append(ne_stats)
			ne_consumer_surplus.append(ne_welfare['consumer_surplus'])
			ne_producer_surplus.append(ne_welfare['producer_surplus'])
			ne_total_welfare.append(ne_welfare['total_welfare'])
	
	# Calculate averages
	avg_consumer_surplus = np.mean(ne_consumer_surplus) if ne_consumer_surplus else 0
	avg_producer_surplus = np.mean(ne_producer_surplus) if ne_producer_surplus else 0
	avg_total_welfare = np.mean(ne_total_welfare) if ne_total_welfare else 0
	
	# Create summary
	summary = {
		'metadata': {
			'run_id': run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
			'timestamp': datetime.now().isoformat(),
			'base_path': base_path,
			'analysis_type': 'transit_isp_pricing_competition'
		},
		'parameters': {
			'P0': float(parameters['P0']),
			'gamma': float(parameters['gamma']),
			'beta': float(parameters['beta']),
			'alpha': float(parameters['alpha']),
			's0': float(parameters['s0'])
		},
		'nash_equilibria': {
			'count': len(nash_equilibria),
			'strategies': [(ne[0], ne[1]) for ne in nash_equilibria],
			'details': ne_welfare_stats,
			'averages': {
				'consumer_surplus': float(avg_consumer_surplus),
				'producer_surplus': float(avg_producer_surplus),
				'total_welfare': float(avg_total_welfare)
			}
		},
		'optimal_outcomes': {
			'max_total_welfare': {
				'strategy': (welfare_analysis['max_total_welfare']['strategy_A'], 
				           welfare_analysis['max_total_welfare']['strategy_B']),
				'consumer_surplus': float(welfare_analysis['max_total_welfare']['consumer_surplus']),
				'producer_surplus': float(welfare_analysis['max_total_welfare']['producer_surplus']),
				'total_welfare': float(welfare_analysis['max_total_welfare']['total_welfare']),
				'efficiency_ratio': float(welfare_analysis['max_total_welfare']['efficiency_ratio'])
			},
			'max_consumer_surplus': {
				'strategy': (welfare_analysis['max_consumer_surplus']['strategy_A'], 
				           welfare_analysis['max_consumer_surplus']['strategy_B']),
				'consumer_surplus': float(welfare_analysis['max_consumer_surplus']['consumer_surplus']),
				'producer_surplus': float(welfare_analysis['max_consumer_surplus']['producer_surplus']),
				'total_welfare': float(welfare_analysis['max_consumer_surplus']['total_welfare'])
			},
			'max_producer_surplus': {
				'strategy': (welfare_analysis['max_producer_surplus']['strategy_A'], 
				           welfare_analysis['max_producer_surplus']['strategy_B']),
				'consumer_surplus': float(welfare_analysis['max_producer_surplus']['consumer_surplus']),
				'producer_surplus': float(welfare_analysis['max_producer_surplus']['producer_surplus']),
				'total_welfare': float(welfare_analysis['max_producer_surplus']['total_welfare'])
			}
		},
		'welfare_comparison': {
			'ne_vs_optimal': {
				'welfare_gap': float(welfare_analysis['max_total_welfare']['total_welfare'] - avg_total_welfare),
				'efficiency_gap': float(welfare_analysis['max_total_welfare']['efficiency_ratio'] - 
				                      np.mean([ne['efficiency_ratio'] for ne in ne_welfare_stats]) if ne_welfare_stats else 0)
			}
		},
		'files_generated': {
			'payoff_matrix_csv': f"{base_path}/payoff_matrix.csv",
			'payoff_matrix_pdf': f"{base_path}/payoff_matrix.pdf",
			'summary_json': f"{base_path}/summary.json",
			'log_file': f"{base_path}/analysis.log"
		}
	}
	
	return summary


def setup_logger(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
	"""
	Set up logger with console and optional file output.
	
	Args:
		log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
		log_file: Optional log file path
	
	Returns:
		Configured logger instance
	"""
	# Create logger
	logger = logging.getLogger('transit_isp_analysis')
	logger.setLevel(getattr(logging, log_level.upper()))
	
	# Clear any existing handlers
	logger.handlers.clear()
	
	# Create formatter
	formatter = logging.Formatter(
		'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	
	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(getattr(logging, log_level.upper()))
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	
	# File handler (if specified)
	if log_file:
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(getattr(logging, log_level.upper()))
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	
	return logger


def parse_arguments():
	"""
	Parse command line arguments for model parameters.
	
	Returns:
		argparse.Namespace: Parsed arguments
	"""
	parser = argparse.ArgumentParser(
		description='Transit ISP Pricing Competition Analysis',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	
	# Model parameters
	parser.add_argument('--gamma', type=float, default=0.005,
	                   help='Variable cost per mile ($/mile/TB)')
	parser.add_argument('--beta', type=float, default=0.5,
	                   help='Fixed cost per TB ($/TB)')
	parser.add_argument('--alpha', type=float, default=2.0,
	                   help='Price elasticity parameter')
	parser.add_argument('--s0', type=float, default=0.2,
	                   help='Competition intensity parameter (P0 is calculated endogenously)')
	
	# File paths
	parser.add_argument('--data', type=str, default='netflow_grouped_by_src_dst.csv',
	                   help='Path to netflow CSV file')
	parser.add_argument('--config', type=str, default='tier_strategies.json',
	                   help='Path to tier strategy JSON configuration')
	
	# Base path for outputs
	parser.add_argument('--base-path', type=str, required=True,
	                   help='Base path for saving all outputs (figures, CSV, logs, summary.json)')
	
	# Analysis options
	parser.add_argument('--run-id', type=str, default=None,
	                   help='Optional run identifier for summary.json')
	
	# Logging options
	parser.add_argument('--log-level', type=str, default='INFO',
	                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
	                   help='Logging level')
	
	return parser.parse_args()


# Main execution
if __name__ == "__main__":
	# Parse command line arguments
	args = parse_arguments()
	
	# Create base path directory
	base_path = Path(args.base_path)
	base_path.mkdir(parents=True, exist_ok=True)
	
	# Set up file paths
	log_file = base_path / "analysis.log"
	payoff_matrix_csv = base_path / "payoff_matrix.csv"
	payoff_matrix_pdf = base_path / "payoff_matrix.pdf"
	summary_json = base_path / "summary.json"
	
	# Set up logger
	logger = setup_logger(args.log_level, str(log_file))
	
	logger.info("="*80)
	logger.info("TRANSIT ISP PRICING COMPETITION ANALYSIS")
	logger.info("="*80)
	logger.info(f"Base path: {base_path}")
	logger.info(f"Parameters:")
	logger.info(f"  γ (distance cost): ${args.gamma}/mile/TB")
	logger.info(f"  β (fixed cost): ${args.beta}/TB")
	logger.info(f"  α (price elasticity): {args.alpha}")
	logger.info(f"  s0 (competition intensity): {args.s0}")
	logger.info(f"  P0 (baseline price): Calculated endogenously from cost + markup")
	logger.info(f"  Data file: {args.data}")
	logger.info(f"  Config file: {args.config}")
	logger.info(f"  Log level: {args.log_level}")
	logger.info("="*80)
	
	# Initialize model
	model = TransitISPCompetitionModel(
		data_path=args.data,
		tier_config_path=args.config,
		logger=logger
	)
	
	# Run analysis with command line parameters
	results = model.run_analysis(
		gamma=args.gamma,
		beta=args.beta,
		alpha=args.alpha,
		s0=args.s0
	)
	
	# Save payoff matrix to CSV
	results['payoff_matrix'].to_csv(payoff_matrix_csv)
	logger.info(f"✅ Payoff matrix saved to '{payoff_matrix_csv}'")
	
	# Generate visualization
	logger.info("\n" + "="*80)
	logger.info("GENERATING VISUALIZATION")
	logger.info("="*80)
	model.visualize_payoff_matrix(
		results['payoff_matrix'],
		results['nash_equilibria'],
		results['parameters'],
		output_path=str(payoff_matrix_pdf)
	)
	
	# Create and save summary JSON
	logger.info("\n" + "="*80)
	logger.info("CREATING SUMMARY")
	logger.info("="*80)
	summary = create_summary_json(results, str(base_path), args.run_id)
	
	with open(summary_json, 'w') as f:
		json.dump(summary, f, indent=2)
	
	logger.info(f"✅ Summary saved to '{summary_json}'")
	
	logger.info("\n" + "="*80)
	logger.info("ANALYSIS COMPLETE")
	logger.info("="*80)
	logger.info(f"All outputs saved to: {base_path}")
	logger.info(f"  - Payoff matrix CSV: {payoff_matrix_csv}")
	logger.info(f"  - Payoff matrix PDF: {payoff_matrix_pdf}")
	logger.info(f"  - Summary JSON: {summary_json}")
	logger.info(f"  - Analysis log: {log_file}")