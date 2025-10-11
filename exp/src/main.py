"""Main orchestration for Transit ISP Pricing Competition Analysis."""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict

from .models.pricing import PricingModel
from .models.competition import CompetitionModel
from .analysis.welfare import WelfareAnalyzer
from .analysis.visualization import PayoffMatrixVisualizer
from .utils.logger import setup_logger
from .utils.io_handler import load_json, create_summary_json, parse_payoff_matrix_csv, calculate_welfare_matrix, calculate_strategy_details, save_json


class TransitISPAnalysis:
    
    def __init__(self, data_path: str, tier_config_path: str, logger):
        self.logger = logger
        
        self.df = pd.read_csv(data_path)
        self.df = self.df[self.df['distance'] > 0].reset_index(drop=True)
        self.df['demand_tb'] = self.df['demand'] / 1e6
        
        self.tier_config = load_json(tier_config_path)
        
        self.logger.info(f"Loaded {len(self.df)} flows, Total: {self.df['demand_tb'].sum():.2f} TB")
        self.logger.info(f"Distance: {self.df['distance'].min():.0f}-{self.df['distance'].max():.0f} mi")
        
        self.pricing_model = PricingModel(self.df, self.tier_config, logger)
        self.competition_model = CompetitionModel(self.df, logger)
        self.welfare_analyzer = WelfareAnalyzer(self.df, logger)
        self.visualizer = PayoffMatrixVisualizer(logger)
    
    def build_all_strategies(self, gamma: float, beta: float, alpha: float, s0: float, max_tiers: int = 10) -> Dict:
        """Build pricing strategies from 1-tier to max_tiers."""
        self.logger.info("="*80)
        self.logger.info("BUILDING PRICING STRATEGIES")
        self.logger.info("="*80)
        
        self.pricing_model.calculate_costs(gamma, beta)
        self.pricing_model.calculate_valuations(alpha, s0)
        
        self.logger.info(f"Parameters: γ={gamma}, β={beta}, α={alpha}, s0={s0}")
        self.logger.info(f"Endogenous P0=${self.pricing_model.P0:.2f}/TB\n")
        
        strategies = {}
        for n_tiers in range(1, max_tiers + 1):
            strategy_name = "Flat" if n_tiers == 1 else f"{n_tiers}-Tier"
            df_strategy, prices = self.pricing_model.create_pricing_strategy(n_tiers, gamma, beta, alpha, s0)
            strategies[strategy_name] = {'prices': prices, 'data': df_strategy, 'n_tiers': n_tiers}
            self.pricing_model.log_strategy_summary(strategy_name, df_strategy, n_tiers)
        
        return strategies
    
    def run_single_matrix_analysis(self, gamma: float, beta: float, alpha: float, s0: float, max_tiers: int) -> Dict:
        """Run analysis for a single NxN matrix where N=max_tiers."""
        strategies = self.build_all_strategies(gamma, beta, alpha, s0, max_tiers)
        payoff_matrix = self.competition_model.build_payoff_matrix(strategies, alpha)
        self.competition_model.log_payoff_matrix(payoff_matrix)
        
        nash_equilibria = self.competition_model.find_nash_equilibria(payoff_matrix)
        self.competition_model.log_nash_equilibria(nash_equilibria)
        
        best_responses = self.competition_model.analyze_best_responses(payoff_matrix)
        self.competition_model.log_best_responses(best_responses)
        
        welfare_analysis = self.welfare_analyzer.analyze_comprehensive_welfare(payoff_matrix, strategies, alpha)
        self.welfare_analyzer.analyze_nash_welfare(
            nash_equilibria, welfare_analysis['welfare_results'], welfare_analysis['max_total_welfare']
        )
        
        parameters = {
            'P0': self.pricing_model.P0,
            'gamma': gamma, 'beta': beta, 'alpha': alpha, 's0': s0,
            'max_tiers': max_tiers
        }
        
        return {
            'payoff_matrix': payoff_matrix,
            'strategies': strategies,
            'nash_equilibria': nash_equilibria,
            'best_responses': best_responses,
            'welfare_analysis': welfare_analysis,
            'welfare_analyzer': self.welfare_analyzer,
            'parameters': parameters
        }
    
    def run_complete_analysis(self, gamma: float = 0.005, beta: float = 0.5, 
                             alpha: float = 2.0, s0: float = 0.2, max_tiers: int = 10) -> Dict:
        """Run complete competition analysis for specified max_tiers."""
        self.logger.info(f"Starting {max_tiers}x{max_tiers} competition analysis...")
        results = self.run_single_matrix_analysis(gamma, beta, alpha, s0, max_tiers)
        self.logger.info("Analysis completed!")
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transit ISP Pricing Competition Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--gamma', type=float, default=0.005, help='Variable cost per mile ($/mile/TB)')
    parser.add_argument('--beta', type=float, default=0.5, help='Fixed cost per TB ($/TB)')
    parser.add_argument('--alpha', type=float, default=2.0, help='Price elasticity parameter')
    parser.add_argument('--s0', type=float, default=0.2, help='Competition intensity parameter')
    parser.add_argument('--max-tiers', type=int, default=None, help='Max tiers for NxN matrix (1-10, or omit for all scenarios 1x1 to 10x10)')
    parser.add_argument('--data', type=str, default='../../dataset/netflow.csv', help='Path to netflow CSV')
    parser.add_argument('--config', type=str, default='tier_strategies.json', help='Path to tier config JSON')
    parser.add_argument('--base-path', type=str, required=True, help='Base path for outputs')
    parser.add_argument('--run-id', type=str, default=None, help='Run identifier')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(level=getattr(__import__('logging'), args.log_level))
    
    logger.info("="*80)
    logger.info("TRANSIT ISP PRICING COMPETITION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Base path: {base_path}")
    logger.info(f"Parameters: γ={args.gamma}, β={args.beta}, α={args.alpha}, s0={args.s0}")
    logger.info(f"P0: Calculated endogenously")
    
    analysis = TransitISPAnalysis(args.data, args.config, logger)
    
    # Determine which scenarios to run
    if args.max_tiers is not None:
        # Single scenario: NxN where N=max_tiers
        scenarios = [args.max_tiers]
        logger.info(f"Running single scenario: {args.max_tiers}x{args.max_tiers}")
    else:
        # All scenarios: 1x1, 2x2, ..., 10x10
        scenarios = list(range(1, 11))
        logger.info("Running all scenarios: 1x1, 2x2, ..., 10x10")
    
    logger.info("="*80)
    
    # Run analysis for each scenario
    for max_tiers in scenarios:
        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO: {max_tiers}x{max_tiers} MATRIX")
        logger.info(f"{'='*80}\n")
        
        results = analysis.run_complete_analysis(args.gamma, args.beta, args.alpha, args.s0, max_tiers)
        
        # Create scenario-specific output directory
        if len(scenarios) > 1:
            scenario_path = base_path / f"{max_tiers}x{max_tiers}"
            scenario_path.mkdir(exist_ok=True)
        else:
            scenario_path = base_path
        
        payoff_csv = scenario_path / "payoff_matrix.csv"
        payoff_pdf = scenario_path / "payoff_matrix.pdf"
        summary_json = scenario_path / "summary.json"
        
        results['payoff_matrix'].to_csv(payoff_csv)
        logger.info(f"Payoff matrix saved to {payoff_csv}")
        
        logger.info("Generating visualization...")
        analysis.visualizer.visualize_payoff_matrix(
            results['payoff_matrix'], results['nash_equilibria'], results['parameters'], str(payoff_pdf)
        )
        
        logger.info("Creating summary...")
        create_summary_json(
            results['parameters'], results['nash_equilibria'], 
            results['welfare_analysis'], results['strategies'], scenario_path
        )
        logger.info(f"Summary saved to {summary_json}")
        
        # Generate additional detailed analysis files
        logger.info("Generating detailed analysis files...")
        
        # Parse payoff matrix for welfare calculations
        strategies_list, payoff_matrix_dict = parse_payoff_matrix_csv(str(payoff_csv))
        
        # Calculate welfare matrix
        welfare_matrix = calculate_welfare_matrix(
            strategies_list, payoff_matrix_dict, results, results['parameters']['alpha']
        )
        
        # Calculate strategy details
        strategy_details = calculate_strategy_details(results)
        
        # Save welfare matrix
        welfare_json = scenario_path / "welfare_matrix.json"
        save_json({
            'parameters': results['parameters'],
            'welfare_matrix': welfare_matrix
        }, str(welfare_json))
        logger.info(f"Welfare matrix saved to {welfare_json}")
        
        # Save strategy details
        strategy_json = scenario_path / "strategy_details.json"
        save_json({
            'parameters': results['parameters'],
            'strategy_details': strategy_details
        }, str(strategy_json))
        logger.info(f"Strategy details saved to {strategy_json}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL SCENARIOS COMPLETE")
    logger.info(f"All outputs in: {base_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
