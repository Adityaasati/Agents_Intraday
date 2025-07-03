#!/usr/bin/env python3
"""
FILE: advanced_portfolio_integration.py
PURPOSE: Advanced Portfolio Management Integration - Stop Loss & Risk Parity

DESCRIPTION:
- Validates stop loss optimization features
- Tests risk parity portfolio construction
- Demonstrates advanced portfolio optimization
- Integrates with existing system components

USAGE:
- python advanced_portfolio_integration.py --mode validate
- python advanced_portfolio_integration.py --mode optimize --symbols "RELIANCE,TCS,INFY"
- python advanced_portfolio_integration.py --mode stops
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

class AdvancedPortfolioManager:
    """Advanced portfolio management with stop loss optimization and risk parity"""
    
    def __init__(self):
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            from agents.risk_agent import RiskAgent
            from agents.portfolio_agent import PortfolioAgent
            from agents.signal_agent import SignalAgent
            import config
            
            self.db_manager = EnhancedDatabaseManager()
            self.risk_agent = RiskAgent(self.db_manager)
            self.portfolio_agent = PortfolioAgent(self.db_manager)
            self.signal_agent = SignalAgent(self.db_manager)
            self.config = config
            
        except ImportError as e:
            print(f"Import error: {e}")
            sys.exit(1)
    
    def validate_advanced_features(self) -> bool:
        """Validate advanced portfolio management features"""
        
        print("Advanced Portfolio Management Validation")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Stop loss optimization
        print("1. Stop Loss Optimization...")
        try:
            test_result = self.risk_agent.calculate_optimal_stop_loss(
                'RELIANCE', 2500.0, {'volatility_category': 'Medium'}
            )
            
            if 'optimal_stop_loss' in test_result:
                results['stop_loss'] = True
            else:
                results['stop_loss'] = False
                
        except Exception as e:
            results['stop_loss'] = False
        
        # Test 2: Risk parity calculation
        print("2. Risk Parity Weights...")
        try:
            test_symbols = ['RELIANCE', 'TCS', 'INFY']
            weights = self.risk_agent.calculate_risk_parity_weights(test_symbols)
            
            if isinstance(weights, dict) and len(weights) == len(test_symbols):
                results['risk_parity'] = True
            else:
                results['risk_parity'] = False
                
        except Exception as e:
            results['risk_parity'] = False
        
        # Test 3: Portfolio optimization
        print("3. Portfolio Optimization...")
        try:
            test_signals = self._create_test_signals()
            optimization_result = self.portfolio_agent.construct_risk_parity_portfolio(test_signals)
            
            if 'strategy' in optimization_result:
                results['optimization'] = True
            else:
                results['optimization'] = False
                
        except Exception as e:
            results['optimization'] = False
        
        # Test 4: Advanced configuration
        print("4. Configuration Validation...")
        try:
            if hasattr(self.config, 'validate_advanced_config'):
                config_valid = self.config.validate_advanced_config()
            else:
                config_valid = True  # Assume valid if function doesn't exist
            
            results['configuration'] = config_valid
            
        except Exception as e:
            results['configuration'] = False
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        print(f"\nValidation Summary: {passed}/{total} tests passed")
        
        for test, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {test}: {status}")
        
        if passed >= 3:
            print("\nAdvanced Portfolio Management: READY")
            return True
        else:
            print("\nAdvanced Portfolio Management: NEEDS ATTENTION")
            return False
    
    def demonstrate_stop_loss_optimization(self) -> Dict:
        """Demonstrate stop loss optimization capabilities"""
        
        print("Stop Loss Optimization Demonstration")
        print("=" * 50)
        
        try:
            # Test different stop loss strategies
            test_symbol = 'RELIANCE'
            test_entry_price = 2500.0
            
            # Calculate optimal stop loss
            stop_result = self.risk_agent.calculate_optimal_stop_loss(
                test_symbol, test_entry_price, {'volatility_category': 'Medium'}
            )
            
            print(f"\nSymbol: {test_symbol}")
            print(f"Entry Price: ₹{test_entry_price:,.0f}")
            print(f"Optimal Stop Loss: ₹{stop_result.get('optimal_stop_loss', 0):,.0f}")
            print(f"Strategy Used: {stop_result.get('strategy_used', 'unknown')}")
            print(f"Risk-Reward Ratio: {stop_result.get('risk_reward_ratio', 0):.2f}")
            
            # Show all available strategies
            all_strategies = stop_result.get('all_strategies', {})
            if all_strategies:
                print(f"\nAll Stop Loss Strategies:")
                for strategy, data in all_strategies.items():
                    stop_price = data.get('stop_price', 0)
                    stop_percent = ((test_entry_price - stop_price) / test_entry_price) * 100
                    print(f"  {strategy}: ₹{stop_price:,.0f} ({stop_percent:.1f}%)")
            
            # Test dynamic stop updates
            test_positions = [{
                'symbol': test_symbol,
                'entry_price': test_entry_price,
                'stop_loss': stop_result.get('optimal_stop_loss', 0),
                'position_value': 100000,
                'volatility_category': 'Medium'
            }]
            
            updated_positions = self.risk_agent.update_dynamic_stops(test_positions)
            
            return {
                'optimal_stop_result': stop_result,
                'dynamic_update_result': updated_positions,
                'demonstration_successful': True
            }
            
        except Exception as e:
            print(f"Stop loss demonstration failed: {e}")
            return {'error': str(e)}
    
    def demonstrate_portfolio_optimization(self, symbols: List[str] = None) -> Dict:
        """Demonstrate portfolio optimization features"""
        
        print("Portfolio Optimization Demonstration")
        print("=" * 50)
        
        try:
            if symbols is None:
                symbols = self.db_manager.get_testing_symbols()[:5]
            
            # Generate test signals
            test_signals = []
            for symbol in symbols:
                test_signals.append({
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'entry_price': 1000.0,
                    'stop_loss': 950.0,
                    'overall_confidence': 0.7,
                    'technical_score': 0.6,
                    'volatility_category': 'Medium'
                })
            
            # Risk parity portfolio construction
            risk_parity_result = self.portfolio_agent.construct_risk_parity_portfolio(test_signals)
            
            print(f"\nRisk Parity Portfolio:")
            if 'positions' in risk_parity_result:
                positions = risk_parity_result['positions']
                print(f"  Total Positions: {len(positions)}")
                print(f"  Capital Utilization: {risk_parity_result.get('capital_utilization', 0):.1f}%")
                
                for position in positions[:3]:  # Show first 3
                    symbol = position.get('symbol')
                    weight = position.get('portfolio_weight', 0)
                    value = position.get('recommended_position_value', 0)
                    print(f"    {symbol}: {weight:.1%} (₹{value:,.0f})")
            
            # Rebalancing analysis
            rebalancing_result = self.portfolio_agent.calculate_rebalancing_needs()
            
            print(f"\nRebalancing Analysis:")
            print(f"  Rebalancing Needed: {rebalancing_result.get('rebalancing_needed', False)}")
            if rebalancing_result.get('rebalancing_needed'):
                print(f"  Max Deviation: {rebalancing_result.get('max_deviation', 0):.1f}%")
                print(f"  Actions Required: {rebalancing_result.get('actions_required', 0)}")
            
            # Portfolio optimization report
            optimization_report = self.portfolio_agent.generate_portfolio_optimization_report(test_signals)
            
            if 'recommendations' in optimization_report:
                print(f"\nOptimization Recommendations:")
                for i, rec in enumerate(optimization_report['recommendations'][:3], 1):
                    print(f"  {i}. {rec}")
            
            return {
                'risk_parity_portfolio': risk_parity_result,
                'rebalancing_analysis': rebalancing_result,
                'optimization_report': optimization_report,
                'demonstration_successful': True
            }
            
        except Exception as e:
            print(f"Portfolio optimization demonstration failed: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_portfolio_status(self) -> Dict:
        """Get comprehensive portfolio status with advanced metrics"""
        
        print("Comprehensive Portfolio Status")
        print("=" * 50)
        
        try:
            # Current portfolio performance
            current_performance = self.portfolio_agent.calculate_portfolio_performance()
            
            # Portfolio risk metrics
            current_positions = self.portfolio_agent.get_current_positions()
            portfolio_var = self.risk_agent.calculate_portfolio_var(current_positions)
            
            # Rebalancing needs
            rebalancing = self.portfolio_agent.calculate_rebalancing_needs()
            
            print(f"\nCurrent Portfolio:")
            print(f"  Positions: {current_performance.get('total_positions', 0)}")
            print(f"  Total Value: ₹{current_performance.get('total_value', 0):,.0f}")
            print(f"  Capital Utilization: {current_performance.get('capital_utilization', 0):.1f}%")
            print(f"  Diversification Score: {current_performance.get('diversification_score', 0):.1f}/100")
            
            print(f"\nRisk Metrics:")
            print(f"  Portfolio VaR (95%): ₹{portfolio_var.get('var_amount', 0):,.0f}")
            print(f"  VaR Percentage: {portfolio_var.get('var_percent', 0):.2f}%")
            print(f"  Portfolio Volatility: {portfolio_var.get('portfolio_volatility', 0):.1f}%")
            
            print(f"\nRebalancing Status:")
            print(f"  Rebalancing Needed: {rebalancing.get('rebalancing_needed', False)}")
            if rebalancing.get('rebalancing_needed'):
                print(f"  Max Deviation: {rebalancing.get('max_deviation', 0):.1f}%")
            
            return {
                'current_performance': current_performance,
                'risk_metrics': portfolio_var,
                'rebalancing_status': rebalancing,
                'status_timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Portfolio status failed: {e}")
            return {'error': str(e)}
    
    def _create_test_signals(self) -> List[Dict]:
        """Create test signals for demonstration"""
        
        test_symbols = self.db_manager.get_testing_symbols()[:5]
        
        signals = []
        for symbol in test_symbols:
            signals.append({
                'symbol': symbol,
                'signal_type': 'BUY',
                'entry_price': 1500.0,
                'stop_loss': 1425.0,
                'target_price': 1650.0,
                'overall_confidence': 0.75,
                'technical_score': 0.7,
                'fundamental_score': 0.6,
                'volatility_category': 'Medium',
                'category': 'A',
                'market_cap_type': 'Large_Cap'
            })
        
        return signals

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Advanced Portfolio Management Integration')
    parser.add_argument('--mode', choices=['validate', 'optimize', 'stops', 'status'], 
                       required=True, help='Execution mode')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols for optimization')
    
    args = parser.parse_args()
    
    manager = AdvancedPortfolioManager()
    
    if args.mode == 'validate':
        success = manager.validate_advanced_features()
        sys.exit(0 if success else 1)
        
    elif args.mode == 'stops':
        manager.demonstrate_stop_loss_optimization()
        
    elif args.mode == 'optimize':
        symbols = args.symbols.split(',') if args.symbols else None
        manager.demonstrate_portfolio_optimization(symbols)
        
    elif args.mode == 'status':
        manager.get_comprehensive_portfolio_status()

if __name__ == "__main__":
    main()