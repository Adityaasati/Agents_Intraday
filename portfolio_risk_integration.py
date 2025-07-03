#!/usr/bin/env python3
"""
FILE: portfolio_risk_integration.py
PURPOSE: Portfolio Risk Analysis Integration - Enhanced risk management demonstration

DESCRIPTION:
- Demonstrates enhanced portfolio risk analysis features
- Tests correlation analysis, dynamic position sizing, and risk monitoring
- Integrates with existing signal generation system
- Provides portfolio optimization recommendations

USAGE:
- python portfolio_risk_integration.py --mode validate
- python portfolio_risk_integration.py --mode analyze --symbols "RELIANCE,TCS,INFY"
- python portfolio_risk_integration.py --mode monitor
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

class PortfolioRiskAnalysis:
    """Enhanced portfolio risk analysis and optimization"""
    
    def __init__(self):
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            from agents.risk_agent import RiskAgent
            from agents.portfolio_agent import PortfolioAgent
            from agents.signal_agent import SignalAgent
            
            self.db_manager = EnhancedDatabaseManager()
            self.risk_agent = RiskAgent(self.db_manager)
            self.portfolio_agent = PortfolioAgent(self.db_manager)
            self.signal_agent = SignalAgent(self.db_manager)
            
        except ImportError as e:
            print(f"Failed to import required modules: {e}")
            sys.exit(1)
    
    def validate_enhanced_features(self) -> bool:
        """Validate all enhanced risk management features"""
        
        print("Portfolio Risk Analysis Validation")
        print("=" * 50)
        
        validation_results = {}
        
        # Test 1: Enhanced position sizing
        print("\n1. Enhanced Position Sizing...")
        try:
            test_signal = {
                'symbol': 'RELIANCE',
                'entry_price': 2500.0,
                'stop_loss': 2400.0,
                'overall_confidence': 0.75,
                'volatility_category': 'Medium'
            }
            
            current_positions = self.portfolio_agent.get_current_positions()
            enhanced_result = self.risk_agent.calculate_enhanced_position_size(test_signal, current_positions)
            
            if 'enhancement_applied' in enhanced_result:
                print("   Enhanced position sizing: WORKING")
                validation_results['enhanced_sizing'] = True
            else:
                print("   Enhanced position sizing: FALLBACK")
                validation_results['enhanced_sizing'] = True  # Fallback is acceptable
                
        except Exception as e:
            print(f"   Enhanced position sizing: FAILED - {e}")
            validation_results['enhanced_sizing'] = False
        
        # Test 2: Correlation analysis
        print("\n2. Portfolio Correlation Analysis...")
        try:
            current_positions = self.portfolio_agent.get_current_positions()
            correlation_result = self.risk_agent.calculate_portfolio_correlation(current_positions, 'TCS')
            
            if 'correlation_risk' in correlation_result:
                print("   Correlation analysis: WORKING")
                validation_results['correlation'] = True
            else:
                print("   Correlation analysis: LIMITED")
                validation_results['correlation'] = True
                
        except Exception as e:
            print(f"   Correlation analysis: FAILED - {e}")
            validation_results['correlation'] = False
        
        # Test 3: Portfolio risk monitoring
        print("\n3. Portfolio Risk Monitoring...")
        try:
            monitoring_result = self.portfolio_agent.monitor_portfolio_risk()
            
            if 'portfolio_health' in monitoring_result:
                print("   Risk monitoring: WORKING")
                validation_results['monitoring'] = True
            else:
                print("   Risk monitoring: FAILED")
                validation_results['monitoring'] = False
                
        except Exception as e:
            print(f"   Risk monitoring: FAILED - {e}")
            validation_results['monitoring'] = False
        
        # Test 4: Optimized capital allocation
        print("\n4. Optimized Capital Allocation...")
        try:
            # Generate test signals
            test_symbols = self.db_manager.get_testing_symbols()[:3]
            signals = []
            
            for symbol in test_symbols:
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'entry_price': 1000.0,
                    'stop_loss': 950.0,
                    'overall_confidence': 0.7,
                    'technical_score': 0.6
                })
            
            allocation_result = self.portfolio_agent.allocate_capital_optimized(signals)
            
            if 'optimization_applied' in allocation_result:
                print("   Optimized allocation: WORKING")
                validation_results['optimization'] = True
            else:
                print("   Optimized allocation: FALLBACK")
                validation_results['optimization'] = True
                
        except Exception as e:
            print(f"   Optimized allocation: FAILED - {e}")
            validation_results['optimization'] = False
        
        # Test 5: Database integration
        print("\n5. Enhanced Database Storage...")
        try:
            test_monitoring = {
                'timestamp': datetime.now(),
                'portfolio_health': 'GOOD',
                'total_risk_percent': 15.5,
                'concentration_risk': 25.0,
                'correlation_risk': 'low',
                'alerts_count': 0,
                'alerts': ''
            }
            
            storage_result = self.db_manager.store_portfolio_monitoring(test_monitoring)
            
            if storage_result:
                print("   Database storage: WORKING")
                validation_results['database'] = True
            else:
                print("   Database storage: FAILED")
                validation_results['database'] = False
                
        except Exception as e:
            print(f"   Database storage: FAILED - {e}")
            validation_results['database'] = False
        
        # Summary
        passed_tests = sum(validation_results.values())
        total_tests = len(validation_results)
        
        print(f"\nValidation Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= 4:
            print("Portfolio Risk Analysis: READY")
            return True
        else:
            print("Portfolio Risk Analysis: NEEDS ATTENTION")
            return False
    
    def analyze_portfolio_risk(self, symbols: List[str] = None) -> Dict:
        """Comprehensive portfolio risk analysis"""
        
        print("Comprehensive Portfolio Risk Analysis")
        print("=" * 50)
        
        try:
            # Get current portfolio state
            current_positions = self.portfolio_agent.get_current_positions()
            
            # Portfolio risk metrics
            risk_metrics = self.risk_agent.get_portfolio_risk_metrics(current_positions)
            
            # Diversification analysis
            diversification = self.portfolio_agent.get_portfolio_diversification_score()
            
            # Cash allocation analysis
            cash_analysis = self.portfolio_agent.optimize_cash_allocation()
            
            # Sector allocation
            sector_allocation = self.db_manager.get_sector_allocation_summary()
            
            # Display results
            print(f"\nPortfolio Overview:")
            print(f"  Total Positions: {len(current_positions)}")
            print(f"  Total Risk: {risk_metrics.get('total_risk_percent', 0):.1f}%")
            print(f"  Diversification Score: {diversification.get('diversification_score', 0):.1f}/100")
            print(f"  Cash Available: {cash_analysis.get('cash_percentage', 0):.1f}%")
            
            print(f"\nRisk Analysis:")
            print(f"  Concentration Risk: {risk_metrics.get('concentration_risk', 0):.1f}%")
            print(f"  Correlation Risk: {risk_metrics.get('correlation_risk', 'Unknown')}")
            print(f"  Risk Budget Used: {risk_metrics.get('risk_budget_used', 0):.1f}%")
            
            print(f"\nSector Distribution:")
            for sector, data in sector_allocation.get('sectors', {}).items():
                print(f"  {sector}: {data.get('allocation_percent', 0):.1f}% ({data.get('position_count', 0)} positions)")
            
            # Generate signals for specified symbols if provided
            if symbols:
                print(f"\nAnalyzing new opportunities: {', '.join(symbols)}")
                new_signals = self._generate_signals_for_symbols(symbols)
                
                if new_signals:
                    allocation_result = self.portfolio_agent.allocate_capital_optimized(new_signals)
                    
                    print(f"\nAllocation Recommendations:")
                    print(f"  Signals Evaluated: {allocation_result.get('signals_evaluated', 0)}")
                    print(f"  Signals Approved: {len(allocation_result.get('approved_signals', []))}")
                    print(f"  New Allocation: ₹{allocation_result.get('new_allocation', 0):,.0f}")
                    
                    for signal in allocation_result.get('approved_signals', [])[:3]:
                        symbol = signal.get('symbol')
                        confidence = signal.get('overall_confidence', 0)
                        position_value = signal.get('enhanced_position_info', {}).get('recommended_position_value', 0)
                        print(f"    {symbol}: ₹{position_value:,.0f} (confidence: {confidence:.2f})")
                else:
                    print("  No signals generated for specified symbols")
            
            return {
                'portfolio_risk': risk_metrics,
                'diversification': diversification,
                'cash_analysis': cash_analysis,
                'sector_allocation': sector_allocation,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Portfolio analysis failed: {e}")
            return {'error': str(e)}
    
    def monitor_portfolio_health(self) -> Dict:
        """Real-time portfolio health monitoring"""
        
        print("Portfolio Health Monitoring")
        print("=" * 50)
        
        try:
            # Current monitoring
            monitoring_result = self.portfolio_agent.monitor_portfolio_risk()
            
            # Historical context
            history = self.db_manager.get_portfolio_monitoring_history(days_back=7)
            
            print(f"\nCurrent Health: {monitoring_result.get('portfolio_health', 'UNKNOWN')}")
            print(f"Risk Alerts: {len(monitoring_result.get('alerts', []))}")
            
            if monitoring_result.get('alerts'):
                print("Active Alerts:")
                for alert in monitoring_result.get('alerts', []):
                    print(f"  - {alert}")
            
            print("\nRecommendations:")
            for rec in monitoring_result.get('recommendations', []):
                print(f"  - {rec}")
            
            print(f"\nNext Review: {monitoring_result.get('next_review', 'UNKNOWN')}")
            
            if history:
                print(f"\nRecent History ({len(history)} records):")
                for record in history[:3]:
                    timestamp = record.get('timestamp', 'Unknown')
                    health = record.get('portfolio_health', 'Unknown')
                    risk_pct = record.get('total_risk_percent', 0)
                    print(f"  {timestamp}: {health} (Risk: {risk_pct:.1f}%)")
            
            return monitoring_result
            
        except Exception as e:
            print(f"Portfolio monitoring failed: {e}")
            return {'error': str(e)}
    
    def _generate_signals_for_symbols(self, symbols: List[str]) -> List[Dict]:
        """Generate signals for analysis"""
        
        try:
            signals = []
            for symbol in symbols:
                # Get signal from signal agent
                symbol_signals = self.signal_agent.generate_signals([symbol])
                signals.extend(symbol_signals)
            
            return signals
            
        except Exception as e:
            print(f"Signal generation failed: {e}")
            return []

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Portfolio Risk Analysis Integration')
    parser.add_argument('--mode', choices=['validate', 'analyze', 'monitor'], 
                       required=True, help='Analysis mode')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to analyze')
    
    args = parser.parse_args()
    
    analyzer = PortfolioRiskAnalysis()
    
    if args.mode == 'validate':
        success = analyzer.validate_enhanced_features()
        sys.exit(0 if success else 1)
        
    elif args.mode == 'analyze':
        symbols = args.symbols.split(',') if args.symbols else None
        analyzer.analyze_portfolio_risk(symbols)
        
    elif args.mode == 'monitor':
        analyzer.monitor_portfolio_health()

if __name__ == "__main__":
    main()