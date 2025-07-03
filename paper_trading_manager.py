#!/usr/bin/env python3
"""
FILE: paper_trading_manager.py
PURPOSE: Paper Trading Manager - Orchestrates paper trading operations

DESCRIPTION:
- Manages paper trading workflow
- Coordinates signal generation and execution
- Provides performance monitoring
- Generates trading reports

USAGE:
- python paper_trading_manager.py --mode run
- python paper_trading_manager.py --mode status
- python paper_trading_manager.py --mode report
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

class PaperTradingManager:
    """Orchestrate paper trading operations"""
    
    def __init__(self):
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            from agents.signal_agent import SignalAgent
            from agents.portfolio_agent import PortfolioAgent
            import config
            
            self.db_manager = EnhancedDatabaseManager()
            self.signal_agent = SignalAgent(self.db_manager)
            self.portfolio_agent = PortfolioAgent(self.db_manager)
            self.config = config
            
        except ImportError as e:
            print(f"Import error: {e}")
            sys.exit(1)
    
    def run_paper_trading_session(self) -> Dict:
        """Run complete paper trading session"""
        
        print("Paper Trading Session")
        print("=" * 40)
        
        try:
            # Update existing positions
            print("1. Updating positions...")
            update_result = self.portfolio_agent.update_paper_positions()
            print(f"   Updated {update_result.get('positions_updated', 0)} positions")
            
            # Generate new signals and execute
            print("2. Generating signals...")
            trade_result = self.signal_agent.generate_trade_orders()
            print(f"   Generated {trade_result.get('signals_generated', 0)} signals")
            print(f"   Executed {trade_result.get('execution_result', {}).get('executed', 0)} trades")
            
            # Get portfolio status
            print("3. Portfolio summary...")
            portfolio_summary = self.portfolio_agent.get_paper_portfolio_summary()
            
            if 'error' not in portfolio_summary:
                print(f"   Portfolio Value: ₹{portfolio_summary.get('portfolio_value', 0):,.0f}")
                print(f"   Return: {portfolio_summary.get('total_return_percent', 0):.1f}%")
                print(f"   Open Positions: {portfolio_summary.get('open_positions', 0)}")
                print(f"   Win Rate: {portfolio_summary.get('win_rate', 0):.1f}%")
            
            return {
                'status': 'completed',
                'positions_updated': update_result.get('positions_updated', 0),
                'signals_generated': trade_result.get('signals_generated', 0),
                'trades_executed': trade_result.get('execution_result', {}).get('executed', 0),
                'portfolio_summary': portfolio_summary
            }
            
        except Exception as e:
            print(f"Paper trading session failed: {e}")
            return {'error': str(e)}
    
    def get_paper_trading_status(self) -> Dict:
        """Get current paper trading status"""
        
        try:
            portfolio_summary = self.portfolio_agent.get_paper_portfolio_summary()
            signal_performance = self.signal_agent.get_signal_performance_summary()
            recent_signals = self.signal_agent.get_executable_signals()
            
            return {
                'paper_trading_enabled': self.config.PAPER_TRADING_MODE,
                'portfolio_summary': portfolio_summary,
                'signal_performance': signal_performance,
                'pending_signals': len(recent_signals),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_daily_report(self) -> Dict:
        """Generate daily paper trading report"""
        
        try:
            print("Daily Paper Trading Report")
            print("=" * 50)
            print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            portfolio_summary = self.portfolio_agent.get_paper_portfolio_summary()
            
            if 'error' not in portfolio_summary:
                print(f"\nPortfolio Performance:")
                print(f"  Portfolio Value: ₹{portfolio_summary.get('portfolio_value', 0):,.0f}")
                print(f"  Available Cash: ₹{portfolio_summary.get('available_cash', 0):,.0f}")
                print(f"  Total Return: {portfolio_summary.get('total_return_percent', 0):+.2f}%")
                print(f"  Realized P&L: ₹{portfolio_summary.get('realized_pnl', 0):+,.0f}")
                print(f"  Unrealized P&L: ₹{portfolio_summary.get('unrealized_pnl', 0):+,.0f}")
            
            signal_performance = self.signal_agent.get_signal_performance_summary()
            
            if 'error' not in signal_performance:
                print(f"\nTrading Activity:")
                print(f"  Total Trades: {signal_performance.get('total_signals', 0)}")
                print(f"  Win Rate: {signal_performance.get('success_rate', 0):.1f}%")
                print(f"  Avg Confidence: {signal_performance.get('avg_confidence', 0):.1f}")
                print(f"  High Conf Success: {signal_performance.get('high_confidence_success_rate', 0):.1f}%")
            
            open_positions = self.portfolio_agent._get_open_paper_positions()
            
            if open_positions:
                print(f"\nOpen Positions ({len(open_positions)}):")
                for pos in open_positions[:5]:
                    symbol = pos.get('symbol', 'N/A')
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_pct = pos.get('unrealized_pnl_percent', 0)
                    print(f"  {symbol}: ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)")
            
            return {
                'status': 'completed',
                'portfolio_summary': portfolio_summary,
                'signal_performance': signal_performance,
                'open_positions_count': len(open_positions)
            }
            
        except Exception as e:
            print(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    def validate_paper_trading_setup(self) -> bool:
        """Validate paper trading setup"""
        
        print("Paper Trading Setup Validation")
        print("=" * 40)
        
        checks = {
            'Database Connection': False,
            'Paper Trading Config': False,
            'Signal Generation': False,
            'Portfolio Tracking': False
        }
        
        try:
            if self.db_manager.test_connection():
                checks['Database Connection'] = True
            
            if hasattr(self.config, 'PAPER_TRADING_MODE') and self.config.PAPER_TRADING_MODE:
                checks['Paper Trading Config'] = True
            
            test_signal = self.signal_agent.generate_signal('RELIANCE')
            if test_signal and 'error' not in test_signal:
                checks['Signal Generation'] = True
            
            portfolio_summary = self.portfolio_agent.get_paper_portfolio_summary()
            if 'error' not in portfolio_summary:
                checks['Portfolio Tracking'] = True
            
        except Exception as e:
            print(f"Validation error: {e}")
        
        for check, status in checks.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            print(f"  {check:20}: {status_str}")
        
        all_passed = all(checks.values())
        print(f"\nValidation: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Paper Trading Manager')
    parser.add_argument('--mode', choices=['run', 'status', 'report', 'validate'], 
                       default='run', help='Operation mode')
    
    args = parser.parse_args()
    
    try:
        manager = PaperTradingManager()
        
        if args.mode == 'run':
            result = manager.run_paper_trading_session()
        elif args.mode == 'status':
            result = manager.get_paper_trading_status()
            print(f"Paper Trading Status: {result}")
        elif args.mode == 'report':
            result = manager.generate_daily_report()
        elif args.mode == 'validate':
            success = manager.validate_paper_trading_setup()
            sys.exit(0 if success else 1)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()