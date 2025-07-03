#!/usr/bin/env python3
"""
FILE: live_trading_manager.py
PURPOSE: Live Trading Manager with Trade Mode Control

DESCRIPTION:
- Manages live trading workflow with TRADE_MODE control
- TRADE_MODE=no: Generate signals only, no orders
- TRADE_MODE=yes: Place actual orders via Kite API
- Automatic token management integration

USAGE:
- python live_trading_manager.py --mode run
- python live_trading_manager.py --mode status
- python live_trading_manager.py --mode signals
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

class LiveTradingManager:
    """Orchestrate live trading operations with trade mode control"""
    
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
    
    def run_live_trading_session(self) -> Dict:
        """Run trading session with trade mode awareness"""
        
        print("Live Trading Session")
        print("=" * 40)
        
        try:
            # Check trade mode
            if not self.config.TRADE_MODE:
                print("TRADE_MODE=no: Generating signals only (no orders)")
                return self._run_signal_generation_session()
            
            print("TRADE_MODE=yes: Live trading enabled")
            
            if not self.config.LIVE_TRADING_MODE:
                print("LIVE_TRADING_MODE=false: Running paper trading")
                from paper_trading_manager import PaperTradingManager
                paper_manager = PaperTradingManager()
                return paper_manager.run_paper_trading_session()
            
            if not self.portfolio_agent._is_market_open():
                print("Market closed - session skipped")
                return {'status': 'market_closed'}
            
            # Check Kite token
            kite = self.portfolio_agent._get_kite_connection()
            if not kite:
                print("Kite API not available - check token generation")
                print("Run: python kite_token_generator.py")
                return {'status': 'kite_unavailable', 'action': 'generate_token'}
            
            # Update live positions
            print("1. Updating live positions...")
            update_result = self.portfolio_agent.update_live_positions()
            print(f"   Updated {update_result.get('positions_updated', 0)} positions")
            
            # Generate and execute live signals
            print("2. Generating live signals...")
            trade_result = self.signal_agent.generate_live_trade_orders()
            
            execution_result = trade_result.get('execution_result', {})
            print(f"   Generated {trade_result.get('signals_generated', 0)} signals")
            print(f"   Live executed {execution_result.get('live_executed', 0)} trades")
            
            # Get live portfolio status
            print("3. Live portfolio summary...")
            portfolio_summary = self.portfolio_agent.get_live_portfolio_summary()
            
            if 'error' not in portfolio_summary:
                print(f"   Portfolio Value: ₹{portfolio_summary.get('portfolio_value', 0):,.0f}")
                print(f"   Return: {portfolio_summary.get('total_return_percent', 0):.1f}%")
                print(f"   Live Positions: {portfolio_summary.get('open_positions', 0)}")
            
            return {
                'status': 'completed',
                'execution_type': 'LIVE',
                'trade_mode': 'enabled',
                'positions_updated': update_result.get('positions_updated', 0),
                'signals_generated': trade_result.get('signals_generated', 0),
                'live_trades_executed': execution_result.get('live_executed', 0),
                'portfolio_summary': portfolio_summary
            }
            
        except Exception as e:
            print(f"Live trading session failed: {e}")
            return {'error': str(e)}
    
    def _run_signal_generation_session(self) -> Dict:
        """Run signal-only session (TRADE_MODE=no)"""
        
        try:
            # Generate signals without executing
            print("1. Generating trading signals...")
            trade_result = self.signal_agent.generate_live_trade_orders()
            
            execution_result = trade_result.get('execution_result', {})
            signals_generated = execution_result.get('signals_generated', 0)
            
            print(f"   Generated {trade_result.get('signals_generated', 0)} total signals")
            print(f"   High-confidence signals: {trade_result.get('live_executable_signals', 0)}")
            print(f"   Would execute: {signals_generated} (if TRADE_MODE=yes)")
            
            # Show signal analysis
            print("2. Signal analysis...")
            signals = trade_result.get('signals', [])
            
            if signals:
                for signal in signals[:3]:  # Show top 3
                    symbol = signal.get('symbol', 'N/A')
                    confidence = signal.get('overall_confidence', 0)
                    entry_price = signal.get('entry_price', 0)
                    print(f"   {symbol}: {confidence:.1%} confidence @ ₹{entry_price:.2f}")
            else:
                print("   No high-confidence signals generated")
            
            return {
                'status': 'completed',
                'execution_type': 'SIGNAL_ONLY',
                'trade_mode': 'disabled',
                'signals_generated': trade_result.get('signals_generated', 0),
                'executable_signals': trade_result.get('live_executable_signals', 0)
            }
            
        except Exception as e:
            print(f"Signal generation session failed: {e}")
            return {'error': str(e)}
    
    def show_signals_only(self) -> Dict:
        """Show current signals without any execution"""
        
        try:
            print("Current Trading Signals")
            print("=" * 40)
            
            # Get all current signals
            signals = self.signal_agent.get_live_executable_signals()
            
            if not signals:
                print("No high-confidence signals currently available")
                return {'status': 'no_signals'}
            
            print(f"Found {len(signals)} high-confidence signals:")
            print("")
            
            for i, signal in enumerate(signals, 1):
                symbol = signal.get('symbol', 'N/A')
                confidence = signal.get('overall_confidence', 0)
                entry_price = signal.get('entry_price', 0)
                stop_loss = signal.get('stop_loss', 0)
                target = signal.get('target_price', 0)
                
                print(f"{i}. {symbol}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Entry: ₹{entry_price:.2f}")
                print(f"   Stop Loss: ₹{stop_loss:.2f}")
                print(f"   Target: ₹{target:.2f}")
                
                if self.config.TRADE_MODE:
                    print(f"   Status: Will execute if market open and conditions met")
                else:
                    print(f"   Status: Signal only (TRADE_MODE=no)")
                print("")
            
            return {
                'status': 'completed',
                'signals_count': len(signals),
                'signals': signals
            }
            
        except Exception as e:
            print(f"Signal display failed: {e}")
            return {'error': str(e)}
    
    def validate_live_trading_setup(self) -> bool:
        """Validate trading setup with trade mode awareness"""
        
        print("Live Trading Setup Validation")
        print("=" * 40)
        
        checks = {
            'Database Connection': False,
            'Trade Mode Config': False,
            'Kite API Setup': False,
            'Market Access': False,
            'Token Management': False
        }
        
        try:
            if self.db_manager.test_connection():
                checks['Database Connection'] = True
            
            if hasattr(self.config, 'TRADE_MODE') and hasattr(self.config, 'LIVE_TRADING_MODE'):
                checks['Trade Mode Config'] = True
                print(f"  TRADE_MODE: {'yes' if self.config.TRADE_MODE else 'no'}")
                print(f"  LIVE_TRADING_MODE: {self.config.LIVE_TRADING_MODE}")
            
            # Check Kite API only if trading is enabled
            if self.config.TRADE_MODE and self.config.LIVE_TRADING_MODE:
                kite = self.portfolio_agent._get_kite_connection()
                if kite:
                    checks['Kite API Setup'] = True
                    checks['Token Management'] = True
                else:
                    print("  Kite API: Run 'python kite_token_generator.py' to generate token")
            else:
                checks['Kite API Setup'] = True  # Not required in signal-only mode
                checks['Token Management'] = True
            
            if self.portfolio_agent._is_market_open():
                checks['Market Access'] = True
            else:
                print("  Note: Market currently closed")
                checks['Market Access'] = True  # Not a failure
            
        except Exception as e:
            print(f"Validation error: {e}")
        
        for check, status in checks.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            print(f"  {check:20}: {status_str}")
        
        critical_checks = ['Database Connection', 'Trade Mode Config']
        if self.config.TRADE_MODE and self.config.LIVE_TRADING_MODE:
            critical_checks.extend(['Kite API Setup', 'Token Management'])
        
        critical_passed = all(checks[check] for check in critical_checks)
        
        print(f"\nValidation: {'PASSED' if critical_passed else 'FAILED'}")
        
        return critical_passed

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Live Trading Manager')
    parser.add_argument('--mode', choices=['run', 'status', 'report', 'validate', 'signals'], 
                       default='run', help='Operation mode')
    
    args = parser.parse_args()
    
    try:
        manager = LiveTradingManager()
        
        if args.mode == 'run':
            result = manager.run_live_trading_session()
        elif args.mode == 'status':
            result = manager.get_live_trading_status()
            print(f"Trading Status: {result}")
        elif args.mode == 'signals':
            result = manager.show_signals_only()
        elif args.mode == 'validate':
            success = manager.validate_live_trading_setup()
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