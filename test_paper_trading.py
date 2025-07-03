#!/usr/bin/env python3
"""
FILE: test_paper_trading.py
PURPOSE: Test Paper Trading Implementation

DESCRIPTION:
- Tests paper trading functionality
- Validates signal-to-trade conversion
- Tests portfolio tracking
- Generates test report

USAGE:
- python test_paper_trading.py --quick
- python test_paper_trading.py --full
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

class PaperTradingValidator:
    """Validate Paper Trading Implementation"""
    
    def __init__(self):
        self.results = {}
        
    def run_tests(self, quick_mode=True):
        """Run paper trading validation tests"""
        
        print("Paper Trading Validation")
        print("=" * 50)
        
        tests = [
            ("Configuration", self.test_configuration),
            ("Database Schema", self.test_database_schema),
            ("Signal Generation", self.test_signal_generation),
            ("Paper Trade Execution", self.test_paper_execution),
            ("Portfolio Tracking", self.test_portfolio_tracking),
            ("Performance Reporting", self.test_reporting)
        ]
        
        if not quick_mode:
            tests.append(("Full Integration", self.test_full_integration))
        
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                self.results[test_name] = result
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {status}")
            except Exception as e:
                self.results[test_name] = False
                print(f"  ✗ ERROR: {e}")
        
        self.print_summary()
        return all(self.results.values())
    
    def test_configuration(self) -> bool:
        """Test paper trading configuration"""
        
        try:
            import config
            
            required_configs = [
                'PAPER_TRADING_MODE',
                'PAPER_TRADING_INITIAL_CAPITAL',
                'AUTO_EXECUTE_SIGNALS',
                'PAPER_MAX_POSITIONS'
            ]
            
            for conf in required_configs:
                if not hasattr(config, conf):
                    print(f"    Missing config: {conf}")
                    return False
            
            print(f"    Paper trading: {'Enabled' if config.PAPER_TRADING_MODE else 'Disabled'}")
            print(f"    Initial capital: ₹{config.PAPER_TRADING_INITIAL_CAPITAL:,.0f}")
            
            return True
            
        except Exception as e:
            print(f"    Config test failed: {e}")
            return False
    
    def test_database_schema(self) -> bool:
        """Test database schema for paper trading"""
        
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'agent_paper_performance'
                        )
                    """)
                    table_exists = cursor.fetchone()[0]
                    
                    if table_exists:
                        print("    Paper performance table: EXISTS")
                    else:
                        print("    Paper performance table: MISSING")
                        return False
                    
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_name = 'agent_portfolio_positions' 
                            AND column_name = 'executed_at'
                        )
                    """)
                    column_exists = cursor.fetchone()[0]
                    
                    if column_exists:
                        print("    Extended portfolio table: EXISTS")
                    else:
                        print("    Extended portfolio table: MISSING")
                        return False
            
            return True
            
        except Exception as e:
            print(f"    Database test failed: {e}")
            return False
    
    def test_signal_generation(self) -> bool:
        """Test signal generation with paper trading"""
        
        try:
            from agents.signal_agent import SignalAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            signal_agent = SignalAgent(db_manager)
            
            signal = signal_agent.generate_signal('RELIANCE')
            
            if signal and 'error' not in signal:
                print(f"    Signal generated for RELIANCE: {signal.get('overall_confidence', 0):.2f} confidence")
                
                executable = signal_agent.get_executable_signals()
                print(f"    Executable signals found: {len(executable)}")
                
                return True
            else:
                print("    Signal generation failed")
                return False
                
        except Exception as e:
            print(f"    Signal test failed: {e}")
            return False
    
    def test_paper_execution(self) -> bool:
        """Test paper trade execution"""
        
        try:
            from agents.portfolio_agent import PortfolioAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            portfolio_agent = PortfolioAgent(db_manager)
            
            test_signal = {
                'symbol': 'TEST',
                'entry_price': 100.0,
                'recommended_shares': 50,
                'recommended_position_value': 5000,
                'stop_loss': 95.0,
                'target_price': 110.0
            }
            
            result = portfolio_agent.execute_paper_trade(test_signal)
            
            if 'error' not in result:
                print(f"    Paper trade executed: {result.get('status')}")
                print(f"    Execution price: ₹{result.get('execution_price', 0):.2f}")
                return True
            else:
                print(f"    Paper execution failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"    Paper execution test failed: {e}")
            return False
    
    def test_portfolio_tracking(self) -> bool:
        """Test portfolio tracking functionality"""
        
        try:
            from agents.portfolio_agent import PortfolioAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            portfolio_agent = PortfolioAgent(db_manager)
            
            summary = portfolio_agent.get_paper_portfolio_summary()
            
            if 'error' not in summary:
                print(f"    Portfolio value: ₹{summary.get('portfolio_value', 0):,.0f}")
                print(f"    Open positions: {summary.get('open_positions', 0)}")
                print(f"    Win rate: {summary.get('win_rate', 0):.1f}%")
                return True
            else:
                print(f"    Portfolio tracking failed: {summary.get('error')}")
                return False
                
        except Exception as e:
            print(f"    Portfolio tracking test failed: {e}")
            return False
    
    def test_reporting(self) -> bool:
        """Test reporting functionality"""
        
        try:
            from paper_trading_manager import PaperTradingManager
            
            manager = PaperTradingManager()
            
            status = manager.get_paper_trading_status()
            
            if 'error' not in status:
                print(f"    Paper trading enabled: {status.get('paper_trading_enabled')}")
                print(f"    Pending signals: {status.get('pending_signals', 0)}")
                return True
            else:
                print(f"    Reporting failed: {status.get('error')}")
                return False
                
        except Exception as e:
            print(f"    Reporting test failed: {e}")
            return False
    
    def test_full_integration(self) -> bool:
        """Test full paper trading integration"""
        
        try:
            from paper_trading_manager import PaperTradingManager
            
            manager = PaperTradingManager()
            
            result = manager.run_paper_trading_session()
            
            if 'error' not in result:
                print(f"    Session completed: {result.get('status')}")
                print(f"    Trades executed: {result.get('trades_executed', 0)}")
                return True
            else:
                print(f"    Integration test failed: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"    Integration test failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        
        print("\n" + "=" * 50)
        print("PAPER TRADING VALIDATION SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        for test_name, result in self.results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:25}: {status}")
        
        print("-" * 50)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 PAPER TRADING READY!")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} tests failed")

def main():
    """Main validation function"""
    
    parser = argparse.ArgumentParser(description='Test Paper Trading')
    parser.add_argument('--quick', action='store_true', help='Quick validation')
    parser.add_argument('--full', action='store_true', help='Full validation with integration test')
    
    args = parser.parse_args()
    
    validator = PaperTradingValidator()
    
    quick_mode = args.quick or not args.full
    success = validator.run_tests(quick_mode)
    
    if success:
        print("\n🚀 Ready to use paper trading!")
        print("\nUsage:")
        print("  python main.py --mode paper")
        print("  python paper_trading_manager.py --mode run")
    else:
        print("\n🔧 Please fix issues before using paper trading")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()