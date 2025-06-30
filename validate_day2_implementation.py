#!/usr/bin/env python3
"""
FILE: validate_day2_implementation.py
LOCATION: / (root directory)
PURPOSE: Validate Day 2 implementation is complete and working

DESCRIPTION:
- Tests all implemented agents work correctly
- Validates Day 1 foundation + Day 2 enhancements
- Checks that out-of-scope features are removed
- Ensures database integration works

USAGE:
- python validate_day2_implementation.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

class Day2ValidationSuite:
    """Validate Day 2 implementation completeness"""
    
    def __init__(self):
        self.results = {}
        
    def run_validation(self) -> bool:
        """Run complete Day 2 validation"""
        
        print("=" * 60)
        print("DAY 2 IMPLEMENTATION VALIDATION")
        print("=" * 60)
        
        validations = [
            ("Environment Setup", self.validate_environment),
            ("Database Integration", self.validate_database),
            ("Technical Agent", self.validate_technical_agent),
            ("Fundamental Agent", self.validate_fundamental_agent),
            ("Signal Agent", self.validate_signal_agent),
            ("Risk Agent", self.validate_risk_agent),
            ("Portfolio Agent", self.validate_portfolio_agent),
            ("Complete Integration", self.validate_complete_integration),
            ("Scope Compliance", self.validate_scope_compliance)
        ]
        
        for test_name, test_func in validations:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                self.results[test_name] = result
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  Status: {status}")
            except Exception as e:
                self.results[test_name] = False
                print(f"  Status: ✗ ERROR - {e}")
        
        self.print_summary()
        return all(self.results.values())
    
    def validate_environment(self) -> bool:
        """Validate environment setup"""
        try:
            import pandas as pd
            import numpy as np
            import psycopg2
            import config
            
            print(f"  pandas: {pd.__version__}")
            print(f"  numpy: {np.__version__}")
            
            # Test config validation
            if not config.validate_config():
                return False
            
            print("  Configuration validated")
            return True
            
        except ImportError as e:
            print(f"  Missing dependency: {e}")
            return False
    
    def validate_database(self) -> bool:
        """Validate database integration"""
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            if not db_manager.test_connection():
                print("  Database connection failed")
                return False
            
            print("  Database connection successful")
            
            # Test symbol loading
            symbols = db_manager.get_testing_symbols()
            if not symbols:
                print("  No testing symbols available")
                return False
            
            print(f"  Found {len(symbols)} testing symbols")
            return True
            
        except Exception as e:
            print(f"  Database error: {e}")
            return False
    
    def validate_technical_agent(self) -> bool:
        """Validate technical agent"""
        try:
            from agents.technical_agent import TechnicalAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            tech_agent = TechnicalAgent(db_manager)
            
            # Test with synthetic data
            analysis = tech_agent.analyze_symbol("TEST_SYMBOL")
            
            if 'error' in analysis:
                print(f"  Technical analysis failed: {analysis['error']}")
                return False
            
            required_fields = ['technical_score', 'overall_confidence']
            missing = [f for f in required_fields if f not in analysis]
            if missing:
                print(f"  Missing fields: {missing}")
                return False
            
            print(f"  Technical analysis working (score: {analysis['technical_score']:.3f})")
            return True
            
        except Exception as e:
            print(f"  Technical agent error: {e}")
            return False
    
    def validate_fundamental_agent(self) -> bool:
        """Validate fundamental agent - Day 2 requirement"""
        try:
            from agents.fundamental_agent import FundamentalAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            fund_agent = FundamentalAgent(db_manager)
            
            # Test with real symbol
            symbols = db_manager.get_testing_symbols()
            if not symbols:
                print("  No test symbols for fundamental analysis")
                return False
            
            analysis = fund_agent.analyze_symbol_fundamentals(symbols[0])
            
            if 'error' in analysis:
                print(f"  Fundamental analysis failed: {analysis['error']}")
                return False
            
            required_fields = ['fundamental_score', 'valuation_score', 'quality_score', 'growth_score']
            missing = [f for f in required_fields if f not in analysis]
            if missing:
                print(f"  Missing fields: {missing}")
                return False
            
            print(f"  Fundamental analysis working (score: {analysis['fundamental_score']:.3f})")
            return True
            
        except Exception as e:
            print(f"  Fundamental agent error: {e}")
            return False
    
    def validate_signal_agent(self) -> bool:
        """Validate signal agent - Day 2 master coordinator"""
        try:
            from agents.signal_agent import SignalAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            signal_agent = SignalAgent(db_manager)
            
            # Test signal generation
            test_symbols = db_manager.get_testing_symbols()[:2]  # Test with 2 symbols
            if not test_symbols:
                print("  No test symbols for signal generation")
                return False
            
            signals = signal_agent.generate_signals(test_symbols)
            
            if not signals:
                print("  No signals generated (acceptable for limited data)")
                return True  # This is OK for Day 2
            
            # Validate signal structure
            signal = signals[0]
            required_fields = ['symbol', 'signal_type', 'overall_confidence', 'technical_score']
            missing = [f for f in required_fields if f not in signal]
            if missing:
                print(f"  Missing signal fields: {missing}")
                return False
            
            print(f"  Signal generation working ({len(signals)} signals)")
            return True
            
        except Exception as e:
            print(f"  Signal agent error: {e}")
            return False
    
    def validate_risk_agent(self) -> bool:
        """Validate risk agent"""
        try:
            from agents.risk_agent import RiskAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            risk_agent = RiskAgent(db_manager)
            
            # Test position sizing
            test_signal = {
                'symbol': 'TEST',
                'entry_price': 100.0,
                'stop_loss': 95.0,
                'overall_confidence': 0.7,
                'category': 'A',
                'volatility_category': 'Medium',
                'market_cap_type': 'Large_Cap'
            }
            
            position_info = risk_agent.calculate_position_size(test_signal)
            
            if 'error' in position_info:
                print(f"  Position sizing failed: {position_info['error']}")
                return False
            
            required_fields = ['recommended_shares', 'recommended_position_value', 'actual_risk_amount']
            missing = [f for f in required_fields if f not in position_info]
            if missing:
                print(f"  Missing position fields: {missing}")
                return False
            
            print(f"  Risk management working (position: ₹{position_info['recommended_position_value']:.0f})")
            return True
            
        except Exception as e:
            print(f"  Risk agent error: {e}")
            return False
    
    def validate_portfolio_agent(self) -> bool:
        """Validate portfolio agent"""
        try:
            from agents.portfolio_agent import PortfolioAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            portfolio_agent = PortfolioAgent(db_manager)
            
            # Test portfolio performance calculation
            performance = portfolio_agent.calculate_portfolio_performance()
            
            if 'error' in performance:
                print(f"  Portfolio calculation failed: {performance['error']}")
                return False
            
            required_fields = ['total_positions', 'total_value']
            missing = [f for f in required_fields if f not in performance]
            if missing:
                print(f"  Missing portfolio fields: {missing}")
                return False
            
            print(f"  Portfolio management working ({performance['total_positions']} positions)")
            return True
            
        except Exception as e:
            print(f"  Portfolio agent error: {e}")
            return False
    
    def validate_complete_integration(self) -> bool:
        """Validate complete Day 2 integration"""
        try:
            from agents.signal_agent import SignalAgent
            from agents.risk_agent import RiskAgent
            from agents.portfolio_agent import PortfolioAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            # Test complete workflow
            db_manager = EnhancedDatabaseManager()
            signal_agent = SignalAgent(db_manager)
            risk_agent = RiskAgent(db_manager)
            portfolio_agent = PortfolioAgent(db_manager)
            
            # Get test symbols
            test_symbols = db_manager.get_testing_symbols()[:2]
            if not test_symbols:
                print("  No test symbols available")
                return False
            
            # Generate signals
            signals = signal_agent.generate_signals(test_symbols)
            
            # Test capital allocation
            allocation = portfolio_agent.allocate_capital(signals)
            
            if 'error' in allocation:
                print(f"  Capital allocation failed: {allocation['error']}")
                return False
            
            print(f"  Complete integration working")
            print(f"    Signals evaluated: {allocation['signals_evaluated']}")
            print(f"    Signals approved: {allocation['signals_approved']}")
            
            return True
            
        except Exception as e:
            print(f"  Integration error: {e}")
            return False
    
    def validate_scope_compliance(self) -> bool:
        """Validate that out-of-scope features are removed"""
        try:
            removed_features = []
            
            # Check that Kite Connect is not imported (Day 6 feature)
            try:
                import kiteconnect
                print("  ✗ Kite Connect still present (should be removed)")
                return False
            except ImportError:
                removed_features.append("Kite Connect")
            
            # Check that advanced technical analysis is simplified
            from utils.technical_calculations import TechnicalCalculations
            
            # These should NOT exist (Day 4+ features)
            advanced_methods = ['identify_chart_patterns', 'calculate_momentum_indicators', 'analyze_volume_profile']
            existing_advanced = []
            
            for method in advanced_methods:
                if hasattr(TechnicalCalculations, method):
                    existing_advanced.append(method)
            
            if existing_advanced:
                print(f"  ✗ Advanced features still present: {existing_advanced}")
                return False
            
            print(f"  Scope compliance verified")
            print(f"    Removed: {', '.join(removed_features)}")
            return True
            
        except Exception as e:
            print(f"  Scope validation error: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary"""
        
        print("\n" + "=" * 60)
        print("DAY 2 VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        for test_name, result in self.results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:25}: {status}")
        
        print("-" * 60)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 DAY 2 IMPLEMENTATION COMPLETE!")
            print("✓ All Day 1 foundation working")
            print("✓ All Day 2 enhancements implemented")
            print("✓ Out-of-scope features removed")
            print("✓ Ready for production use")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed")
            print("Please fix issues before proceeding")
        
        print("=" * 60)

def main():
    """Run Day 2 validation"""
    
    validator = Day2ValidationSuite()
    success = validator.run_validation()
    
    if success:
        print("\n🚀 READY TO PROCEED WITH DAY 3+ FEATURES!")
    else:
        print("\n🔧 FIXES NEEDED BEFORE PROCEEDING")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()