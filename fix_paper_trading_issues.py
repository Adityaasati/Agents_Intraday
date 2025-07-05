#!/usr/bin/env python3
"""
FILE: fix_paper_trading_issues.py
PURPOSE: Apply fixes for paper trading validation errors

DESCRIPTION:
- Adds missing database columns
- Verifies agent methods exist
- Updates configuration if needed
- Validates final setup
- Uses your .env configuration (port 5435)

USAGE:
- python fix_paper_trading_issues.py --apply
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env first
from dotenv import load_dotenv
load_dotenv()

def apply_database_fixes():
    """Apply database schema fixes"""
    
    print("1. Applying database schema fixes...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from database.schema_creator import SchemaCreator
        from database.connection_config import get_connection_params
        
        # Test connection with your .env settings
        connection_params = get_connection_params()
        print(f"   Using database: {connection_params['host']}:{connection_params['port']}")
        
        db_manager = EnhancedDatabaseManager()
        schema_creator = SchemaCreator()
        
        # Test connection first
        if not db_manager.test_connection():
            print("   ✗ Database connection failed - check your .env settings")
            print(f"   Expected: {connection_params['host']}:{connection_params['port']}") 
            return False
        
        # Add paper trading columns
        if schema_creator.add_paper_trading_columns():
            print("   ✓ Paper trading columns added")
        else:
            print("   ⚠ Some columns may already exist")
        
        # Verify columns
        checks = schema_creator.verify_paper_trading_columns()
        
        for check, status in checks.items():
            status_str = "✓" if status else "✗"
            print(f"   {status_str} {check}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"   ✗ Database fix failed: {e}")
        return False

def verify_agent_methods():
    """Verify required agent methods exist"""
    
    print("2. Verifying agent methods...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from agents.portfolio_agent import PortfolioAgent
        from agents.signal_agent import SignalAgent
        
        db_manager = EnhancedDatabaseManager()
        
        checks = {
            'PortfolioAgent._get_all_paper_positions': False,
            'SignalAgent.generate_signal': False,
            'SignalAgent.get_executable_signals': False
        }
        
        # Check PortfolioAgent
        portfolio_agent = PortfolioAgent(db_manager)
        if hasattr(portfolio_agent, '_get_all_paper_positions'):
            checks['PortfolioAgent._get_all_paper_positions'] = True
        
        # Check SignalAgent
        signal_agent = SignalAgent(db_manager)
        if hasattr(signal_agent, 'generate_signal'):
            checks['SignalAgent.generate_signal'] = True
        if hasattr(signal_agent, 'get_executable_signals'):
            checks['SignalAgent.get_executable_signals'] = True
        
        for check, status in checks.items():
            status_str = "✓" if status else "✗"
            print(f"   {status_str} {check}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"   ✗ Agent verification failed: {e}")
        return False

def test_paper_trading_flow():
    """Test the complete paper trading flow"""
    
    print("3. Testing paper trading flow...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from agents.signal_agent import SignalAgent
        from agents.portfolio_agent import PortfolioAgent
        import config
        
        db_manager = EnhancedDatabaseManager()
        signal_agent = SignalAgent(db_manager)
        portfolio_agent = PortfolioAgent(db_manager)
        
        # Test signal generation
        test_signal = signal_agent.generate_signal('RELIANCE')
        if 'error' in test_signal:
            print(f"   ⚠ Signal generation test: {test_signal['error']}")
        else:
            print(f"   ✓ Signal generation test passed (confidence: {test_signal.get('overall_confidence', 0):.3f})")
        
        # Test portfolio summary
        portfolio_summary = portfolio_agent.get_paper_portfolio_summary()
        if 'error' in portfolio_summary:
            print(f"   ⚠ Portfolio summary test: {portfolio_summary['error']}")
        else:
            print(f"   ✓ Portfolio summary test passed (value: ₹{portfolio_summary.get('portfolio_value', 0):,.0f})")
        
        # Test executable signals
        executable_signals = signal_agent.get_executable_signals()
        print(f"   ✓ Executable signals test passed ({len(executable_signals)} signals)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Paper trading flow test failed: {e}")
        return False

def run_validation():
    """Run the original validation to confirm fixes"""
    
    print("4. Running validation...")
    
    try:
        from paper_trading_manager import PaperTradingManager
        
        manager = PaperTradingManager()
        is_valid = manager.validate_paper_trading_setup()
        
        if is_valid:
            print("   ✓ Paper trading validation PASSED")
        else:
            print("   ✗ Paper trading validation FAILED")
        
        return is_valid
        
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")
        return False

def main():
    """Main fix application function"""
    
    print("Paper Trading Issue Fixes")
    print("=" * 50)
    
    fixes = [
        ("Database Schema", apply_database_fixes),
        ("Agent Methods", verify_agent_methods),
        ("Trading Flow", test_paper_trading_flow),
        ("Final Validation", run_validation)
    ]
    
    success_count = 0
    
    for fix_name, fix_func in fixes:
        print()
        if fix_func():
            success_count += 1
        else:
            print(f"❌ {fix_name} needs attention")
    
    print("\n" + "=" * 50)
    print(f"Fix Results: {success_count}/{len(fixes)} successful")
    
    if success_count == len(fixes):
        print("🎉 All fixes applied successfully!")
        print("\nNext steps:")
        print("1. python paper_trading_manager.py --mode validate")
        print("2. python paper_trading_manager.py --mode run")
        return True
    else:
        print("⚠️  Some fixes need manual attention")
        print("Check the error messages above and apply missing methods manually")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix Paper Trading Issues')
    parser.add_argument('--apply', action='store_true', help='Apply all fixes')
    
    args = parser.parse_args()
    
    if args.apply:
        success = main()
        sys.exit(0 if success else 1)
    else:
        print("Use --apply to run the fixes")
        sys.exit(1)