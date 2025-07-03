#!/usr/bin/env python3
"""
FILE: setup_paper_trading.py
PURPOSE: Setup Paper Trading features

DESCRIPTION:
- Adds paper trading configuration to config.py
- Extends existing agents with paper trading methods
- Creates paper trading database schema
- Validates setup

USAGE:
- python setup_paper_trading.py --execute
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_config():
    """Add paper trading config to config.py"""
    
    config_addition = '''
# ==========================================
# PAPER TRADING CONFIGURATION  
# ==========================================

# Paper Trading Settings
PAPER_TRADING_MODE = True
PAPER_TRADING_INITIAL_CAPITAL = float(os.getenv('PAPER_TRADING_CAPITAL', '100000'))
PAPER_TRADING_COMMISSION = 0.03  # 0.03% commission per trade
PAPER_TRADING_SLIPPAGE = 0.05    # 0.05% slippage simulation

# Order Management
MAX_ORDERS_PER_DAY = 20
ORDER_VALIDITY_HOURS = 24
AUTO_EXECUTE_SIGNALS = True

# Performance Tracking
TRACK_TRADE_PERFORMANCE = True
PERFORMANCE_REPORTING_FREQUENCY = 'daily'
BENCHMARK_SYMBOL = 'NIFTY50'

# Trade Execution Simulation
EXECUTION_DELAY_SECONDS = 2
MARKET_IMPACT_THRESHOLD = 50000
MARKET_IMPACT_PERCENT = 0.02

# Risk Controls for Paper Trading
PAPER_MAX_LOSS_PER_DAY = 5000
PAPER_MAX_POSITIONS = 10
PAPER_POSITION_SIZE_LIMIT = 0.15

# Reporting Configuration
GENERATE_DAILY_REPORTS = True
SAVE_TRADE_HISTORY = True
TRADE_HISTORY_RETENTION_DAYS = 90
'''
    
    try:
        config_path = Path('config.py')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            if 'PAPER_TRADING_MODE' not in content:
                with open(config_path, 'a') as f:
                    f.write(config_addition)
                print("✓ Added paper trading config to config.py")
            else:
                print("✓ Paper trading config already exists in config.py")
        else:
            print("✗ config.py not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config.py: {e}")
        return False

def setup_database_schema():
    """Setup paper trading database schema"""
    
    try:
        from database.schema_creator import SchemaCreator
        
        schema_creator = SchemaCreator()
        
        # Create paper trading tables
        success = schema_creator.create_paper_trading_tables()
        
        if success:
            print("✓ Paper trading database schema created")
        else:
            print("✗ Failed to create paper trading schema")
            
        return success
        
    except Exception as e:
        print(f"✗ Database schema setup failed: {e}")
        return False

def update_env_template():
    """Add paper trading variables to .env template"""
    
    env_addition = '''
# Paper Trading Configuration
PAPER_TRADING_CAPITAL=100000
'''
    
    try:
        env_template_path = Path('.env.template')
        
        if env_template_path.exists():
            with open(env_template_path, 'r') as f:
                content = f.read()
            
            if 'PAPER_TRADING_CAPITAL' not in content:
                with open(env_template_path, 'a') as f:
                    f.write(env_addition)
                print("✓ Updated .env.template")
            else:
                print("✓ .env.template already contains paper trading vars")
        else:
            print("ℹ .env.template not found (optional)")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to update .env.template: {e}")
        return False

def validate_setup():
    """Validate paper trading setup"""
    
    print("\nValidating Paper Trading Setup")
    print("=" * 40)
    
    checks = {
        'Config Updated': False,
        'Database Schema': False,
        'Agent Methods': False,
        'Manager Class': False
    }
    
    try:
        import config
        if hasattr(config, 'PAPER_TRADING_MODE'):
            checks['Config Updated'] = True
        
        from database.enhanced_database_manager import EnhancedDatabaseManager
        db_manager = EnhancedDatabaseManager()
        
        if db_manager.test_connection():
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'agent_paper_performance'
                        )
                    """)
                    if cursor.fetchone()[0]:
                        checks['Database Schema'] = True
        
        from agents.portfolio_agent import PortfolioAgent
        portfolio_agent = PortfolioAgent(db_manager)
        
        if hasattr(portfolio_agent, 'execute_paper_trade'):
            checks['Agent Methods'] = True
        
        from paper_trading_manager import PaperTradingManager
        manager = PaperTradingManager()
        
        if hasattr(manager, 'run_paper_trading_session'):
            checks['Manager Class'] = True
        
    except Exception as e:
        print(f"Validation error: {e}")
    
    for check, status in checks.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"  {check:20}: {status_str}")
    
    all_passed = all(checks.values())
    print(f"\nOverall Status: {'✅ READY' if all_passed else '❌ ISSUES'}")
    
    return all_passed

def main():
    """Main setup function"""
    
    print("Paper Trading Setup")
    print("=" * 50)
    
    setup_tasks = [
        ("Config Setup", setup_config),
        ("Database Schema", setup_database_schema), 
        ("Environment Template", update_env_template)
    ]
    
    success_count = 0
    
    for task_name, task_func in setup_tasks:
        print(f"\n{task_name}...")
        if task_func():
            success_count += 1
    
    print(f"\nSetup Results: {success_count}/{len(setup_tasks)} tasks completed")
    
    if validate_setup():
        print("\n🎉 Paper Trading Setup Complete!")
        print("\nNext Steps:")
        print("1. Run: python paper_trading_manager.py --mode validate")
        print("2. Run: python main.py --mode paper")
        print("3. Run: python paper_trading_manager.py --mode report")
        return True
    else:
        print("\n⚠️ Setup completed with issues - check validation results")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Paper Trading')
    parser.add_argument('--execute', action='store_true', help='Execute setup')
    
    args = parser.parse_args()
    
    if args.execute:
        success = main()
        sys.exit(0 if success else 1)
    else:
        print("Use --execute to run the setup")
        sys.exit(1)