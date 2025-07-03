#!/usr/bin/env python3
"""
FILE: setup_live_trading.py
PURPOSE: Setup Live Trading features with Trade Mode Control

DESCRIPTION:
- Adds live trading configuration to config.py
- Creates live trading database schema
- Sets up token generator
- Updates .env template

USAGE:
- python setup_live_trading.py --execute
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# CRITICAL FIX: Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

def setup_config():
    """Add live trading config to config.py"""
    
    config_addition = '''
# ==========================================
# LIVE TRADING CONFIGURATION
# ==========================================

# Master Trading Control
TRADE_MODE = os.getenv('TRADE_MODE', 'no').lower() == 'yes'  # Simple yes/no control
LIVE_TRADING_MODE = bool(os.getenv('LIVE_TRADING_MODE', 'False').lower() == 'true')
LIVE_TRADING_CAPITAL = float(os.getenv('LIVE_TRADING_CAPITAL', '50000'))

# Kite API Configuration (simplified)
KITE_API_KEY = os.getenv('KITE_API_KEY', '')
KITE_API_SECRET = os.getenv('KITE_API_SECRET', '')

# Trading Logic Control
GENERATE_SIGNALS_ONLY = not TRADE_MODE

# Order Management
LIVE_ORDER_TIMEOUT_SECONDS = 30
LIVE_ORDER_RETRY_ATTEMPTS = 3
LIVE_ORDER_RETRY_DELAY_SECONDS = 2

# Live Trading Risk Controls (Conservative)
LIVE_MAX_LOSS_PER_DAY = 5000
LIVE_MAX_POSITIONS = 5
LIVE_POSITION_SIZE_LIMIT = 0.10  # 10% max per position
LIVE_MIN_CONFIDENCE_THRESHOLD = 0.75

# Market Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Live Execution Settings
LIVE_EXECUTION_DELAY_SECONDS = 5
LIVE_PRICE_TOLERANCE_PERCENT = 0.5
LIVE_QUANTITY_ROUNDING = True

# Monitoring and Alerts
LIVE_MONITORING_INTERVAL_SECONDS = 30
LIVE_POSITION_CHECK_INTERVAL_SECONDS = 60
LIVE_HEARTBEAT_INTERVAL_SECONDS = 300

# Safety Mechanisms
LIVE_CIRCUIT_BREAKER_LOSS_PERCENT = 3.0
LIVE_EMERGENCY_EXIT_ENABLED = True
LIVE_MAX_ORDERS_PER_MINUTE = 5

# Token Management
KITE_TOKEN_FILE = "kite_token.txt"
AUTO_REFRESH_TOKEN = True

# Live Trading Logging
LIVE_TRADING_LOG_LEVEL = 'INFO'
LIVE_ORDER_LOG_RETENTION_DAYS = 30

# Approved Symbols for Live Trading (Conservative list)
LIVE_TRADING_APPROVED_SYMBOLS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC', 
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'LT', 'WIPRO'
]
'''
    
    try:
        config_path = Path('config.py')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            if 'TRADE_MODE' not in content:
                with open(config_path, 'a') as f:
                    f.write(config_addition)
                print("✓ Added live trading config to config.py")
            else:
                print("✓ Live trading config already exists in config.py")
        else:
            print("✗ config.py not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config.py: {e}")
        return False

def setup_database_schema():
    """Setup live trading database schema"""
    
    try:
        # Import AFTER loading environment variables
        from database.schema_creator import SchemaCreator
        
        # Test connection first with debug info
        print("Testing database connection...")
        schema_creator = SchemaCreator()
        
        # Debug: Print connection params (mask password)
        connection_debug = {
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT'),
            'database': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': '***' if os.getenv('DATABASE_PASSWORD') else 'MISSING'
        }
        print(f"Connection params: {connection_debug}")
        
        # Test basic connection
        if not schema_creator.test_database_connection():
            print("✗ Database connection test failed")
            return False
        
        print("✓ Database connection successful")
        
        # Create live trading tables
        success = schema_creator.create_live_trading_tables()
        
        if success:
            print("✓ Live trading database schema created")
        else:
            print("✗ Failed to create live trading schema")
            
        return success
        
    except Exception as e:
        print(f"✗ Database schema setup failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False

def update_env_template():
    """Add live trading variables to .env template"""
    
    env_addition = '''
# Live Trading Configuration
TRADE_MODE=no                        # Set to 'yes' to enable actual trading
LIVE_TRADING_MODE=true               # Enable live trading system
LIVE_TRADING_CAPITAL=50000           # Capital for live trading

# Kite API Credentials (get from Zerodha Developer Console)
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
'''
    
    try:
        env_template_path = Path('.env.template')
        
        if env_template_path.exists():
            with open(env_template_path, 'r') as f:
                content = f.read()
            
            if 'TRADE_MODE' not in content:
                with open(env_template_path, 'a') as f:
                    f.write(env_addition)
                print("✓ Updated .env.template")
            else:
                print("✓ .env.template already contains live trading vars")
        else:
            # Create .env.template if it doesn't exist
            with open(env_template_path, 'w') as f:
                f.write("""# Nexus Trading System Environment Variables

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5435
DATABASE_NAME=nexus_trading
DATABASE_USER=your_db_user
DATABASE_PASSWORD=your_db_password

# Paper Trading Configuration
PAPER_TRADING_CAPITAL=100000
""")
                f.write(env_addition)
            print("✓ Created .env.template with live trading configuration")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to update .env.template: {e}")
        return False

def setup_token_generator():
    """Copy token generator to project root if needed"""
    
    try:
        token_generator_path = Path('kite_token_generator.py')
        
        if token_generator_path.exists():
            print("✓ kite_token_generator.py already exists")
            return True
        
        # Create the token generator script (content from previous artifact)
        token_generator_content = '''#!/usr/bin/env python3
"""
FILE: kite_token_generator.py
PURPOSE: Generate and manage Kite API tokens
"""

from kiteconnect import KiteConnect
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

KITE_API_KEY = os.getenv('KITE_API_KEY')
KITE_API_SECRET = os.getenv('KITE_API_SECRET')
TOKEN_FILE = "kite_token.txt"

def first_run():
    """Generate and save Kite Connect access token"""
    
    if not KITE_API_KEY or not KITE_API_SECRET:
        print("Error: KITE_API_KEY and KITE_API_SECRET must be set in .env file")
        return None
    
    # Check if token exists and is valid for today
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                token_data = json.loads(f.read().strip())
                today = datetime.now().strftime("%Y-%m-%d")
                
                if token_data.get("date") == today and token_data.get("access_token"):
                    print(f"Valid token found for {today}")
                    return token_data["access_token"]
                else:
                    os.remove(TOKEN_FILE)  # Delete old token
        except:
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
    
    # Generate new token
    print("Generating new Kite API token...")
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url()
    
    print("\\nSTEPS TO GENERATE TOKEN:")
    print("1. Open this URL in browser:")
    print(f"   {login_url}")
    print("2. Login with your Zerodha credentials")
    print("3. Copy the request_token from the redirected URL")
    print("4. Paste it below\\n")
    
    request_token = input("Paste request_token: ").strip()
    
    if not request_token:
        print("Error: No request token provided")
        return None
    
    try:
        session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = session["access_token"]
        
        # Save with today's date
        token_data = {
            "access_token": access_token, 
            "date": datetime.now().strftime("%Y-%m-%d"),
            "user_id": session.get("user_id", "unknown")
        }
        
        with open(TOKEN_FILE, "w") as f:
            f.write(json.dumps(token_data, indent=2))
        
        print(f"✓ Token generated and saved for {token_data['date']}")
        print(f"✓ User: {token_data['user_id']}")
        
        return access_token
        
    except Exception as e:
        print(f"Failed to generate session: {e}")
        return None

def main():
    """Main token generation function"""
    
    print("Kite API Token Generator")
    print("=" * 40)
    
    token = first_run()
    if token:
        print("\\n✅ Token ready for live trading")
        print("\\nNext steps:")
        print("1. Set TRADE_MODE=yes in .env to enable trading")
        print("2. Run: python live_trading_manager.py --mode validate")
        print("3. Run: python main.py --mode live")
    else:
        print("\\n❌ Token generation failed")

if __name__ == "__main__":
    main()
'''
        
        with open(token_generator_path, 'w') as f:
            f.write(token_generator_content)
        
        # Make it executable
        import stat
        os.chmod(token_generator_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print("✓ Created kite_token_generator.py")
        return True
        
    except Exception as e:
        print(f"✗ Failed to setup token generator: {e}")
        return False

def validate_setup():
    """Validate live trading setup"""
    
    print("\nValidating Live Trading Setup")
    print("=" * 40)
    
    checks = {
        'Config Updated': False,
        'Database Schema': False,
        'Token Generator': False,
        'Agent Methods': False
    }
    
    try:
        import config
        if hasattr(config, 'TRADE_MODE'):
            checks['Config Updated'] = True
        
        from database.enhanced_database_manager import EnhancedDatabaseManager
        db_manager = EnhancedDatabaseManager()
        
        if db_manager.test_connection():
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'agent_live_orders'
                        )
                    """)
                    if cursor.fetchone()[0]:
                        checks['Database Schema'] = True
        
        token_generator_path = Path('kite_token_generator.py')
        if token_generator_path.exists():
            checks['Token Generator'] = True
        
        from agents.portfolio_agent import PortfolioAgent
        portfolio_agent = PortfolioAgent(db_manager)
        
        if hasattr(portfolio_agent, 'execute_live_trade'):
            checks['Agent Methods'] = True
        
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
    
    print("Live Trading Setup with Trade Mode Control")
    print("=" * 50)
    
    # Verify environment variables are loaded
    print(f"\nEnvironment Check:")
    print(f"  DATABASE_HOST: {os.getenv('DATABASE_HOST', 'MISSING')}")
    print(f"  DATABASE_PORT: {os.getenv('DATABASE_PORT', 'MISSING')}")
    print(f"  DATABASE_NAME: {os.getenv('DATABASE_NAME', 'MISSING')}")
    print(f"  DATABASE_USER: {os.getenv('DATABASE_USER', 'MISSING')}")
    print(f"  DATABASE_PASSWORD: {'***' if os.getenv('DATABASE_PASSWORD') else 'MISSING'}")
    
    setup_tasks = [
        ("Config Setup", setup_config),
        ("Database Schema", setup_database_schema),
        ("Environment Template", update_env_template),
        ("Token Generator", setup_token_generator)
    ]
    
    success_count = 0
    
    for task_name, task_func in setup_tasks:
        print(f"\n{task_name}...")
        if task_func():
            success_count += 1
    
    print(f"\nSetup Results: {success_count}/{len(setup_tasks)} tasks completed")
    
    # Validate complete setup
    if validate_setup():
        print("\n🎉 Live Trading Setup Complete!")
        
        print("\n📋 USAGE MODES:")
        print("\n1. SIGNAL GENERATION ONLY (Safe):")
        print("   - Set TRADE_MODE=no in .env")
        print("   - Run: python main.py --mode signals")
        print("   - Run: python main.py --mode live")
        print("   - Generates signals without placing orders")
        
        print("\n2. LIVE TRADING (Real Money):")
        print("   - Generate token: python kite_token_generator.py")
        print("   - Set TRADE_MODE=yes in .env")
        print("   - Run: python main.py --mode live")
        print("   - Places actual orders via Kite API")
        
        print("\n⚠️  SAFETY RECOMMENDATIONS:")
        print("   - Start with TRADE_MODE=no to test signals")
        print("   - Use small capital initially (₹25,000-₹50,000)")
        print("   - Monitor actively during market hours")
        
        return True
    else:
        print("\n⚠️ Setup completed with issues - check validation results")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Live Trading')
    parser.add_argument('--execute', action='store_true', help='Execute setup')
    
    args = parser.parse_args()
    
    if args.execute:
        success = main()
        sys.exit(0 if success else 1)
    else:
        print("Use --execute to run the setup")
        sys.exit(1)