#!/usr/bin/env python3
"""
FILE: verify_env_config.py
PURPOSE: Verify .env configuration matches your setup

DESCRIPTION:
- Checks .env file exists and has correct values
- Tests database connection with your port 5435
- Verifies historical data tables exist
- Validates paper trading configuration

USAGE:
- python verify_env_config.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_env_file():
    """Check .env file exists and has required variables"""
    
    print("1. Checking .env file...")
    
    if not Path('.env').exists():
        print("   ✗ .env file not found!")
        print("   Create .env file with your settings:")
        print("   DATABASE_HOST=localhost")
        print("   DATABASE_PORT=5435")
        print("   DATABASE_NAME=nexus_trading")
        print("   DATABASE_USER=trading_user")
        print("   DATABASE_PASSWORD=T")
        return False
    
    # Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'DATABASE_HOST': os.getenv('DATABASE_HOST'),
        'DATABASE_PORT': os.getenv('DATABASE_PORT'),
        'DATABASE_NAME': os.getenv('DATABASE_NAME'),
        'DATABASE_USER': os.getenv('DATABASE_USER'),
        'DATABASE_PASSWORD': os.getenv('DATABASE_PASSWORD')
    }
    
    missing_vars = []
    for var, value in required_vars.items():
        if not value:
            missing_vars.append(var)
            print(f"   ✗ {var}: Missing")
        else:
            # Show your actual values (mask password)
            display_value = "***" if var == 'DATABASE_PASSWORD' else value
            print(f"   ✓ {var}: {display_value}")
    
    if missing_vars:
        print(f"   Missing variables: {', '.join(missing_vars)}")
        return False
    
    # Verify your specific port
    if os.getenv('DATABASE_PORT') != '5435':
        print(f"   ⚠ Port is {os.getenv('DATABASE_PORT')}, expected 5435")
    
    return True

def test_database_connection():
    """Test connection with your specific configuration"""
    
    print("\n2. Testing database connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        import psycopg2
        from database.connection_config import get_connection_params
        
        # Get your connection parameters
        params = get_connection_params()
        print(f"   Connecting to: {params['host']}:{params['port']}")
        print(f"   Database: {params['database']}")
        print(f"   User: {params['user']}")
        
        # Test connection
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                print(f"   ✓ Connected to PostgreSQL: {version}")
                
                # Test your specific database
                cursor.execute("SELECT current_database()")
                db_name = cursor.fetchone()[0]
                print(f"   ✓ Current database: {db_name}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        print("   Check if PostgreSQL is running on port 5435")
        print("   Verify database name 'nexus_trading' exists")
        print("   Confirm user 'trading_user' has access")
        return False

def check_historical_data_tables():
    """Check if your historical data tables exist"""
    
    print("\n3. Checking historical data tables...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        import psycopg2
        from database.connection_config import get_connection_params
        from datetime import datetime
        
        params = get_connection_params()
        
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cursor:
                # Check stocks_categories_table
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'stocks_categories_table'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    cursor.execute("SELECT COUNT(*) FROM stocks_categories_table")
                    stock_count = cursor.fetchone()[0]
                    print(f"   ✓ stocks_categories_table: {stock_count} stocks")
                else:
                    print("   ✗ stocks_categories_table: Missing")
                    return False
                
                # Check current quarter historical data
                now = datetime.now()
                quarter = (now.month - 1) // 3 + 1
                table_name = f"historical_data_3m_{now.year}_q{quarter}"
                
                cursor.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}")
                    symbol_count = cursor.fetchone()[0]
                    print(f"   ✓ {table_name}: {symbol_count} symbols")
                else:
                    print(f"   ⚠ {table_name}: Missing - paper trading may use fallback prices")
                
                # Check for other quarters
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name LIKE 'historical_data_3m_%'
                    ORDER BY table_name
                """)
                
                historical_tables = [row[0] for row in cursor.fetchall()]
                print(f"   Available historical tables: {len(historical_tables)}")
                for table in historical_tables:
                    print(f"     - {table}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to check tables: {e}")
        return False

def check_paper_trading_config():
    """Check paper trading configuration"""
    
    print("\n4. Checking paper trading configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        import config
        
        # Check paper trading settings
        paper_mode = getattr(config, 'PAPER_TRADING_MODE', False)
        initial_capital = getattr(config, 'PAPER_TRADING_INITIAL_CAPITAL', 0)
        
        print(f"   Paper trading mode: {paper_mode}")
        print(f"   Initial capital: ₹{initial_capital:,.0f}")
        
        if not paper_mode:
            print("   ⚠ Paper trading is disabled - enable in config.py")
        
        if initial_capital <= 0:
            print("   ⚠ Initial capital not set - check config.py")
        
        return paper_mode and initial_capital > 0
        
    except Exception as e:
        print(f"   ✗ Config check failed: {e}")
        return False

def main():
    """Main verification function"""
    
    print("Environment Configuration Verification")
    print("=" * 50)
    print("Using your settings:")
    print("- Database Port: 5435")
    print("- Database Name: nexus_trading") 
    print("- Database User: trading_user")
    print()
    
    checks = [
        ("Environment File", check_env_file),
        ("Database Connection", test_database_connection),
        ("Historical Data Tables", check_historical_data_tables),
        ("Paper Trading Config", check_paper_trading_config)
    ]
    
    success_count = 0
    
    for check_name, check_func in checks:
        if check_func():
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"Verification Results: {success_count}/{len(checks)} passed")
    
    if success_count == len(checks):
        print("🎉 Environment verification successful!")
        print("Your configuration is ready for paper trading.")
        return True
    else:
        print("⚠️ Some issues found - fix them before running paper trading")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)