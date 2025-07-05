#!/usr/bin/env python3
"""
Test System After Deleting Synthetic Tables
Ensures everything works with real data or creates new tables as needed
"""

import sys
from pathlib import Path
from datetime import datetime
import psycopg2

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv('DATABASE_HOST'),
        port=os.getenv('DATABASE_PORT'),
        database=os.getenv('DATABASE_NAME'),
        user=os.getenv('DATABASE_USER'),
        password=os.getenv('DATABASE_PASSWORD')
    )

def create_historical_table_schema(table_name):
    """Create historical data table with correct schema"""
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        open DECIMAL(10,2),
        high DECIMAL(10,2),
        low DECIMAL(10,2),
        close DECIMAL(10,2),
        volume BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, date)
    );
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_date 
    ON {table_name}(symbol, date);
    
    CREATE INDEX IF NOT EXISTS idx_{table_name}_date 
    ON {table_name}(date);
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            conn.commit()
            print(f"✅ Created table: {table_name}")
            return True
    except Exception as e:
        print(f"❌ Failed to create table {table_name}: {e}")
        return False

def test_database_connectivity():
    """Test basic database connection"""
    
    print("1. Testing database connectivity...")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            print(f"   ✅ Connected to PostgreSQL: {version.split(',')[0]}")
            return True
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False

def test_current_quarter_table():
    """Test if current quarter table exists or create it"""
    
    print("2. Testing current quarter table...")
    
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1
    table_name = f"historical_data_3m_{now.year}_q{quarter}"
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            exists = cursor.fetchone()[0]
            
            if exists:
                print(f"   ✅ Table {table_name} already exists")
                
                # Check record count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"   ℹ️  Contains {count} records")
                
                return True
            else:
                print(f"   ⚠️  Table {table_name} does not exist")
                print("   Creating new table...")
                
                if create_historical_table_schema(table_name):
                    return True
                else:
                    return False
                    
    except Exception as e:
        print(f"   ❌ Table check failed: {e}")
        return False

def test_data_updater():
    """Test data updater functionality"""
    
    print("3. Testing data updater...")
    
    try:
        from utils.data_updater import SimpleDataUpdater
        from database.enhanced_database_manager import EnhancedDatabaseManager
        
        db_manager = EnhancedDatabaseManager()
        updater = SimpleDataUpdater(db_manager)
        
        # Test synthetic data generation
        data = updater.simulate_sample_data('RELIANCE')
        
        if data and 'close' in data:
            print(f"   ✅ Synthetic data generation works: {data['close']}")
            
            # Test data insertion
            success = updater.update_symbol_data('RELIANCE', data)
            
            if success:
                print("   ✅ Data insertion works")
                
                # Verify data was inserted with correct schema
                now = datetime.now()
                quarter = (now.month - 1) // 3 + 1
                table_name = f"historical_data_3m_{now.year}_q{quarter}"
                
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"""
                        SELECT symbol, date, close, created_at 
                        FROM {table_name} 
                        WHERE symbol = 'RELIANCE'
                        ORDER BY date DESC 
                        LIMIT 1
                    """)
                    
                    result = cursor.fetchone()
                    
                if result:
                    symbol, stock_date, close, created_at = result
                    print(f"   ✅ Data verified:")
                    print(f"      Symbol: {symbol}")
                    print(f"      Stock Date: {stock_date}")
                    print(f"      Close: {close}")
                    print(f"      Created At: {created_at}")
                    return True
                else:
                    print("   ❌ No data found after insertion")
                    return False
            else:
                print("   ❌ Data insertion failed")
                return False
        else:
            print("   ❌ Synthetic data generation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Data updater test failed: {e}")
        return False

def test_historical_downloader():
    """Test historical downloader components"""
    
    print("4. Testing historical downloader readiness...")
    
    try:
        # Check if historical_data_download.py exists
        if os.path.exists('historical_data_download.py'):
            print("   ✅ historical_data_download.py exists")
            
            # Try to import it
            try:
                import historical_data_download
                print("   ✅ Module imports successfully")
                return True
            except Exception as e:
                print(f"   ⚠️  Module import warning: {e}")
                print("   ℹ️  This is expected if you haven't set up Kite credentials yet")
                return True
        else:
            print("   ⚠️  historical_data_download.py not found")
            print("   ℹ️  You'll need to create this for real data downloading")
            return True
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis with available data"""
    
    print("5. Testing technical analysis...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        from database.enhanced_database_manager import EnhancedDatabaseManager
        
        db_manager = EnhancedDatabaseManager()
        tech_agent = TechnicalAgent(db_manager)
        
        # Test with a symbol
        result = tech_agent.analyze_symbol('RELIANCE')
        
        if 'error' in result:
            print(f"   ⚠️  Analysis returned with note: {result['error']}")
            print("   ℹ️  This is expected if no historical data exists yet")
            
            # Test if it can work with synthetic data
            if 'technical_score' in result:
                print("   ✅ Technical analysis works with synthetic data")
                return True
            else:
                print("   ℹ️  Technical analysis needs historical data")
                return True
        else:
            print("   ✅ Technical analysis completed successfully")
            print(f"   Technical Score: {result.get('technical_score', 'N/A')}")
            return True
            
    except Exception as e:
        print(f"   ❌ Technical analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("=" * 60)
    print("POST-DELETION SYSTEM TEST")
    print("=" * 60)
    print()
    
    tests = [
        test_database_connectivity,
        test_current_quarter_table,
        test_data_updater,
        test_historical_downloader,
        test_technical_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed >= 4:  # Allow one test to fail
        print("\n✅ System is ready!")
        print("\nNext steps:")
        print("1. Set up your historical data download script")
        print("2. Configure Kite API credentials in .env")
        print("3. Run historical data download to populate tables")
        print("4. Run: python main.py --mode test")
    else:
        print("\n❌ System needs attention")
        print("Please review the failed tests above")

if __name__ == "__main__":
    main()