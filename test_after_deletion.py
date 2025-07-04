#!/usr/bin/env python3
"""
Test After Deletion - Verify system works after deleting historical tables
"""

import sys
from pathlib import Path
import psycopg2
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Get database connection using your existing config"""
    return psycopg2.connect(
        host=os.getenv('DATABASE_HOST'),
        port=os.getenv('DATABASE_PORT'),
        database=os.getenv('DATABASE_NAME'),
        user=os.getenv('DATABASE_USER'),
        password=os.getenv('DATABASE_PASSWORD')
    )

def check_historical_tables():
    """Check if any historical tables exist"""
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'historical_data_3m_%'
            AND table_schema = 'public'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        return tables

def test_database_connection():
    """Test database connection"""
    
    print("1. Testing database connection...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        db_manager = EnhancedDatabaseManager()
        
        if db_manager.test_connection():
            print("   ✅ Database connected successfully")
            return True, db_manager
        else:
            print("   ❌ Database connection failed")
            return False, None
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False, None

def test_table_recreation():
    """Test if tables can be recreated"""
    
    print("2. Testing table recreation...")
    
    try:
        from historical_data_download import HistoricalDataDownloader
        
        downloader = HistoricalDataDownloader()
        
        # Test creating a table
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        test_table = f"historical_data_3m_{now.year}_q{quarter}"
        
        print(f"   Creating test table: {test_table}")
        downloader.create_table(test_table)
        
        # Verify table was created
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (test_table,))
            
            columns = cursor.fetchall()
            
        if columns:
            print("   ✅ Table created successfully with columns:")
            for col_name, col_type in columns:
                print(f"      - {col_name}: {col_type}")
            return True
        else:
            print("   ❌ Table creation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Table recreation test failed: {e}")
        return False

def test_data_updater():
    """Test data updater works with new schema"""
    
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
    
    print("4. Testing historical downloader...")
    
    try:
        from historical_data_download import HistoricalDataDownloader, get_real_market_data
        
        downloader = HistoricalDataDownloader()
        
        # Test basic functionality
        print(f"   ✅ HistoricalDataDownloader initialized")
        print(f"   ✅ Market hours: {downloader.market_start} - {downloader.market_end}")
        
        # Test database config
        print(f"   ✅ Database config loaded:")
        print(f"      Host: {downloader.db_config['host']}")
        print(f"      Database: {downloader.db_config['database']}")
        
        # Test symbols loading
        try:
            symbols = downloader.get_symbols(5)
            print(f"   ✅ Symbol loading works: {len(symbols)} symbols")
        except Exception as e:
            print(f"   ⚠️ Symbol loading failed: {e} (normal without stocks_categories_table)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Historical downloader test failed: {e}")
        return False

def test_main_integration():
    """Test main.py live_data mode"""
    
    print("5. Testing main.py integration...")
    
    try:
        # Test import
        import main
        print("   ✅ Main module imports successfully")
        
        # Test config loading
        import config
        download_freq = getattr(config, 'DOWNLOAD_FREQUENCY', 'NOT_SET')
        print(f"   ✅ DOWNLOAD_FREQUENCY: {download_freq}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Main integration test failed: {e}")
        return False

def main():
    """Run post-deletion tests"""
    
    print("=" * 60)
    print("TEST AFTER DELETION - VERIFY SYSTEM WORKS")
    print("=" * 60)
    
    # Check if tables are actually deleted
    tables = check_historical_tables()
    
    if tables:
        print(f"⚠️ Warning: Found {len(tables)} historical tables still exist:")
        for table in tables:
            print(f"   - {table}")
        print("\nRun delete_synthetic_tables.py first")
        return
    else:
        print("✅ Confirmed: No historical data tables exist")
    
    print("\nTesting if system works after table deletion...\n")
    
    # Run tests
    tests = [
        test_database_connection,
        test_table_recreation, 
        test_data_updater,
        test_historical_downloader,
        test_main_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ SYSTEM WORKS PERFECTLY AFTER TABLE DELETION!")
        print("\n📋 READY FOR REAL DATA DOWNLOAD:")
        print("1. Set DOWNLOAD_FREQUENCY=once in .env")
        print("2. Add Kite API credentials to .env")
        print("3. Run: python main.py --mode live_data")
        print("4. Tables will be recreated automatically with real data")
        print("5. Date field = stock's timestamp, updated_at = insertion time")
    else:
        print(f"\n❌ SYSTEM NEEDS FIXES")
        print("Some components failed after table deletion")
        print("Please review the failed tests above")

if __name__ == "__main__":
    main()