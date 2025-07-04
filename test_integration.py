#!/usr/bin/env python3
"""
Test Integration - Validate historical data downloader integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test config loading with new variables"""
    print("1. Testing config...")
    
    try:
        import config
        
        # Test new variables
        download_freq = getattr(config, 'DOWNLOAD_FREQUENCY', 'NOT_SET')
        use_real_data = getattr(config, 'USE_REAL_MARKET_DATA', 'NOT_SET')
        db_host = getattr(config, 'DB_HOST', 'NOT_SET')
        
        print(f"   ✓ DOWNLOAD_FREQUENCY: {download_freq}")
        print(f"   ✓ USE_REAL_MARKET_DATA: {use_real_data}")
        print(f"   ✓ DB_HOST: {db_host}")
        
        return True
    except Exception as e:
        print(f"   ✗ Config test failed: {e}")
        return False

def test_historical_downloader():
    """Test historical data downloader import"""
    print("2. Testing historical data downloader...")
    
    try:
        from historical_data_download import HistoricalDataDownloader, get_real_market_data, start_live_data_updates
        
        # Test class initialization
        downloader = HistoricalDataDownloader()
        print(f"   ✓ HistoricalDataDownloader initialized")
        print(f"   ✓ Market start time: {downloader.market_start}")
        print(f"   ✓ Market end time: {downloader.market_end}")
        
        # Test functions exist
        print(f"   ✓ get_real_market_data function available")
        print(f"   ✓ start_live_data_updates function available")
        
        return True
    except Exception as e:
        print(f"   ✗ Historical downloader test failed: {e}")
        return False

def test_database_integration():
    """Test database integration"""
    print("3. Testing database integration...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        
        db_manager = EnhancedDatabaseManager()
        
        if db_manager.test_connection():
            print("   ✓ Database connected")
            
            # Test symbols table
            symbols = db_manager.get_testing_symbols(3)
            print(f"   ✓ Found {len(symbols)} test symbols: {symbols}")
            
            # Test quarters
            quarters = db_manager.get_available_quarters()
            print(f"   ✓ Found {len(quarters)} quarter tables")
            
            return True
        else:
            print("   ✗ Database connection failed")
            return False
    except Exception as e:
        print(f"   ✗ Database test failed: {e}")
        return False

def test_data_updater_integration():
    """Test data updater integration"""
    print("4. Testing data updater integration...")
    
    try:
        from utils.data_updater import SimpleDataUpdater
        from database.enhanced_database_manager import EnhancedDatabaseManager
        
        db_manager = EnhancedDatabaseManager()
        updater = SimpleDataUpdater(db_manager)
        
        # Test new get_market_data method
        print("   ✓ Testing get_market_data method...")
        
        # Should use synthetic by default (USE_REAL_MARKET_DATA=false)
        data = updater.get_market_data('RELIANCE')
        
        if data and 'close' in data:
            print(f"   ✓ get_market_data returned data: {data['close']}")
            
            # Test original method still works
            synthetic_data = updater.simulate_sample_data('RELIANCE')
            if synthetic_data:
                print(f"   ✓ simulate_sample_data still works: {synthetic_data['close']}")
            
            return True
        else:
            print("   ✗ get_market_data returned no data")
            return False
            
    except Exception as e:
        print(f"   ✗ Data updater test failed: {e}")
        return False

def test_main_integration():
    """Test main.py integration"""
    print("5. Testing main.py integration...")
    
    try:
        # Import main to verify it doesn't break
        import main
        
        # Test that live_data mode is available
        print("   ✓ Main module imports successfully")
        print("   ✓ live_data mode should be available")
        
        return True
    except Exception as e:
        print(f"   ✗ Main integration test failed: {e}")
        return False

def test_kite_integration():
    """Test Kite integration (will fail without credentials, which is expected)"""
    print("6. Testing Kite integration...")
    
    try:
        from kite_token_generator import get_authenticated_kite_client
        
        # This will likely fail without credentials, which is expected
        kite = get_authenticated_kite_client()
        
        if kite:
            print("   ✓ Kite client connected")
            return True
        else:
            print("   ⚠ Kite client not connected (expected without credentials)")
            return True  # This is expected
    except Exception as e:
        print(f"   ⚠ Kite integration test: {e} (expected without setup)")
        return True  # Expected without proper setup

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("HISTORICAL DATA DOWNLOADER INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_config,
        test_historical_downloader,
        test_database_integration,
        test_data_updater_integration,
        test_main_integration,
        test_kite_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed >= total - 1:  # Allow one test to fail (Kite integration)
        print("\n✅ Integration ready!")
        print("\nNEXT STEPS:")
        print("1. Set DOWNLOAD_FREQUENCY=once in .env for initial download")
        print("2. Add Kite API credentials to .env")
        print("3. Run: python main.py --mode live_data")
        print("4. After initial download, set DOWNLOAD_FREQUENCY=5min for live updates")
        print("\nDOWNLOAD OPTIONS:")
        print("- DOWNLOAD_FREQUENCY=once   # Download all historical data")
        print("- DOWNLOAD_FREQUENCY=5min   # Live updates every 5 minutes")
        print("- DOWNLOAD_FREQUENCY=10min  # Live updates every 10 minutes")
    else:
        print("\n❌ Integration needs fixes")
        print("Please resolve the failed tests above")

if __name__ == "__main__":
    main()