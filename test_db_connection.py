"""
Quick Database Connection Test
=============================
Test database connectivity before running full test suite
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_env_file():
    """Test if .env file exists and is readable"""
    print("Step 1: Checking .env file...")
    print("-" * 40)
    
    env_files = [Path('.env'), Path('../.env')]
    env_found = False
    
    for env_path in env_files:
        if env_path.exists():
            print(f"✅ Found .env file: {env_path.absolute()}")
            env_found = True
            
            # Read and display content (mask passwords)
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            print("\n.env file contents:")
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if 'PASSWORD' in line:
                        key = line.split('=')[0]
                        print(f"  {key}=*****")
                    else:
                        print(f"  {line}")
            break
    
    if not env_found:
        print("❌ No .env file found!")
        print("Create a .env file with:")
        print("""
DATABASE_HOST=localhost
DATABASE_PORT=5435
DATABASE_NAME=your_database_name
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password
""")
        return False
    
    return True

def test_dotenv_import():
    """Test if python-dotenv is available"""
    print("\nStep 2: Checking python-dotenv...")
    print("-" * 40)
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv is available")
        return True
    except ImportError:
        print("❌ python-dotenv not installed")
        print("Install with: pip install python-dotenv")
        return False

def test_connection_config():
    """Test connection configuration loading"""
    print("\nStep 3: Testing connection configuration...")
    print("-" * 40)
    
    try:
        # Update the connection_config.py file first
        from database.connection_config import get_connection_params, validate_connection_config
        
        params = get_connection_params()
        print("Connection parameters loaded:")
        for key, value in params.items():
            if key == 'password':
                print(f"  {key}: {'*' * len(value) if value else 'NOT SET'}")
            else:
                print(f"  {key}: {value}")
        
        is_valid = validate_connection_config()
        return is_valid
        
    except ImportError as e:
        print(f"❌ Failed to import connection config: {e}")
        return False

def test_raw_connection():
    """Test raw PostgreSQL connection"""
    print("\nStep 4: Testing raw database connection...")
    print("-" * 40)
    
    try:
        import psycopg2
        from database.connection_config import get_connection_params
        
        params = get_connection_params()
        
        print(f"Attempting connection to: {params['host']}:{params['port']}")
        
        conn = psycopg2.connect(
            host=params['host'],
            port=int(params['port']),
            database=params['database'],
            user=params['user'],
            password=params['password']
        )
        
        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print("✅ Raw database connection successful!")
        print(f"PostgreSQL version: {version}")
        return True
        
    except Exception as e:
        print(f"❌ Raw database connection failed: {e}")
        return False

def test_enhanced_db_manager():
    """Test enhanced database manager"""
    print("\nStep 5: Testing Enhanced Database Manager...")
    print("-" * 40)
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        
        db_manager = EnhancedDatabaseManager()
        
        # Test connection
        if db_manager.test_connection():
            print("✅ Enhanced Database Manager connection successful!")
            
            # Test pool status
            pool_status = db_manager.get_pool_status()
            print(f"Connection pool status: {pool_status}")
            
            # Test basic health check
            health = db_manager.get_system_health()
            print(f"System health: {health}")
            
            return True
        else:
            print("❌ Enhanced Database Manager connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced Database Manager test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("DATABASE CONNECTION DIAGNOSTIC TEST")
    print("=" * 60)
    
    tests = [
        ("Environment File", test_env_file),
        ("Python Dotenv", test_dotenv_import),
        ("Connection Config", test_connection_config),
        ("Raw Connection", test_raw_connection),
        ("Enhanced DB Manager", test_enhanced_db_manager)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your database connection is ready.")
        print("You can now run: python run_all_tests.py --mode quick")
    else:
        print("❌ Some tests failed. Please fix the issues above before running the full test suite.")
        
        # Provide specific guidance
        if not results.get("Environment File"):
            print("\n🔧 NEXT STEPS:")
            print("1. Create a .env file in your project root with your database settings")
            print("2. Make sure it includes DATABASE_PORT=5435")
        
        if not results.get("Python Dotenv"):
            print("\n🔧 NEXT STEPS:")
            print("1. Install python-dotenv: pip install python-dotenv")
        
        if not results.get("Raw Connection"):
            print("\n🔧 NEXT STEPS:")
            print("1. Check if PostgreSQL is running on port 5435")
            print("2. Verify database name, username, and password in .env")
            print("3. Check if your database allows connections from localhost")

if __name__ == "__main__":
    main()