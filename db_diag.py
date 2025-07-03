#!/usr/bin/env python3
"""
Database Connection Diagnostic Tool
"""

import os
import psycopg2
from dotenv import load_dotenv

def diagnose_database_connection():
    """Diagnose database connection issues step by step"""
    
    print("🔍 Database Connection Diagnostic")
    print("=" * 50)
    
    # Step 1: Check .env file
    print("\n1. Checking .env file...")
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📝 Create .env file from .env.template:")
        print("   cp .env.template .env")
        return False
    else:
        print("✅ .env file exists")
    
    # Step 2: Load environment variables
    print("\n2. Loading environment variables...")
    load_dotenv()
    
    # Step 3: Check required variables
    print("\n3. Checking required database variables...")
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
            print(f"❌ {var}: Missing or empty")
            missing_vars.append(var)
        else:
            # Mask password for security
            display_value = "***" if var == 'DATABASE_PASSWORD' else value
            print(f"✅ {var}: {display_value}")
    
    if missing_vars:
        print(f"\n🚨 Missing variables: {', '.join(missing_vars)}")
        print("📝 Please add these to your .env file")
        return False
    
    # Step 4: Test database connection
    print("\n4. Testing database connection...")
    try:
        connection_params = {
            'host': required_vars['DATABASE_HOST'],
            'port': int(required_vars['DATABASE_PORT']),
            'database': required_vars['DATABASE_NAME'],
            'user': required_vars['DATABASE_USER'],
            'password': required_vars['DATABASE_PASSWORD']
        }
        
        with psycopg2.connect(**connection_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result[0] == 1:
                    print("✅ Database connection successful!")
                    return True
                    
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        print(f"❌ Database connection failed: {error_msg}")
        
        # Provide specific troubleshooting based on error
        if "fe_sendauth: no password supplied" in error_msg:
            print("\n🔧 Solution: DATABASE_PASSWORD is missing or empty in .env")
        elif "password authentication failed" in error_msg:
            print("\n🔧 Solution: Wrong username or password")
        elif "could not connect to server" in error_msg:
            print("\n🔧 Solution: PostgreSQL service not running or wrong host/port")
        elif "database" in error_msg and "does not exist" in error_msg:
            print("\n🔧 Solution: Database 'nexus_trading' doesn't exist")
        
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def suggest_fixes():
    """Suggest common fixes for database issues"""
    
    print("\n🛠️ Common Fixes:")
    print("=" * 30)
    
    print("\n1. PostgreSQL Service:")
    print("   # Windows (if using PostgreSQL service)")
    print("   net start postgresql-x64-13")
    print("   # Or start through Services.msc")
    
    print("\n2. Create Database (if missing):")
    print("   psql -U postgres")
    print("   CREATE DATABASE nexus_trading;")
    print("   CREATE USER your_username WITH PASSWORD 'your_password';")
    print("   GRANT ALL PRIVILEGES ON DATABASE nexus_trading TO your_username;")
    
    print("\n3. Test Manual Connection:")
    print("   psql -h localhost -p 5435 -U your_username -d nexus_trading")
    
    print("\n4. Check .env File Format:")
    print("   DATABASE_HOST=localhost")
    print("   DATABASE_PORT=5435")
    print("   DATABASE_NAME=nexus_trading")
    print("   DATABASE_USER=your_actual_username")
    print("   DATABASE_PASSWORD=your_actual_password")

if __name__ == "__main__":
    success = diagnose_database_connection()
    
    if not success:
        suggest_fixes()
        print("\n❌ Fix the issues above and try again")
    else:
        print("\n🎉 Database connection is working!")
        print("💡 You can now run: python setup_live_trading.py --execute")