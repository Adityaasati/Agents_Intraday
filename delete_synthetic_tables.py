#!/usr/bin/env python3
"""
Delete Synthetic Data Tables - Clean slate for real historical data
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

def find_historical_tables():
    """Find all historical_data_3m_* tables"""
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE 'historical_data_3m_%'
            AND table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        return tables

def check_table_data(table_name):
    """Check what kind of data is in the table"""
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check total records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = cursor.fetchone()[0]
            
            # Check recent records (might be synthetic)
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE DATE(date) >= CURRENT_DATE - INTERVAL '7 days'
            """)
            recent_count = cursor.fetchone()[0]
            
            # Sample some data to see if it looks synthetic
            cursor.execute(f"""
                SELECT symbol, date, close, volume 
                FROM {table_name} 
                ORDER BY date DESC 
                LIMIT 5
            """)
            sample_data = cursor.fetchall()
            
            return {
                'total_count': total_count,
                'recent_count': recent_count,
                'sample_data': sample_data
            }
    except Exception as e:
        return {'error': str(e)}

def delete_historical_tables(confirm=False):
    """Delete all historical data tables"""
    
    if not confirm:
        print("❌ This will delete ALL historical data tables!")
        print("❌ This action cannot be undone!")
        response = input("Type 'DELETE ALL TABLES' to confirm: ")
        if response != 'DELETE ALL TABLES':
            print("❌ Deletion cancelled")
            return False
    
    tables = find_historical_tables()
    
    if not tables:
        print("✅ No historical data tables found")
        return True
    
    print(f"🗑️ Deleting {len(tables)} historical data tables...")
    
    deleted_count = 0
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for table in tables:
            try:
                print(f"   Deleting {table}...")
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {table}: {e}")
        
        conn.commit()
    
    print(f"✅ Deleted {deleted_count}/{len(tables)} tables")
    return deleted_count == len(tables)

def verify_clean_slate():
    """Verify all historical tables are gone"""
    
    tables = find_historical_tables()
    
    if not tables:
        print("✅ Clean slate confirmed - no historical data tables exist")
        return True
    else:
        print(f"⚠️ Warning: {len(tables)} historical tables still exist: {tables}")
        return False

def main():
    """Main function with safety checks"""
    
    print("=" * 60)
    print("DELETE SYNTHETIC DATA TABLES")
    print("=" * 60)
    
    try:
        # Step 1: Find existing tables
        print("1. Checking existing historical data tables...")
        tables = find_historical_tables()
        
        if not tables:
            print("   ✅ No historical data tables found")
            print("   ✅ Ready for fresh real data download")
            return
        
        print(f"   Found {len(tables)} historical data tables:")
        for table in tables:
            print(f"     - {table}")
        
        # Step 2: Analyze data in tables
        print("\n2. Analyzing data in tables...")
        
        for table in tables[:3]:  # Check first 3 tables
            print(f"\n   Checking {table}:")
            data_info = check_table_data(table)
            
            if 'error' in data_info:
                print(f"     ❌ Error: {data_info['error']}")
            else:
                print(f"     📊 Total records: {data_info['total_count']:,}")
                print(f"     📅 Recent records (7 days): {data_info['recent_count']:,}")
                
                if data_info['sample_data']:
                    print(f"     📋 Sample data:")
                    for row in data_info['sample_data'][:2]:
                        symbol, date, close, volume = row
                        print(f"        {symbol}: {date} | Close: {close} | Volume: {volume:,}")
        
        # Step 3: Confirm deletion
        print(f"\n3. Deletion options:")
        print(f"   This will delete ALL {len(tables)} historical data tables")
        print(f"   After deletion, run historical data download to get real data")
        print(f"   Tables will be recreated automatically with real data")
        
        # Step 4: Delete tables
        print(f"\n4. Deleting tables...")
        success = delete_historical_tables()
        
        if success:
            # Step 5: Verify clean slate
            print(f"\n5. Verifying clean slate...")
            verify_clean_slate()
            
            print(f"\n" + "=" * 60)
            print("✅ DELETION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print("📋 NEXT STEPS:")
            print("1. Set DOWNLOAD_FREQUENCY=once in .env")
            print("2. Add Kite API credentials to .env") 
            print("3. Run: python main.py --mode live_data")
            print("4. This will download real historical data and recreate tables")
        else:
            print(f"\n❌ DELETION FAILED")
            print("Some tables could not be deleted. Check the errors above.")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()