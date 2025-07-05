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
        print("❌ Make sure you have backups if needed!")
        print()
        response = input("Type 'DELETE ALL TABLES' to confirm: ")
        if response != 'DELETE ALL TABLES':
            print("Operation cancelled.")
            return False
    
    tables = find_historical_tables()
    
    if not tables:
        print("No historical data tables found.")
        return True
    
    print(f"\nFound {len(tables)} historical data tables:")
    for table in tables:
        data_info = check_table_data(table)
        print(f"  - {table}: {data_info.get('total_count', 0)} records")
    
    print("\nDeleting tables...")
    
    deleted_count = 0
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for table in tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                conn.commit()
                print(f"  ✅ Deleted: {table}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete {table}: {e}")
                conn.rollback()
    
    print(f"\nDeleted {deleted_count}/{len(tables)} tables.")
    return deleted_count == len(tables)

def delete_specific_table(table_name):
    """Delete a specific table"""
    
    print(f"Checking table: {table_name}")
    
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
        
        if not exists:
            print(f"Table {table_name} does not exist.")
            return True
        
        # Get table info
        data_info = check_table_data(table_name)
        print(f"Table contains {data_info.get('total_count', 0)} records")
        
        # Delete the table
        try:
            cursor.execute(f"DROP TABLE {table_name} CASCADE")
            conn.commit()
            print(f"✅ Successfully deleted: {table_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete {table_name}: {e}")
            conn.rollback()
            return False

def main():
    """Main execution"""
    
    print("=" * 60)
    print("SYNTHETIC DATA TABLE DELETION")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Delete synthetic data tables')
    parser.add_argument('--all', action='store_true', help='Delete ALL historical tables')
    parser.add_argument('--table', help='Delete specific table')
    parser.add_argument('--list', action='store_true', help='List all historical tables')
    parser.add_argument('--check', help='Check data in specific table')
    
    args = parser.parse_args()
    
    if args.list:
        tables = find_historical_tables()
        print(f"Found {len(tables)} historical tables:")
        for table in tables:
            print(f"  - {table}")
    
    elif args.check:
        info = check_table_data(args.check)
        print(f"Table: {args.check}")
        print(f"Total records: {info.get('total_count', 0)}")
        print(f"Recent records: {info.get('recent_count', 0)}")
        print("Sample data:")
        for row in info.get('sample_data', []):
            print(f"  {row}")
    
    elif args.table:
        delete_specific_table(args.table)
    
    elif args.all:
        delete_historical_tables()
    
    else:
        # Default: Delete just the 2025 Q3 table
        print("Deleting synthetic data table: historical_data_3m_2025_q3")
        delete_specific_table('historical_data_3m_2025_q3')

if __name__ == "__main__":
    main()