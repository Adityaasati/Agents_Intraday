#!/usr/bin/env python3
"""
FILE: fix_database_columns.py
LOCATION: / (root directory)
PURPOSE: Simple fix for missing database columns

DESCRIPTION:
- Fixes missing close_price column in agent_technical_indicators
- Adds any other missing columns from schema
- Handles the error gracefully without complex recreation

USAGE:
- python fix_database_columns.py
"""

import os
import sys
from pathlib import Path
import psycopg2

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def fix_missing_columns():
    """Fix missing columns in agent_technical_indicators table"""
    
    connection_params = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5435)),
        'database': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD')
    }
    
    print("Fixing missing database columns...")
    
    try:
        with psycopg2.connect(**connection_params) as conn:
            with conn.cursor() as cursor:
                
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_technical_indicators'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    print("Table agent_technical_indicators does not exist")
                    return False
                
                # Get current columns
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'agent_technical_indicators'
                """)
                
                existing_columns = [row[0] for row in cursor.fetchall()]
                print(f"Current columns: {len(existing_columns)}")
                
                # Columns that might be missing
                columns_to_add = {
                    'close_price': 'DECIMAL(10,2)',
                    'open_price': 'DECIMAL(10,2)', 
                    'high_price': 'DECIMAL(10,2)',
                    'low_price': 'DECIMAL(10,2)',
                    'volume': 'BIGINT',
                    'rsi_21': 'DECIMAL(5,2)',
                    'sma_20': 'DECIMAL(10,2)',
                    'sma_50': 'DECIMAL(10,2)',
                    'ma_trend': 'VARCHAR(20)',
                    'macd_signal_line': 'DECIMAL(10,4)',
                    'macd_histogram': 'DECIMAL(10,4)',
                    'bb_upper': 'DECIMAL(10,2)',
                    'bb_middle': 'DECIMAL(10,2)',
                    'bb_lower': 'DECIMAL(10,2)',
                    'volume_ratio': 'DECIMAL(5,2)',
                    'atr_14': 'DECIMAL(10,2)',
                    'buy_signal': 'BOOLEAN DEFAULT false',
                    'sell_signal': 'BOOLEAN DEFAULT false'
                }
                
                # Add missing columns
                added_count = 0
                for column_name, column_type in columns_to_add.items():
                    if column_name not in existing_columns:
                        try:
                            alter_query = f"""
                            ALTER TABLE agent_technical_indicators 
                            ADD COLUMN {column_name} {column_type}
                            """
                            cursor.execute(alter_query)
                            print(f"✓ Added column: {column_name}")
                            added_count += 1
                        except Exception as e:
                            print(f"✗ Failed to add {column_name}: {e}")
                
                conn.commit()
                
                if added_count > 0:
                    print(f"\n✓ Successfully added {added_count} columns")
                else:
                    print("\n✓ No columns needed to be added")
                
                return True
                
    except Exception as e:
        print(f"✗ Error fixing columns: {e}")
        return False

def main():
    """Main function"""
    
    print("=" * 50)
    print("DATABASE COLUMN FIX")
    print("=" * 50)
    
    success = fix_missing_columns()
    
    if success:
        print("\n🎉 Database columns fixed successfully!")
        print("You can now run the system without column errors")
    else:
        print("\n⚠️ Column fix failed")
        print("Please check your database connection and permissions")
    
    print("=" * 50)

if __name__ == "__main__":
    main()