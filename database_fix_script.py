#!/usr/bin/env python3
"""
FILE: database_fix_script.py
LOCATION: / (root directory)
PURPOSE: Fix database column issues for agent_technical_indicators table

DESCRIPTION:
- Fixes the missing close_price column error
- Adds any missing columns to existing tables
- Provides option to recreate table with correct schema
- Handles database schema migration safely

USAGE:
- python database_fix_script.py --check    # Check table structure
- python database_fix_script.py --fix      # Fix missing columns
- python database_fix_script.py --recreate # Recreate table (WARNING: deletes data)
"""

import sys
import os
import argparse
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

class DatabaseColumnFixer:
    """Fix database column issues"""
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', 5432)),
            'database': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD')
        }
    
    def check_table_structure(self):
        """Check current table structure"""
        
        print("Checking agent_technical_indicators table structure...")
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'agent_technical_indicators'
                        )
                    """)
                    
                    table_exists = cursor.fetchone()['exists']
                    
                    if not table_exists:
                        print("❌ Table 'agent_technical_indicators' does not exist")
                        print("   Run: python main.py --mode test (to create tables)")
                        return False
                    
                    # Get current columns
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns 
                        WHERE table_name = 'agent_technical_indicators'
                        ORDER BY ordinal_position
                    """)
                    
                    current_columns = cursor.fetchall()
                    
                    print(f"✅ Table exists with {len(current_columns)} columns:")
                    for col in current_columns:
                        print(f"   - {col['column_name']} ({col['data_type']})")
                    
                    # Check for required columns
                    required_columns = [
                        'close_price', 'open_price', 'high_price', 'low_price', 'volume',
                        'rsi_14', 'ema_20', 'ema_50', 'technical_score'
                    ]
                    
                    existing_column_names = [col['column_name'] for col in current_columns]
                    missing_columns = [col for col in required_columns if col not in existing_column_names]
                    
                    if missing_columns:
                        print(f"\n❌ Missing columns: {missing_columns}")
                        return False
                    else:
                        print("\n✅ All required columns are present")
                        return True
                        
        except Exception as e:
            print(f"❌ Error checking table structure: {e}")
            return False
    
    def fix_missing_columns(self):
        """Add missing columns to existing table"""
        
        print("Fixing missing columns in agent_technical_indicators table...")
        
        # Define missing columns with their types
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
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    
                    # Get existing columns
                    cursor.execute("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = 'agent_technical_indicators'
                    """)
                    
                    existing_columns = [row[0] for row in cursor.fetchall()]
                    
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
                                print(f"✅ Added column: {column_name}")
                                added_count += 1
                            except Exception as e:
                                print(f"❌ Failed to add column {column_name}: {e}")
                    
                    conn.commit()
                    
                    if added_count > 0:
                        print(f"\n✅ Successfully added {added_count} columns")
                    else:
                        print("\n✅ No columns needed to be added")
                    
                    return True
                    
        except Exception as e:
            print(f"❌ Error fixing columns: {e}")
            return False
    
    def recreate_table(self):
        """Recreate table with correct schema (WARNING: deletes existing data)"""
        
        print("⚠️  WARNING: This will delete all existing data in agent_technical_indicators table")
        response = input("Are you sure you want to continue? (type 'yes' to confirm): ")
        
        if response.lower() != 'yes':
            print("Operation cancelled")
            return False
        
        print("Recreating agent_technical_indicators table...")
        
        # Correct schema
        create_table_query = """
        DROP TABLE IF EXISTS agent_technical_indicators CASCADE;
        
        CREATE TABLE agent_technical_indicators (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            timeframe VARCHAR(10) NOT NULL DEFAULT '5m',
            close_price DECIMAL(10,2),
            open_price DECIMAL(10,2),
            high_price DECIMAL(10,2),
            low_price DECIMAL(10,2),
            volume BIGINT,
            rsi_14 DECIMAL(5,2),
            rsi_21 DECIMAL(5,2),
            rsi_signal VARCHAR(20),
            ema_20 DECIMAL(10,2),
            ema_50 DECIMAL(10,2),
            sma_20 DECIMAL(10,2),
            sma_50 DECIMAL(10,2),
            ma_trend VARCHAR(20),
            macd_line DECIMAL(10,4),
            macd_signal_line DECIMAL(10,4),
            macd_histogram DECIMAL(10,4),
            bb_upper DECIMAL(10,2),
            bb_middle DECIMAL(10,2),
            bb_lower DECIMAL(10,2),
            volume_ratio DECIMAL(5,2),
            atr_14 DECIMAL(10,2),
            technical_score DECIMAL(3,2),
            buy_signal BOOLEAN DEFAULT false,
            sell_signal BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date, timeframe)
        );
        
        CREATE INDEX idx_technical_indicators_symbol_date ON agent_technical_indicators(symbol, date);
        CREATE INDEX idx_technical_indicators_signals ON agent_technical_indicators(buy_signal, sell_signal);
        """
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                    conn.commit()
                    
            print("✅ Table recreated successfully with correct schema")
            return True
            
        except Exception as e:
            print(f"❌ Error recreating table: {e}")
            return False

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Fix database column issues')
    parser.add_argument('--check', action='store_true', help='Check table structure')
    parser.add_argument('--fix', action='store_true', help='Fix missing columns')
    parser.add_argument('--recreate', action='store_true', help='Recreate table (deletes data)')
    
    args = parser.parse_args()
    
    if not any([args.check, args.fix, args.recreate]):
        parser.print_help()
        return
    
    fixer = DatabaseColumnFixer()
    
    if args.check:
        fixer.check_table_structure()
    
    elif args.fix:
        if fixer.check_table_structure():
            print("Table structure is already correct")
        else:
            fixer.fix_missing_columns()
            print("\nRechecking table structure...")
            fixer.check_table_structure()
    
    elif args.recreate:
        fixer.recreate_table()
        print("\nChecking new table structure...")
        fixer.check_table_structure()

if __name__ == "__main__":
    main()