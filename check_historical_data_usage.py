#!/usr/bin/env python3
"""
FILE: check_historical_data_usage.py
PURPOSE: Show how historical data tables are used in paper trading

DESCRIPTION:
- Demonstrates data flow from historical tables to trading decisions
- Shows which tables are used for current prices
- Explains fallback mechanisms

USAGE:
- python check_historical_data_usage.py --symbol RELIANCE
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def show_data_sources_for_symbol(symbol: str):
    """Show how data flows for a specific symbol"""
    
    print(f"Historical Data Usage for {symbol}")
    print("=" * 50)
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from datetime import datetime
        
        db_manager = EnhancedDatabaseManager()
        
        # 1. Check stocks_categories_table (fundamental data)
        print("1. Fundamental Data Source:")
        symbol_data = db_manager.get_symbol_data(symbol)
        if symbol_data:
            print(f"   ✓ stocks_categories_table has {symbol}")
            print(f"     Category: {symbol_data.get('category', 'N/A')}")
            print(f"     Sector: {symbol_data.get('sector', 'N/A')}")
            print(f"     Current Price: ₹{symbol_data.get('current_price', 0):,.2f}")
            print(f"     Market Cap: {symbol_data.get('market_cap_type', 'N/A')}")
        else:
            print(f"   ✗ {symbol} not found in stocks_categories_table")
            return False
        
        # 2. Check historical data tables (for technical analysis)
        print("\n2. Historical Price Data Sources:")
        
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        current_table = f"historical_data_3m_{now.year}_q{quarter}"
        
        # Check current quarter
        historical_data = db_manager.get_historical_data(symbol, limit=5)
        if not historical_data.empty:
            latest_row = historical_data.iloc[0]
            print(f"   ✓ {current_table} has recent data for {symbol}")
            print(f"     Latest Date: {latest_row['date']}")
            print(f"     Latest Close: ₹{latest_row['close']:,.2f}")
            print(f"     Volume: {latest_row['volume']:,}")
            print(f"     Available Records: {len(historical_data)}")
        else:
            print(f"   ⚠ {current_table} has no data for {symbol}")
        
        # 3. Show how technical analysis uses this data
        print("\n3. Technical Analysis Data Usage:")
        
        if not historical_data.empty:
            print("   Historical data used for:")
            print("   • Current price determination")
            print("   • RSI calculation (14-period)")
            print("   • MACD calculation (12,26,9)")
            print("   • Bollinger Bands (20-period)")
            print("   • Moving averages (5, 10, 20, 50)")
            print("   • Support/Resistance levels")
            
            # Show data range
            oldest_date = historical_data.iloc[-1]['date']
            latest_date = historical_data.iloc[0]['date']
            print(f"   Data Range: {oldest_date} to {latest_date}")
            print(f"   Days of Data: {len(historical_data)}")
        else:
            print("   ⚠ Insufficient data for technical analysis")
        
        # 4. Show paper trading price source
        print("\n4. Paper Trading Price Source:")
        print("   Paper trading gets current price from:")
        print("   1. Latest close price from historical_data_3m tables")
        print("   2. Fallback to stocks_categories_table.current_price")
        print("   3. Simulated price movement during trading hours")
        
        # Demonstrate current price lookup
        from agents.portfolio_agent import PortfolioAgent
        portfolio_agent = PortfolioAgent(db_manager)
        
        try:
            current_price = portfolio_agent._get_current_price(symbol)
            if current_price:
                print(f"   Current Price for Trading: ₹{current_price:,.2f}")
                
                # Compare with fundamental data
                fundamental_price = symbol_data.get('current_price', 0)
                if fundamental_price:
                    diff = abs(current_price - fundamental_price)
                    print(f"   Fundamental Price: ₹{fundamental_price:,.2f}")
                    print(f"   Price Difference: ₹{diff:,.2f}")
            else:
                print("   ✗ Could not determine current price")
        except Exception as e:
            print(f"   ✗ Price lookup failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error checking data sources: {e}")
        return False

def show_overall_data_flow():
    """Show the complete data flow in paper trading"""
    
    print("\nPaper Trading Data Flow")
    print("=" * 50)
    
    print("1. Signal Generation:")
    print("   stocks_categories_table → Fundamental Analysis")
    print("   historical_data_3m_* → Technical Analysis")
    print("   Combined scores → Trading Signal")
    
    print("\n2. Position Entry:")
    print("   Latest historical_data_3m close price → Entry Price")
    print("   Apply slippage simulation → Final Entry Price")
    print("   Store in agent_portfolio_positions table")
    
    print("\n3. Position Monitoring:")
    print("   Continuous price updates from historical data")
    print("   Calculate unrealized P&L")
    print("   Check stop-loss and target levels")
    
    print("\n4. Position Exit:")
    print("   Current price from historical data → Exit Price")
    print("   Calculate realized P&L")
    print("   Update position status to CLOSED")
    
    print("\n5. Performance Tracking:")
    print("   agent_portfolio_positions → Portfolio Summary")
    print("   agent_live_signals → Signal Performance")
    print("   Combined metrics → Trading Reports")

def check_data_availability():
    """Check overall data availability for paper trading"""
    
    print("\nData Availability Check")
    print("=" * 50)
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from datetime import datetime
        import psycopg2
        
        db_manager = EnhancedDatabaseManager()
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check fundamental data
                cursor.execute("SELECT COUNT(*) FROM stocks_categories_table")
                stock_count = cursor.fetchone()[0]
                print(f"Fundamental Data: {stock_count} stocks in stocks_categories_table")
                
                # Check historical tables
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_name LIKE 'historical_data_3m_%'
                    ORDER BY table_name DESC
                """)
                
                tables = cursor.fetchall()
                print(f"Historical Tables: {len(tables)} quarterly tables found")
                
                # Check data in latest table
                if tables:
                    latest_table = tables[0][0]
                    cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {latest_table}")
                    symbol_count = cursor.fetchone()[0]
                    
                    cursor.execute(f"SELECT MAX(date) FROM {latest_table}")
                    latest_date = cursor.fetchone()[0]
                    
                    print(f"Latest Table: {latest_table}")
                    print(f"  - {symbol_count} symbols")
                    print(f"  - Latest data: {latest_date}")
                    
                    # Sample some data
                    cursor.execute(f"""
                        SELECT symbol, close, volume FROM {latest_table}
                        WHERE date = (SELECT MAX(date) FROM {latest_table})
                        LIMIT 5
                    """)
                    
                    sample_data = cursor.fetchall()
                    print(f"  Sample prices:")
                    for symbol, close, volume in sample_data:
                        print(f"    {symbol}: ₹{close:,.2f} (Vol: {volume:,})")
        
        return True
        
    except Exception as e:
        print(f"Data availability check failed: {e}")
        return False

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Historical Data Usage')
    parser.add_argument('--symbol', default='RELIANCE', help='Symbol to analyze')
    parser.add_argument('--overview', action='store_true', help='Show data flow overview')
    
    args = parser.parse_args()
    
    if args.overview:
        show_overall_data_flow()
        check_data_availability()
    else:
        show_data_sources_for_symbol(args.symbol)
        show_overall_data_flow()

if __name__ == "__main__":
    main()