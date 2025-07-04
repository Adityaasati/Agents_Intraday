import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import List, Dict, Optional

class SimpleDataUpdater:
    """Simplified Data Updater for Day 1 - Nexus Trading System"""
    
    def __init__(self, db_manager=None):
        self.logger = logging.getLogger(__name__)
        self.ist = pytz.timezone('Asia/Kolkata')
        self.db_manager = db_manager
        
        # Simple market hours
        self.market_start_hour = 9
        self.market_end_hour = 15
    
    def is_market_hours(self) -> bool:
        """Simple check if current time is within market hours"""
        now = datetime.now(self.ist)
        return self.market_start_hour <= now.hour <= self.market_end_hour
    
    def get_current_quarter_table(self) -> str:
        """Get current quarter table name"""
        now = datetime.now(self.ist)
        quarter = ((now.month - 1) // 3) + 1
        return f"historical_data_3m_{now.year}_q{quarter}"
    
    def ensure_current_quarter_table_exists(self) -> bool:
        """Ensure current quarter table exists"""
        if not self.db_manager:
            return False
            
        now = datetime.now(self.ist)
        quarter = ((now.month - 1) // 3) + 1
        
        return self.db_manager.create_quarterly_historical_table(now.year, quarter)
    
    def validate_ohlcv_data(self, ohlcv_data: Dict) -> bool:
        """Validate OHLCV data structure and values"""
        
        required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required fields
        for field in required_fields:
            if field not in ohlcv_data:
                self.logger.warning(f"Missing field: {field}")
                return False
        
        try:
            # Validate data types and basic ranges
            open_price = float(ohlcv_data['open'])
            high_price = float(ohlcv_data['high'])
            low_price = float(ohlcv_data['low'])
            close_price = float(ohlcv_data['close'])
            volume = int(ohlcv_data['volume'])
            
            # Basic OHLC validation
            if high_price < max(open_price, close_price, low_price):
                self.logger.warning("High price is less than other prices")
                return False
                
            if low_price > min(open_price, close_price, high_price):
                self.logger.warning("Low price is greater than other prices")
                return False
                
            if volume < 0:
                self.logger.warning("Negative volume")
                return False
                
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                self.logger.warning("Zero or negative prices found")
                return False
                
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Data type validation failed: {e}")
            return False
    
    def update_symbol_data(self, symbol: str, ohlcv_data: Dict) -> bool:
        """Update single symbol data in current quarter table"""
        
        if not self.db_manager:
            self.logger.error("Database manager not initialized")
            return False
            
        if not self.validate_ohlcv_data(ohlcv_data):
            self.logger.error(f"Invalid OHLCV data for {symbol}")
            return False
        
        try:
            table_name = self.get_current_quarter_table()
            
            # Ensure table exists
            if not self.ensure_current_quarter_table_exists():
                self.logger.error(f"Failed to create/verify table: {table_name}")
                return False
            
            # Simple insert/update
            insert_query = f"""
            INSERT INTO {table_name} (symbol, date, open, high, low, close, volume, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                created_at = EXCLUDED.created_at
            """
            
            data_tuple = (
                symbol,
                ohlcv_data['date'],
                float(ohlcv_data['open']),
                float(ohlcv_data['high']),
                float(ohlcv_data['low']),
                float(ohlcv_data['close']),
                int(ohlcv_data['volume']),
                datetime.now(self.ist)
            )
            
            self.db_manager.execute_query(insert_query, data_tuple, fetch=False)
            self.logger.debug(f"Updated data for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update data for {symbol}: {e}")
            return False
    
    def simulate_sample_data(self, symbol: str) -> Optional[Dict]:
        """Generate sample OHLCV data for testing Day 1 setup
        
        This is just for testing the data pipeline.
        Replace with real data fetching in later days.
        """
        
        try:
            # Get last known price if available
            last_data = self.get_last_known_data(symbol)
            
            if last_data:
                base_price = float(last_data['close'])
            else:
                # Default base prices for common stocks
                base_prices = {
                    'RELIANCE': 2500.0,
                    'TCS': 3500.0,
                    'INFY': 1600.0,
                    'HDFC': 2800.0,
                    'ICICIBANK': 900.0,
                    'HDFCBANK': 1650.0
                }
                base_price = base_prices.get(symbol, 1000.0)
            
            # Generate simple price movement for testing
            import random
            random.seed(int(datetime.now().timestamp()) % 1000)  # Some randomness but reproducible
            
            # Small price movement for realistic testing
            price_change = random.uniform(-0.01, 0.01)  # -1% to +1%
            close_price = base_price * (1 + price_change)
            
            # Generate OHLC around close
            price_range = abs(price_change) * base_price * 0.5
            high_price = close_price + random.uniform(0, price_range)
            low_price = close_price - random.uniform(0, price_range)
            open_price = random.uniform(low_price, high_price)
            
            # Ensure OHLC relationships are correct
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate reasonable volume
            base_volume = random.randint(100000, 1000000)
            
            return {
                'date': datetime.now(self.ist),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': base_volume
            }
            
        except Exception as e:
            self.logger.error(f"Failed to simulate data for {symbol}: {e}")
            return None
    
    def get_last_known_data(self, symbol: str) -> Optional[Dict]:
        """Get last known data for a symbol"""
        
        if not self.db_manager:
            return None
            
        try:
            table_name = self.get_current_quarter_table()
            
            query = f"""
            SELECT * FROM {table_name}
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT 1
            """
            
            result = self.db_manager.execute_query(query, (symbol,))
            return result[0] if result else None
            
        except Exception as e:
            self.logger.debug(f"Could not get last known data for {symbol}: {e}")
            return None
    
    def test_data_pipeline(self, symbols: List[str] = None) -> Dict:
        """Test the data pipeline with sample data"""
        
        if not symbols:
            symbols = ['RELIANCE', 'TCS']  # Just test with 2 symbols
        
        results = {
            'total_symbols': len(symbols),
            'successful_updates': 0,
            'failed_updates': 0,
            'errors': []
        }
        
        for symbol in symbols:
            try:
                # Generate sample data
                sample_data = self.simulate_sample_data(symbol)
                
                if sample_data and self.update_symbol_data(symbol, sample_data):
                    results['successful_updates'] += 1
                    self.logger.info(f"Successfully updated sample data for {symbol}")
                else:
                    results['failed_updates'] += 1
                    results['errors'].append(f"Failed to update {symbol}")
                    
            except Exception as e:
                results['failed_updates'] += 1
                results['errors'].append(f"Error with {symbol}: {str(e)}")
                self.logger.error(f"Error testing data pipeline for {symbol}: {e}")
        
        return results
    
    def get_update_status(self) -> Dict:
        """Get current update status"""
        
        try:
            table_name = self.get_current_quarter_table()
            
            status = {
                'current_table': table_name,
                'is_market_hours': self.is_market_hours(),
                'table_exists': False,
                'recent_records': 0
            }
            
            if self.db_manager:
                # Check if table exists
                query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
                """
                result = self.db_manager.execute_query(query, (table_name,))
                status['table_exists'] = result[0]['exists'] if result else False
                
                # Get recent record count
                if status['table_exists']:
                    query = f"""
                    SELECT COUNT(*) as count FROM {table_name}
                    WHERE DATE(created_at) = CURRENT_DATE
                    """
                    result = self.db_manager.execute_query(query)
                    status['recent_records'] = result[0]['count'] if result else 0
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get update status: {e}")
            return {'error': str(e)}

# Simple helper functions for Day 1
def test_data_updater_with_db(db_manager):
    """Test data updater with database manager"""
    updater = SimpleDataUpdater(db_manager)
    return updater.test_data_pipeline()

def manual_update_symbol(db_manager, symbol: str, ohlcv_data: Dict) -> bool:
    """Manually update data for a single symbol"""
    updater = SimpleDataUpdater(db_manager)
    return updater.update_symbol_data(symbol, ohlcv_data)