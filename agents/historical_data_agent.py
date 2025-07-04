from .base_agent import BaseAgent
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

class HistoricalDataAgent(BaseAgent):
    def __init__(self, db_manager):
        super().__init__(db_manager)
        self.load_config()
        
    def load_config(self):
        """Load config from config.py"""
        import config
        self.download_frequency = getattr(config, 'DOWNLOAD_FREQUENCY', 'once')
        self.interval = getattr(config, 'HISTORICAL_DATA_INTERVAL', '5minute')
        self.start_date = datetime.strptime(getattr(config, 'DOWNLOAD_START_DATE', '2020-01-01'), '%Y-%m-%d')
        self.end_date = datetime.now() if getattr(config, 'DOWNLOAD_END_DATE', 'today') == 'today' else datetime.strptime(config.DOWNLOAD_END_DATE, '%Y-%m-%d')
        self.market_start = getattr(config, 'MARKET_START_TIME', '09:15')
        self.market_end = getattr(config, 'MARKET_END_TIME', '15:30')
        self.expected_records = getattr(config, 'EXPECTED_RECORDS_5MIN', 75)
        self.max_workers = 3
        self.retry_attempts = 3
        self.rate_limit = 1.0
        
    def is_data_current(self):
        """Check if historical data is current - simplified"""
        try:
            now = datetime.now()
            
            # Get last market day
            last_market_day = self.get_last_market_day(now)
            market_end_time = datetime.combine(last_market_day.date(), 
                                            dt_time(*map(int, self.market_end.split(':'))))
            
            # Check current quarter table
            quarter = (now.month - 1) // 3 + 1
            table_name = f"historical_data_3m_{now.year}_q{quarter}"
            
            conn = self.db_manager.get_connection()
            try:
                with conn.cursor() as cursor:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (SELECT FROM information_schema.tables 
                        WHERE table_name = %s)
                    """, (table_name,))
                    
                    if not cursor.fetchone()[0]:
                        self.logger.info(f"Table {table_name} doesn't exist, download needed")
                        return False
                    
                    # Get latest data timestamp for any symbol
                    cursor.execute(f"""
                        SELECT MAX(date) FROM {table_name} 
                        WHERE date::date = %s
                    """, (last_market_day.date(),))
                    
                    result = cursor.fetchone()
                    latest_data_time = result[0] if result[0] else None
                    
                    if not latest_data_time:
                        self.logger.info(f"No data for last market day {last_market_day.date()}")
                        return False
                    
                    # Check if latest data is close to market end (within 30 mins)
                    time_diff = abs((market_end_time - latest_data_time).total_seconds())
                    is_current = time_diff <= 1800  # 30 minutes tolerance
                    
                    self.logger.info(f"Data currency check: Latest={latest_data_time}, Market End={market_end_time}, Current={is_current}")
                    return is_current
                    
            finally:
                self.db_manager.return_connection(conn)
                    
        except Exception as e:
            self.logger.error(f"Data currency check failed: {e}")
            return False
    
    
    def get_last_market_day(self, current_date):
        """Get last market day (excluding weekends)"""
        date = current_date
        while date.weekday() >= 5:  # Skip weekends
            date -= timedelta(days=1)
        return date
    
    def run_download(self):
        """Main download execution"""
        try:
            # Check if data is already current
            if self.download_frequency != 'once' and self.is_data_current():
                self.logger.info("Historical data is current, skipping download")
                return True
            
            # Get kite client
            kite = self.get_kite_client()
            if not kite:
                self.logger.error("Kite authentication failed")
                return False
            
            # Download historical data
            success = self.download_historical_data(kite)
            
            if success:
                self.logger.info("Historical data download completed successfully")
            return success
            
        except Exception as e:
            self.logger.error(f"Download execution failed: {e}")
            return False
    
    def get_kite_client(self):
        """Get authenticated kite client"""
        try:
            from kite_token_generator import get_authenticated_kite_client
            return get_authenticated_kite_client()
        except Exception as e:
            self.logger.error(f"Kite authentication failed: {e}")
            return None
    
    def get_symbols(self, number_of_symbols='all'):
        """Get symbols from stocks_categories_table"""
        # conn = self.db_manager._get_pool_connection() if hasattr(self.db_manager, '_get_pool_connection') else self.db_manager.get_connection()
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                query = "SELECT symbol FROM stocks_categories_table"
                if number_of_symbols != 'all':
                    query += f" LIMIT {number_of_symbols}"
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall()]
        finally:
            # if hasattr(self.db_manager, '_put_pool_connection'):
            #     self.db_manager._put_pool_connection(conn)
            # elif hasattr(self.db_manager, 'return_connection'):
            #     self.db_manager.return_connection(conn)
            # else:
            #     conn.close()
            self.db_manager.return_connection(conn)
    
    def get_instrument_tokens(self, kite, symbols):
        """Convert trading symbols to instrument tokens"""
        try:
            # Get all instruments once
            nse_instruments = pd.DataFrame(kite.instruments("NSE"))
            bse_instruments = pd.DataFrame(kite.instruments("BSE"))
            equity_instruments = pd.concat([nse_instruments, bse_instruments], ignore_index=True)
            equity_instruments = equity_instruments[equity_instruments['instrument_type'] == 'EQ'].copy()
            
            symbol_to_token = {}
            for symbol in symbols:
                # Exact match first, then partial
                match = equity_instruments[equity_instruments['tradingsymbol'] == symbol]
                if match.empty:
                    match = equity_instruments[equity_instruments['tradingsymbol'].str.contains(symbol, case=False, na=False)]
                
                if not match.empty:
                    symbol_to_token[symbol] = match.iloc[0]['instrument_token']
                    
            self.logger.info(f"Mapped {len(symbol_to_token)}/{len(symbols)} symbols to tokens")
            return symbol_to_token
        except Exception as e:
            self.logger.error(f"Token mapping failed: {e}")
            return {}
    
    def get_quarters(self, start_date):
        """Generate quarter tables"""
        quarters, current_date = [], start_date
        now = datetime.now()
        current_quarter_end = datetime(now.year, ((now.month - 1) // 3 + 1) * 3 + 1, 1) - timedelta(seconds=1)
        
        while current_date <= current_quarter_end:
            year, quarter = current_date.year, (current_date.month - 1) // 3 + 1
            quarter_start = datetime(year, (quarter-1)*3 + 1, 1)
            quarter_end = datetime(year, 12, 31, 23, 59, 59) if quarter == 4 else datetime(year, quarter*3 + 1, 1) - timedelta(seconds=1)
            quarters.append((f"historical_data_3m_{year}_q{quarter}", quarter_start, quarter_end))
            current_date = quarter_end + timedelta(days=1)
        return quarters
    
    def create_table(self, table_name):
        """Create quarterly historical table with correct DB pattern"""
        # conn = self.db_manager._get_pool_connection() if hasattr(self.db_manager, '_get_pool_connection') else self.db_manager.get_connection()
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        open DECIMAL(10,2) NOT NULL,
                        high DECIMAL(10,2) NOT NULL,
                        low DECIMAL(10,2) NOT NULL,
                        close DECIMAL(10,2) NOT NULL,
                        volume BIGINT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_date ON {table_name}(symbol, date);
                """)
                conn.commit()
                self.logger.info(f"Created/verified table: {table_name}")
        finally:
            # if hasattr(self.db_manager, '_put_pool_connection'):
            #     self.db_manager._put_pool_connection(conn)
            # elif hasattr(self.db_manager, 'return_connection'):
            #     self.db_manager.return_connection(conn)
            # else:
            #     conn.close()
            self.db_manager.return_connection(conn)
    
    def download_historical_data(self, kite):
        """Main download function following your pattern"""
        # Get symbols and tokens
        symbols = self.get_symbols('all')  # or limit for testing
        symbol_to_token = self.get_instrument_tokens(kite, symbols)
        valid_symbols = list(symbol_to_token.items())
        quarters = self.get_quarters(self.start_date)
        
        self.logger.info(f"Processing {len(valid_symbols)} symbols across {len(quarters)} quarters")
        
        # Process each quarter
        for table_name, quarter_start, quarter_end in quarters:
            self.logger.info(f"{table_name} ({quarter_start.date()} to {quarter_end.date()})")
            
            if self.table_exists_and_current(table_name):
                self.logger.info("Current, skipping...")
                continue
                
            self.create_table(table_name)
            
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.download_symbol_data, kite, symbol, token, 
                                         quarter_start, quarter_end, self.interval, table_name): symbol 
                          for symbol, token in valid_symbols}
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 50 == 0:
                        self.logger.info(f"Progress: {completed}/{len(valid_symbols)} symbols")
                    future.result()
                    
            self.logger.info(f"{table_name} completed")
        
        self.logger.info(f"Download completed for {len(valid_symbols)} symbols!")
        return True
    
    def table_exists_and_current(self, table_name):
        """Check if table is up to date"""
        # conn = self.db_manager._get_pool_connection() if hasattr(self.db_manager, '_get_pool_connection') else self.db_manager.get_connection()
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (table_name,))
                if not cursor.fetchone()[0]: 
                    return False
                
                now = datetime.now()
                if f"{now.year}_q{(now.month-1)//3+1}" in table_name:
                    if now.weekday() >= 5 or now < datetime.strptime(f"{now.date()} {self.market_end}", "%Y-%m-%d %H:%M"):
                        return now.weekday() >= 5  # True if weekend, False if market open
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'RELIANCE' AND date::date = %s", (now.date(),))
                    return cursor.fetchone()[0] >= self.expected_records * 0.9  # 90% threshold
                return True
        finally:
            # if hasattr(self.db_manager, '_put_pool_connection'):
            #     self.db_manager._put_pool_connection(conn)
            # elif hasattr(self.db_manager, 'return_connection'):
            #     self.db_manager.return_connection(conn)
            # else:
            #     conn.close()
            self.db_manager.return_connection(conn)
    
    def download_symbol_data(self, kite, symbol, instrument_token, start_date, end_date, interval, table_name):
        """Download data for single symbol"""
        try:
            chunks = [(start_date + timedelta(days=i*60), min(start_date + timedelta(days=(i+1)*60-1), end_date)) 
                     for i in range((end_date - start_date).days // 60 + 1)]
            all_data = []
            
            for chunk_start, chunk_end in chunks:
                for retry in range(self.retry_attempts):
                    try:
                        data = kite.historical_data(instrument_token, chunk_start, chunk_end, interval)
                        if data: 
                            all_data.extend(data)
                        break
                    except Exception as e:
                        if retry == self.retry_attempts - 1:
                            self.logger.warning(f"{symbol}: Failed after {self.retry_attempts} retries")
                        else:
                            time.sleep(2 ** retry)
                time.sleep(self.rate_limit)
            
            if all_data:
                self.insert_data(table_name, symbol, all_data)
                self.logger.info(f"{symbol}: {len(all_data)} records")
            
        except Exception as e:
            self.logger.error(f"{symbol}: {str(e)}")
    
    def insert_data(self, table_name, symbol, data):
        """Insert data with market hours filtering - correct DB pattern"""
        if not data: 
            return
        
        # conn = self.db_manager._get_pool_connection() if hasattr(self.db_manager, '_get_pool_connection') else self.db_manager.get_connection()
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                market_start_time = dt_time(*map(int, self.market_start.split(':')))
                market_end_time = dt_time(*map(int, self.market_end.split(':')))
                
                # Filter market hours only
                filtered_data = [record for record in data 
                               if market_start_time <= record['date'].time() <= market_end_time]
                
                if not filtered_data: 
                    return
                
                # Batch delete and insert
                first_date, last_date = filtered_data[0]['date'], filtered_data[-1]['date']
                cursor.execute(f"DELETE FROM {table_name} WHERE symbol = %s AND date BETWEEN %s AND %s", 
                             (symbol, first_date, last_date))
                
                # Get current timestamp for created_at
                current_time = datetime.now()
                
                # Batch insert - date is stock's actual timestamp, created_at is when we insert to DB
                cursor.executemany(f"""
                    INSERT INTO {table_name} (symbol, date, open, high, low, close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume, created_at = EXCLUDED.created_at
                """, [(symbol, r['date'], r['open'], r['high'], r['low'], r['close'], r['volume'], current_time) 
                      for r in filtered_data])
                conn.commit()
        finally:
            # if hasattr(self.db_manager, '_put_pool_connection'):
            #     self.db_manager._put_pool_connection(conn)
            # elif hasattr(self.db_manager, 'return_connection'):
            #     self.db_manager.return_connection(conn)
            # else:
            #     conn.close()
            self.db_manager.return_connection(conn)