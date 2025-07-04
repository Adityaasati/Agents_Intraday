import os
import pandas as pd
import psycopg2
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import configparser
import warnings
import time
import threading
import signal
import sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit
import logging

warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')
load_dotenv()

def get_scheduler_logger(name='SchedulerLog'):
    """Get a logger that writes to both console and scheduler.log"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('scheduler.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

class HistoricalDataDownloader:
    def __init__(self):
        # Use your existing database config names
        self.db_config = {
            'host': os.getenv('DATABASE_HOST') or os.getenv('DB_HOST'),
            'port': os.getenv('DATABASE_PORT') or os.getenv('DB_PORT'),
            'database': os.getenv('DATABASE_NAME') or os.getenv('DB_NAME'),
            'user': os.getenv('DATABASE_USER') or os.getenv('DB_USER'),
            'password': os.getenv('DATABASE_PASSWORD') or os.getenv('DB_PASSWORD')
        }
        self.load_config()
        
    def load_config(self):
        """Load all config from config.ini and .env"""
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
        else:
            # Create default config with all settings
            config['market'] = {
                'start_time': '09:15',
                'end_time': '15:30',
                'expected_records_5min': '75'
            }
            config['scheduler'] = {
                'default_interval': '5min',
                'max_workers': '3',
                'retry_attempts': '3',
                'rate_limit_seconds': '1.0'
            }
            config['data'] = {
                'start_date': '2020-01-01',
                'default_symbols': 'all',
                'interval': '5minute'
            }
            with open('config.ini', 'w') as f:
                config.write(f)
            logger = get_scheduler_logger('ConfigSetup')
            logger.info("Created config.ini with defaults")
            
        # Load all settings
        self.market_start = config.get('market', 'start_time', fallback='09:15')
        self.market_end = config.get('market', 'end_time', fallback='15:30')
        self.expected_records = int(config.get('market', 'expected_records_5min', fallback='75'))
        self.max_workers = int(config.get('scheduler', 'max_workers', fallback='3'))
        self.retry_attempts = int(config.get('scheduler', 'retry_attempts', fallback='3'))
        self.rate_limit = float(config.get('scheduler', 'rate_limit_seconds', fallback='1.0'))
        # Clean date string by removing any comments
        start_date_str = config.get('data', 'start_date', fallback='2020-01-01').split('#')[0].strip()
        self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        self.default_symbols = config.get('data', 'default_symbols', fallback='all')
        self.interval = config.get('data', 'interval', fallback='5minute')

    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)

    def is_market_hours(self):
        now = datetime.now()
        if now.weekday() >= 5: return False
        market_end = datetime.strptime(f"{now.date()} {self.market_end}", "%Y-%m-%d %H:%M")
        return now <= market_end

    def get_symbols(self, number_of_symbols='all'):
        with self.get_db_connection() as conn:
            query = "SELECT symbol FROM stocks_categories_table"
            if number_of_symbols != 'all':
                query += f" LIMIT {number_of_symbols}"
            return pd.read_sql(query, conn)['symbol'].tolist()
    
    def get_instrument_tokens(self, kite, symbols):
        """Convert trading symbols to instrument tokens - optimized"""
        logger = get_scheduler_logger('TokenMapper')
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
                    
            logger.info(f"Mapped {len(symbol_to_token)}/{len(symbols)} symbols to tokens")
            return symbol_to_token
        except Exception as e:
            logger.error(f"Token mapping failed: {e}")
            return {}

    def get_quarters(self, start_date):
        """Generate quarter tables - optimized"""
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

    def table_exists_and_current(self, table_name):
        """Check if table is up to date - optimized"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (table_name,))
            if not cursor.fetchone()[0]: return False
            
            now = datetime.now()
            if f"{now.year}_q{(now.month-1)//3+1}" in table_name:
                if now.weekday() >= 5 or now < datetime.strptime(f"{now.date()} {self.market_end}", "%Y-%m-%d %H:%M"):
                    return now.weekday() >= 5  # True if weekend, False if market open
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'RELIANCE' AND date::date = %s", (now.date(),))
                return cursor.fetchone()[0] >= self.expected_records * 0.9  # 90% threshold
            return True

    def create_table(self, table_name):
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
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
            logger = get_scheduler_logger('TableCreation')
            logger.info(f"Created/verified table: {table_name}")

    def download_symbol_data(self, kite, symbol, instrument_token, start_date, end_date, interval, table_name):
        """Download data for single symbol - optimized with better error handling"""
        logger = get_scheduler_logger('SymbolDownload')
        try:
            chunks = [(start_date + timedelta(days=i*60), min(start_date + timedelta(days=(i+1)*60-1), end_date)) 
                     for i in range((end_date - start_date).days // 60 + 1)]
            all_data = []
            
            for chunk_start, chunk_end in chunks:
                for retry in range(self.retry_attempts):
                    try:
                        data = kite.historical_data(instrument_token, chunk_start, chunk_end, interval)
                        if data: all_data.extend(data)
                        break
                    except Exception as e:
                        if retry == self.retry_attempts - 1:
                            logger.warning(f"{symbol}: Failed after {self.retry_attempts} retries")
                        else:
                            time.sleep(2 ** retry)
                time.sleep(self.rate_limit)
            
            if all_data:
                self.insert_data(table_name, symbol, all_data)
                logger.info(f"{symbol}: {len(all_data)} records")
            
        except Exception as e:
            logger.error(f"{symbol}: {str(e)}")
    
    def insert_data(self, table_name, symbol, data):
        """Insert data with market hours filtering - date is stock timestamp, created_at is insertion time"""
        if not data: return
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            market_start_time = dt_time(*map(int, self.market_start.split(':')))
            market_end_time = dt_time(*map(int, self.market_end.split(':')))
            
            # Filter market hours only
            filtered_data = [record for record in data 
                           if market_start_time <= record['date'].time() <= market_end_time]
            
            if not filtered_data: return
            
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

    def get_latest_data_for_symbol(self, symbol):
        """Get latest 5-minute data for a single symbol - for live updates"""
        try:
            from kite_token_generator import get_authenticated_kite_client
            kite = get_authenticated_kite_client()
            if not kite:
                return None
                
            # Get instrument token
            symbol_to_token = self.get_instrument_tokens(kite, [symbol])
            if symbol not in symbol_to_token:
                return None
                
            token = symbol_to_token[symbol]
            
            # Get last 2 bars to ensure we have latest
            end_date = datetime.now()
            start_date = end_date - timedelta(minutes=15)
            
            data = kite.historical_data(token, start_date, end_date, "5minute")
            
            if data:
                latest = data[-1]  # Get most recent bar
                return {
                    'date': latest['date'],  # This is the stock's actual timestamp from market
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': int(latest['volume'])
                    # created_at will be added when inserting to database
                }
        except Exception as e:
            logger = get_scheduler_logger('LiveData')
            logger.error(f"Failed to get latest data for {symbol}: {e}")
            
        return None

class DataScheduler:
    def __init__(self, run_interval='5min'):
        self.scheduler = BackgroundScheduler()
        self.run_interval = run_interval
        self.errors = []
        self.runs_completed = 0
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                          handlers=[logging.FileHandler('scheduler.log'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        
    def get_kite_client(self):
        try:
            from kite_token_generator import get_authenticated_kite_client
            return get_authenticated_kite_client()
        except Exception as e:
            self.logger.error(f"Kite auth failed: {e}")
            return None
    
    def is_trading_time(self):
        """Check if current time is within trading hours"""
        now = datetime.now()
        return (now.weekday() < 5 and 
                now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30))
    
    def scheduled_download(self):
        try:
            # Check download frequency from config
            try:
                import config
                download_freq = getattr(config, 'DOWNLOAD_FREQUENCY', 'once')
            except ImportError:
                download_freq = 'once'
            
            self.logger.info(f"Download #{self.runs_completed + 1} (frequency: {download_freq})")
            
            kite = self.get_kite_client()
            if not kite:
                raise Exception("Kite authentication failed")
            
            # For periodic updates, only update current quarter
            if download_freq != 'once' and self.runs_completed > 0:
                # Update only current quarter for live updates
                downloader = HistoricalDataDownloader()
                symbols = downloader.get_symbols(50)  # Limit symbols for live updates
                self._update_current_quarter_only(kite, symbols)
            else:
                # Full historical download
                download_historical_data(kite, run_at='once')
            
            self.runs_completed += 1
            self.logger.info(f"Download #{self.runs_completed} completed")
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            self.errors.append(f"{datetime.now()}: {e}")
    
    def _update_current_quarter_only(self, kite, symbols):
        """Update only current quarter table for live updates"""
        downloader = HistoricalDataDownloader()
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        table_name = f"historical_data_3m_{now.year}_q{quarter}"
        
        # Ensure table exists
        downloader.create_table(table_name)
        
        # Get tokens
        symbol_to_token = downloader.get_instrument_tokens(kite, symbols)
        
        # Update last 2 days of data for current quarter
        end_date = now
        start_date = now - timedelta(days=2)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for symbol, token in symbol_to_token.items():
                future = executor.submit(
                    downloader.download_symbol_data, 
                    kite, symbol, token, start_date, end_date, 
                    downloader.interval, table_name
                )
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
    
    def start(self):
        if self.run_interval == 'once':
            self.scheduled_download()
            return
        
        # Get download frequency from config
        try:
            import config
            download_freq = getattr(config, 'DOWNLOAD_FREQUENCY', '5min')
            if download_freq == 'once':
                self.scheduled_download()
                return
            else:
                self.run_interval = download_freq
        except ImportError:
            pass
            
        # Parse interval and create trigger for market hours only
        minutes = int(self.run_interval[:-3]) if self.run_interval.endswith('min') else 5
        
        # Market hours only: 9:15 AM to 3:30 PM, Monday to Friday
        trigger = CronTrigger(
            day_of_week='mon-fri',
            hour='9-15',
            minute=f'15-30/{minutes}'  # Start at 9:15, every X minutes until 3:30
        )
        
        self.scheduler.add_job(self.scheduled_download, trigger, max_instances=1, coalesce=True)
        self.scheduler.start()
        
        self.logger.info(f"Scheduler started: every {self.run_interval} during market hours")
        self.logger.info(f"Market hours: 9:15 AM - 3:30 PM (Mon-Fri)")
        self.logger.info(f"Monitor logs: tail -f scheduler.log")
        atexit.register(lambda: self.scheduler.shutdown())
        
        # Keep alive with status updates
        signal.signal(signal.SIGINT, lambda s, f: (self.scheduler.shutdown(), sys.exit(0)))
        
        last_status = datetime.now()
        while True: 
            time.sleep(60)
            # Print status every 30 minutes
            if datetime.now() - last_status >= timedelta(minutes=30):
                self.logger.info(f"Status: {self.runs_completed} runs completed, {len(self.errors)} errors")
                last_status = datetime.now()

def download_historical_data(kite, number_of_symbols=None, start=None, interval=None, max_workers=None, run_at='once'):
    """Main download function - optimized for speed and efficiency"""
    downloader = HistoricalDataDownloader()
    logger = get_scheduler_logger('DataDownload')
    
    # Use config defaults if not provided
    number_of_symbols = number_of_symbols or downloader.default_symbols
    start = start or downloader.start_date  # Use config start_date
    interval = interval or downloader.interval
    max_workers = max_workers or downloader.max_workers
    
    if run_at != 'once':
        return DataScheduler(run_at).start()
    
    if not isinstance(start, datetime):
        start = datetime.strptime(start, '%Y-%m-%d')
        
    logger.info(f"Starting data download at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get symbols and tokens
    symbols = downloader.get_symbols(number_of_symbols)
    symbol_to_token = downloader.get_instrument_tokens(kite, symbols)
    valid_symbols = list(symbol_to_token.items())
    quarters = downloader.get_quarters(start)
    
    logger.info(f"Processing {len(valid_symbols)} symbols across {len(quarters)} quarters")
    
    # Process each quarter with optimized threading
    for table_name, quarter_start, quarter_end in quarters:
        logger.info(f"{table_name} ({quarter_start.date()} to {quarter_end.date()})")
        
        if downloader.table_exists_and_current(table_name):
            logger.info("Current, skipping...")
            continue
            
        downloader.create_table(table_name)
        
        # Optimized parallel processing with progress tracking
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(downloader.download_symbol_data, kite, symbol, token, 
                                     quarter_start, quarter_end, interval, table_name): symbol 
                      for symbol, token in valid_symbols}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0:  # Progress every 50 symbols
                    logger.info(f"Progress: {completed}/{len(valid_symbols)} symbols")
                future.result()  # Ensure exceptions are raised
                
        logger.info(f"{table_name} completed")
    
    logger.info(f"Download completed for {len(valid_symbols)} symbols!")

# Integration functions for your existing system
def get_real_market_data(symbol):
    """Get real market data for a symbol - integrates with your data_updater.py"""
    downloader = HistoricalDataDownloader()
    return downloader.get_latest_data_for_symbol(symbol)

def start_live_data_updates():
    """Start live data updates - integrates with your main.py"""
    try:
        import config
        download_freq = getattr(config, 'DOWNLOAD_FREQUENCY', '5min')
    except ImportError:
        download_freq = '5min'
    
    if download_freq == 'once':
        # One-time historical download
        from kite_token_generator import get_authenticated_kite_client
        kite = get_authenticated_kite_client()
        if kite:
            download_historical_data(kite)
        return True
    else:
        # Start periodic updates
        scheduler = DataScheduler(download_freq)
        scheduler.start()
        return scheduler

# Scheduler functions - minimal
def start_background_scheduler(run_interval='5min'):
    DataScheduler(run_interval).start()

def start_scheduler_thread(run_interval='5min'):
    scheduler = DataScheduler(run_interval)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return scheduler

def check_scheduler_status():
    """Check if scheduler is running and show recent logs"""
    try:
        if os.path.exists('scheduler.log'):
            print("Recent scheduler logs:")
            with open('scheduler.log', 'r') as f:
                lines = f.readlines()
                for line in lines[-15:]:  # Last 15 lines
                    print(f"  {line.strip()}")
        else:
            print("No scheduler.log found - scheduler may not be running")
            
        # Also check if file is being written to recently
        if os.path.exists('scheduler.log'):
            import os.path
            mod_time = os.path.getmtime('scheduler.log')
            last_modified = datetime.fromtimestamp(mod_time)
            now = datetime.now()
            time_diff = now - last_modified
            
            if time_diff.total_seconds() < 300:  # Less than 5 minutes
                print(f"Log file updated {int(time_diff.total_seconds())} seconds ago - scheduler appears active")
            else:
                print(f"Log file last updated {time_diff} ago - scheduler may be idle")
                
    except Exception as e:
        print(f"Error reading logs: {e}")

if __name__ == "__main__":
    print("Import this module and call download_historical_data(kite, run_at='5min')")
    # start_background_scheduler('5min')  # Uncomment to test