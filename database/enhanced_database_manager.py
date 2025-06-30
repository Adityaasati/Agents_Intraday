import os
import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool
import logging
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional, Tuple, Any

class EnhancedDatabaseManager:
    """Simplified Database Manager for Day 1 - Nexus Trading System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ist = pytz.timezone('Asia/Kolkata')
        self.connection_pool = None
        self._initialize_connection_pool()
        
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = SimpleConnectionPool(
                1, 5,  # Start simple: min=1, max=5 connections
                host=os.getenv('DATABASE_HOST', 'localhost'),
                port=int(os.getenv('DATABASE_PORT', 5432)),
                database=os.getenv('DATABASE_NAME'),
                user=os.getenv('DATABASE_USER'),
                password=os.getenv('DATABASE_PASSWORD')
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> List[Dict]:
        """Execute SQL query with connection pooling"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return [dict(row) for row in cursor.fetchall()]
                conn.commit()
                return []
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Query execution failed: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_symbols_from_categories(self, limit: int = 50, categories: List[str] = None, 
                                  market_cap_types: List[str] = None, 
                                  volatility_types: List[str] = None) -> List[Dict]:
        """Get symbols from stocks_categories_table with filtering"""
        
        where_conditions = ["symbol IS NOT NULL", "symbol != ''"]
        params = []
        
        if categories:
            where_conditions.append(f"category = ANY(%s)")
            params.append(categories)
        
        if market_cap_types:
            where_conditions.append(f"market_cap_type = ANY(%s)")
            params.append(market_cap_types)
            
        if volatility_types:
            where_conditions.append(f"volatility_category = ANY(%s)")
            params.append(volatility_types)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT symbol, stock_name, category, market_cap_type, sector, 
               volatility_category, current_price, market_cap,
               pe_ratio, pb_ratio, roe_ratio, roce_ratio,
               revenue_growth_ttm, profit_growth_ttm
        FROM stocks_categories_table 
        WHERE {where_clause}
        ORDER BY market_cap DESC NULLS LAST
        LIMIT %s
        """
        
        params.append(limit)
        return self.execute_query(query, tuple(params))
    
    def get_testing_symbols(self) -> List[str]:
        """Get predefined testing symbols that exist in your table"""
        testing_symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 
            'HDFCBANK', 'HINDUNILVR', 'ITC', 'BAJFINANCE', 'MARUTI'
        ]
        
        # Verify these symbols exist in your table
        placeholders = ','.join(['%s'] * len(testing_symbols))
        query = f"""
        SELECT symbol FROM stocks_categories_table 
        WHERE symbol IN ({placeholders})
        ORDER BY market_cap DESC NULLS LAST
        """
        
        result = self.execute_query(query, tuple(testing_symbols))
        found_symbols = [row['symbol'] for row in result]
        
        # If no predefined symbols found, get any 5 symbols
        if not found_symbols:
            query = """
            SELECT symbol FROM stocks_categories_table 
            WHERE symbol IS NOT NULL 
            ORDER BY market_cap DESC NULLS LAST
            LIMIT 5
            """
            result = self.execute_query(query)
            found_symbols = [row['symbol'] for row in result]
        
        return found_symbols
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """Get complete fundamental data for a symbol"""
        query = """
        SELECT * FROM stocks_categories_table 
        WHERE symbol = %s
        """
        
        result = self.execute_query(query, (symbol,))
        return result[0] if result else {}
    
    def get_available_quarters(self) -> List[str]:
        """Get list of available historical_data_3m_* tables"""
        query = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_name LIKE 'historical_data_3m_%'
        AND table_schema = 'public'
        ORDER BY table_name DESC
        """
        
        result = self.execute_query(query)
        return [row['table_name'] for row in result]
    
    def get_historical_data(self, symbol: str, start_date: datetime = None, 
                          end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Get historical OHLCV data from quarterly tables - simplified for Day 1"""
        
        if not end_date:
            end_date = datetime.now(self.ist)
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get available quarters
        quarters = self.get_available_quarters()
        
        if not quarters:
            self.logger.warning("No historical data tables found")
            return pd.DataFrame()
        
        all_data = []
        for quarter_table in quarters[:2]:  # Check only latest 2 quarters for Day 1
            try:
                query = f"""
                SELECT symbol, date, open, high, low, close, volume
                FROM {quarter_table}
                WHERE symbol = %s 
                AND date >= %s AND date <= %s
                ORDER BY date DESC
                LIMIT %s
                """
                
                data = self.execute_query(query, (symbol, start_date, end_date, limit))
                all_data.extend(data)
                
                if len(all_data) >= limit:  # Stop if we have enough data
                    break
                    
            except Exception as e:
                self.logger.warning(f"Could not query {quarter_table}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def store_technical_indicators(self, symbol: str, indicators_data: List[Dict]) -> bool:
        """Store basic technical indicators - corrected column names"""
        if not indicators_data:
            return False
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Corrected insert query to match actual table schema
                insert_query = """
                INSERT INTO agent_technical_indicators (
                    symbol, date, timeframe, close_price, 
                    rsi_14, ema_20, ema_50, technical_score,
                    created_at
                ) VALUES %s
                ON CONFLICT (symbol, date, timeframe) DO UPDATE SET
                    rsi_14 = EXCLUDED.rsi_14,
                    technical_score = EXCLUDED.technical_score,
                    created_at = CURRENT_TIMESTAMP
                """
                
                # Prepare simplified data tuples with proper column mapping
                data_tuples = []
                for data in indicators_data:
                    tuple_data = (
                        symbol, 
                        data.get('date'), 
                        data.get('timeframe', '5m'),
                        data.get('close', data.get('close_price', 0)),  # Handle both close and close_price
                        data.get('rsi_14'),
                        data.get('ema_20'),
                        data.get('ema_50'),
                        data.get('technical_score', 0.5),
                        datetime.now(self.ist)
                    )
                    data_tuples.append(tuple_data)
                
                execute_values(cursor, insert_query, data_tuples)
                conn.commit()
                return True
                
        except Exception as e:
            conn.rollback()
            # If close_price column doesn't exist, try without it
            if "close_price" in str(e):
                return self._store_technical_indicators_fallback(symbol, indicators_data)
            else:
                self.logger.error(f"Failed to store technical indicators: {e}")
                return False
        finally:
            self.return_connection(conn)
    
    def _store_technical_indicators_fallback(self, symbol: str, indicators_data: List[Dict]) -> bool:
        """Fallback storage without close_price column"""
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Simplified insert without close_price
                insert_query = """
                INSERT INTO agent_technical_indicators (
                    symbol, date, timeframe, 
                    rsi_14, ema_20, ema_50, technical_score,
                    created_at
                ) VALUES %s
                ON CONFLICT (symbol, date, timeframe) DO UPDATE SET
                    rsi_14 = EXCLUDED.rsi_14,
                    technical_score = EXCLUDED.technical_score,
                    created_at = CURRENT_TIMESTAMP
                """
                
                data_tuples = []
                for data in indicators_data:
                    tuple_data = (
                        symbol, 
                        data.get('date'), 
                        data.get('timeframe', '5m'),
                        data.get('rsi_14'),
                        data.get('ema_20'),
                        data.get('ema_50'),
                        data.get('technical_score', 0.5),
                        datetime.now(self.ist)
                    )
                    data_tuples.append(tuple_data)
                
                execute_values(cursor, insert_query, data_tuples)
                conn.commit()
                self.logger.info(f"Stored technical indicators using fallback method for {symbol}")
                return True
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Fallback storage also failed: {e}")
            return False
        finally:
            self.return_connection(conn)
    
    def store_live_signal(self, signal_data: Dict) -> bool:
        """Store generated trading signal - simplified for Day 1"""
        
        # Ensure required fields with defaults
        signal_data.setdefault('signal_uuid', f"{signal_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        signal_data.setdefault('status', 'ACTIVE')
        signal_data.setdefault('created_at', datetime.now(self.ist))
        
        insert_query = """
        INSERT INTO agent_live_signals (
            symbol, signal_uuid, signal_type, signal_time, entry_price,
            stop_loss, target_price, overall_confidence, 
            technical_score, fundamental_score, sentiment_score,
            status, primary_reasoning, created_at
        ) VALUES (
            %(symbol)s, %(signal_uuid)s, %(signal_type)s, %(signal_time)s, %(entry_price)s,
            %(stop_loss)s, %(target_price)s, %(overall_confidence)s,
            %(technical_score)s, %(fundamental_score)s, %(sentiment_score)s,
            %(status)s, %(primary_reasoning)s, %(created_at)s
        )
        """
        
        try:
            self.execute_query(insert_query, signal_data, fetch=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to store signal: {e}")
            return False
    
    def get_active_signals(self, limit: int = 50) -> List[Dict]:
        """Get active trading signals"""
        query = """
        SELECT * FROM agent_live_signals 
        WHERE status = 'ACTIVE' 
        ORDER BY overall_confidence DESC, signal_time DESC 
        LIMIT %s
        """
        
        return self.execute_query(query, (limit,))
    
    def get_system_health(self) -> Dict:
        """Get system health statistics"""
        health_queries = {
            'total_symbols': "SELECT COUNT(*) as count FROM stocks_categories_table",
            'available_quarters': "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name LIKE 'historical_data_3m_%'"
        }
        
        health_data = {}
        for key, query in health_queries.items():
            try:
                result = self.execute_query(query)
                health_data[key] = result[0]['count'] if result else 0
            except Exception as e:
                health_data[key] = f"Error: {e}"
        
        return health_data
    
    def create_quarterly_table(self, year: int, quarter: int) -> bool:
        """Create new quarterly historical data table"""
        table_name = f"historical_data_3m_{year}_q{quarter}"
        
        create_query = f"""
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
        """
        
        try:
            self.execute_query(create_query, fetch=False)
            self.logger.info(f"Created/verified table: {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create table {table_name}: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            result = self.execute_query("SELECT 1 as test")
            return len(result) > 0 and result[0]['test'] == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def close_connections(self):
        """Close all database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Database connections closed")