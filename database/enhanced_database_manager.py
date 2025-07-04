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
from pathlib import Path
from psycopg2 import pool
import threading
import time
import config
import json
from .connection_config import get_connection_params

class EnhancedDatabaseManager:
    """Simplified Database Manager for Day 1 - Nexus Trading System"""
    
    def __del__(self):
        """Cleanup connections on object destruction"""
        try:
            if hasattr(self, 'connection_pool') and self.connection_pool:
                self.connection_pool.closeall()
        except:
            pass
    

    def close_connections(self):
        """Manually close all connections"""
        try:
            if hasattr(self, 'connection_pool') and self.connection_pool:
                self.connection_pool.closeall()
                self.logger.info("All database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    
    # In __init__ method:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': os.getenv('DATABASE_PORT', '5432'),
            'database': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD')
        }
        
        self.logger = logging.getLogger(__name__)
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Use the ORIGINAL attribute name
        self.connection_pool = None  # ← NOT _connection_pool
        self._initialize_connection_pool()
        
        # Keep the rest as is
        self._query_cache = {} if hasattr(config, 'ENABLE_QUERY_CACHE') and config.ENABLE_QUERY_CACHE else None
        self._indicator_cache = {} if hasattr(config, 'ENABLE_INDICATOR_CACHE') and config.ENABLE_INDICATOR_CACHE else None
        self._cache_lock = threading.Lock()
        self._performance_stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'avg_query_time': 0.0,
            'last_cleanup': datetime.now()
        }
    
        
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = SimpleConnectionPool(
                1, 5,  # Start simple: min=1, max=5 connections
                host=os.getenv('DATABASE_HOST', 'localhost'),
                port=int(os.getenv('DATABASE_PORT', 5435)),
                database=os.getenv('DATABASE_NAME'),
                user=os.getenv('DATABASE_USER'),
                password=os.getenv('DATABASE_PASSWORD')
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def _get_current_quarter_table(self) -> str:
        """Get current quarter table name"""
        try:
            import config
            return f"{config.HISTORICAL_DATA_PREFIX}{config.get_current_quarter()}"
        except:
            # Fallback to manual calculation
            from datetime import datetime
            now = datetime.now()
            quarter = ((now.month - 1) // 3) + 1
            return f"historical_data_3m_{now.year}_q{quarter}"
    
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
    
    def get_testing_symbols(self, limit: int = 50) -> List[str]:
        """Get testing symbols with optional limit"""
        try:
            conn = self.get_connection()
            
            query = """
            SELECT symbol FROM stocks_categories_table 
            WHERE category IN ('A', 'B') 
            AND market_cap_type IN ('Large_Cap', 'Mid_Cap')
            ORDER BY market_cap DESC 
            LIMIT %s
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (limit,))
                symbols = [row[0] for row in cursor.fetchall()]
                
            # If no symbols found, return config defaults
            if not symbols:
                import config
                return config.TESTING_SYMBOLS[:limit]
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Testing symbols retrieval failed: {e}")
            # Fallback to config symbols
            import config
            return config.TESTING_SYMBOLS[:limit]
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """Get complete fundamental data for a symbol"""
        try:
            query = """
            SELECT * FROM stocks_categories_table 
            WHERE symbol = %s
            """
            
            result = self.execute_query(query, (symbol,))
            
            # Ensure we always return a dict
            if result and isinstance(result, list) and len(result) > 0:
                return dict(result[0]) if result[0] else {}
            else:
                self.logger.warning(f"No fundamental data found for {symbol}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return {}
    
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
    
    # Update the get_historical_data method to auto-create current quarter table

    def get_historical_data(self, symbol: str, *args) -> pd.DataFrame:
        """Get historical data with flexible parameters"""
        
        # Parameter handling (as before)
        if len(args) == 0:
            limit = 100
        elif len(args) == 1:
            limit = args[0]
        elif len(args) == 2:
            start_date, end_date = args
            days_diff = (end_date - start_date).days if hasattr(args[0], 'days') else 30
            limit = days_diff * 24 * 12
        else:
            limit = 100
        
        try:
            quarter_table = self._get_current_quarter_table()
            
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM {quarter_table}
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT %s
            """
            
            # Fix SQLAlchemy warning by using connection string
            conn_string = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}"
            
            df = pd.read_sql_query(query, conn_string, params=(symbol, limit))
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Historical data query failed: {e}")
            return pd.DataFrame()
    
    def _get_previous_quarter_table(self) -> str:
        """Get previous quarter table name"""
        try:
            from datetime import datetime
            now = datetime.now()
            
            # Calculate previous quarter
            if now.month <= 3:
                # Q1 -> previous year Q4
                prev_year = now.year - 1
                prev_quarter = 4
            else:
                # Q2,Q3,Q4 -> same year previous quarter
                prev_year = now.year
                prev_quarter = ((now.month - 1) // 3)
            
            return f"historical_data_3m_{prev_year}_q{prev_quarter}"
            
        except:
            # Fallback to current quarter
            return self._get_current_quarter_table()
    
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
    

    def create_quarterly_historical_table(self, year: int, quarter: int):
        """Create quarterly historical data table"""
        table_name = f"historical_data_3m_{year}_q{quarter}"
        
        try:
            with self.get_connection() as conn:
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
                        
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_date 
                        ON {table_name}(symbol, date);
                    """)
                    
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
    def close(self):
        """Close all database connections"""
        try:
            if hasattr(self, '_pool') and self._pool:
                self._pool.closeall()
                self.logger.info("Connection pool closed")
        except Exception as e:
            self.logger.error(f"Error closing connection pool: {e}")
    
    # Add these methods to the existing EnhancedDatabaseManager class

    def store_sentiment_data(self, sentiment_data: Dict) -> bool:
        """Store sentiment analysis results - Day 3A implementation"""
        
        try:
            # Simple storage in agent_system_config for Day 3A
            # Will be enhanced with dedicated table in Day 3B
            
            config_key = f"sentiment_{sentiment_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H')}"
            config_value = json.dumps({
                'sentiment_score': sentiment_data['sentiment_score'],
                'articles_count': sentiment_data['articles_count'],
                'analysis_time': sentiment_data['analysis_time'].isoformat(),
                'raw_results': sentiment_data.get('raw_results', [])
            })
            
            insert_query = """
            INSERT INTO agent_system_config 
            (config_key, config_value, config_type, description, category)
            VALUES (%s, %s, 'json', 'Sentiment analysis data', 'sentiment')
            ON CONFLICT (config_key) DO UPDATE SET
                config_value = EXCLUDED.config_value,
                updated_at = CURRENT_TIMESTAMP
            """
            
            self.execute_query(insert_query, (config_key, config_value), fetch=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store sentiment data: {e}")
            return False

    def get_recent_sentiment(self, symbol: str, hours_back: int = 24) -> Optional[Dict]:
        """Get recent sentiment data for symbol"""
        
        try:
            cutoff_time = datetime.now(self.ist) - timedelta(hours=hours_back)
            
            query = """
            SELECT config_value FROM agent_system_config 
            WHERE config_key LIKE %s 
            AND category = 'sentiment'
            AND updated_at > %s
            ORDER BY updated_at DESC
            LIMIT 1
            """
            
            result = self.execute_query(query, (f"sentiment_{symbol}_%", cutoff_time))
            
            if result:
                import json
                return json.loads(result[0]['config_value'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get sentiment data for {symbol}: {e}")
            return None
        

    def store_enhanced_sentiment_analysis(self, analysis_data: Dict) -> bool:
        """Store enhanced sentiment analysis in dedicated table"""
        
        try:
            insert_query = """
            INSERT INTO agent_sentiment_analysis (
                symbol, analysis_time, sentiment_score, confidence, impact_score,
                articles_analyzed, primary_event_type, momentum_score, 
                trend_direction, data_source
            ) VALUES (
                %(symbol)s, %(analysis_time)s, %(sentiment_score)s, %(confidence)s, 
                %(impact_score)s, %(articles_analyzed)s, %(primary_event_type)s,
                %(momentum_score)s, %(trend_direction)s, %(data_source)s
            )
            ON CONFLICT (symbol, analysis_time) DO UPDATE SET
                sentiment_score = EXCLUDED.sentiment_score,
                confidence = EXCLUDED.confidence,
                impact_score = EXCLUDED.impact_score,
                articles_analyzed = EXCLUDED.articles_analyzed,
                primary_event_type = EXCLUDED.primary_event_type,
                momentum_score = EXCLUDED.momentum_score,
                trend_direction = EXCLUDED.trend_direction
            """
            
            self.execute_query(insert_query, analysis_data, fetch=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced sentiment: {e}")
            return False

    def get_sentiment_history(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """Get sentiment history for momentum analysis"""
        
        try:
            cutoff_time = datetime.now(self.ist) - timedelta(days=days_back)
            
            query = """
            SELECT sentiment_score, confidence, momentum_score, trend_direction, analysis_time
            FROM agent_sentiment_analysis 
            WHERE symbol = %s AND analysis_time > %s
            ORDER BY analysis_time ASC
            """
            
            return self.execute_query(query, (symbol, cutoff_time))
            
        except Exception as e:
            self.logger.error(f"Failed to get sentiment history for {symbol}: {e}")
            return []

    def get_latest_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get latest sentiment for symbol"""
        
        try:
            query = """
            SELECT sentiment_score, confidence, momentum_score, trend_direction, 
                primary_event_type, analysis_time
            FROM agent_sentiment_analysis 
            WHERE symbol = %s 
            ORDER BY analysis_time DESC 
            LIMIT 1
            """
            
            result = self.execute_query(query, (symbol,))
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest sentiment for {symbol}: {e}")
            return None

    def store_market_sentiment(self, market_data: Dict) -> bool:
        """Store market-wide sentiment analysis"""
        
        try:
            insert_query = """
            INSERT INTO agent_market_sentiment (
                analysis_time, market_sentiment_score, articles_analyzed, 
                primary_themes, data_sources
            ) VALUES (
                %(analysis_time)s, %(market_sentiment_score)s, %(articles_analyzed)s,
                %(primary_themes)s, %(data_sources)s
            )
            ON CONFLICT (analysis_time) DO UPDATE SET
                market_sentiment_score = EXCLUDED.market_sentiment_score,
                articles_analyzed = EXCLUDED.articles_analyzed,
                primary_themes = EXCLUDED.primary_themes,
                data_sources = EXCLUDED.data_sources
            """
            
            self.execute_query(insert_query, market_data, fetch=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store market sentiment: {e}")
            return False





    def create_sentiment_tables(self) -> bool:
        """Create sentiment analysis tables for Day 3B - CORRECTED"""
        
        try:
            # Correct schema without foreign key reference issues
            schema_sql = """
            CREATE TABLE IF NOT EXISTS agent_sentiment_analysis (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                analysis_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                sentiment_score DECIMAL(3,2) NOT NULL,
                confidence DECIMAL(3,2),
                impact_score DECIMAL(3,2),
                articles_analyzed INTEGER DEFAULT 0,
                primary_event_type VARCHAR(20),
                momentum_score DECIMAL(4,3),
                trend_direction VARCHAR(10),
                data_source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, analysis_time)
            );
            
            CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_symbol_time ON agent_sentiment_analysis(symbol, analysis_time);
            
            CREATE TABLE IF NOT EXISTS agent_market_sentiment (
                id SERIAL PRIMARY KEY,
                analysis_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                market_sentiment_score DECIMAL(3,2) NOT NULL,
                articles_analyzed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(analysis_time)
            );
            """
            
            self.execute_query(schema_sql, fetch=False)
            self.logger.info("Sentiment tables created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create sentiment tables: {e}")
            return False
    
    # Add these methods to existing database/enhanced_database_manager.py file
    # REPLACE the database methods in database/enhanced_database_manager.py with these corrected versions
    # These use the existing connection pool instead of connection_params

    def store_portfolio_monitoring(self, monitoring_data: Dict) -> bool:
        """Store portfolio risk monitoring results"""
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                
                # Create monitoring table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_portfolio_monitoring (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        portfolio_health VARCHAR(20),
                        total_risk_percent DECIMAL(5,2),
                        concentration_risk DECIMAL(5,2),
                        correlation_risk VARCHAR(20),
                        alerts_count INTEGER,
                        alerts TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert monitoring data
                cursor.execute("""
                    INSERT INTO agent_portfolio_monitoring 
                    (timestamp, portfolio_health, total_risk_percent, concentration_risk, 
                    correlation_risk, alerts_count, alerts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    monitoring_data.get('timestamp'),
                    monitoring_data.get('portfolio_health'),
                    monitoring_data.get('total_risk_percent'),
                    monitoring_data.get('concentration_risk'),
                    monitoring_data.get('correlation_risk'),
                    monitoring_data.get('alerts_count'),
                    monitoring_data.get('alerts')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to store portfolio monitoring: {e}")
            return False
        finally:
            self.return_connection(conn)

    def store_correlation_data(self, symbol1: str, symbol2: str, correlation: float, 
                            analysis_date: datetime = None) -> bool:
        """Store symbol correlation data"""
        
        if analysis_date is None:
            analysis_date = datetime.now()
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                
                # Create correlation table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_correlation_data (
                        id SERIAL PRIMARY KEY,
                        symbol1 VARCHAR(20) NOT NULL,
                        symbol2 VARCHAR(20) NOT NULL,
                        correlation DECIMAL(6,4),
                        analysis_date TIMESTAMP WITHOUT TIME ZONE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol1, symbol2, analysis_date)
                    )
                """)
                
                # Insert correlation data
                cursor.execute("""
                    INSERT INTO agent_correlation_data 
                    (symbol1, symbol2, correlation, analysis_date)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (symbol1, symbol2, analysis_date) 
                    DO UPDATE SET correlation = %s
                """, (symbol1, symbol2, correlation, analysis_date, correlation))
                
                conn.commit()
                return True
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to store correlation data: {e}")
            return False
        finally:
            self.return_connection(conn)

    def get_portfolio_monitoring_history(self, days_back: int = 7) -> List[Dict]:
        """Get portfolio monitoring history"""
        
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                cursor.execute("""
                    SELECT * FROM agent_portfolio_monitoring 
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY timestamp DESC
                    LIMIT 50
                """, (days_back,))
                
                results = cursor.fetchall()
                return [dict(row) for row in results] if results else []
                
        except Exception as e:
            self.logger.error(f"Failed to get monitoring history: {e}")
            return []
        finally:
            self.return_connection(conn)

    def get_correlation_matrix(self, symbols: List[str], days_back: int = 1) -> Dict:
        """Get correlation matrix for symbols"""
        
        if len(symbols) < 2:
            return {}
        
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                placeholders = ','.join(['%s'] * len(symbols))
                cursor.execute(f"""
                    SELECT symbol1, symbol2, correlation 
                    FROM agent_correlation_data 
                    WHERE (symbol1 IN ({placeholders}) OR symbol2 IN ({placeholders}))
                    AND analysis_date >= NOW() - INTERVAL '%s days'
                    ORDER BY analysis_date DESC
                """, symbols + symbols + [days_back])
                
                results = cursor.fetchall()
                
                # Build correlation matrix
                correlation_matrix = {}
                for row in results:
                    pair = f"{row['symbol1']}-{row['symbol2']}"
                    if pair not in correlation_matrix:  # Keep most recent
                        correlation_matrix[pair] = float(row['correlation'])
                
                return correlation_matrix
                
        except Exception as e:
            self.logger.error(f"Failed to get correlation matrix: {e}")
            return {}
        finally:
            self.return_connection(conn)

    def get_portfolio_risk_summary(self) -> Dict:
        """Get current portfolio risk summary"""
        
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                # Get latest monitoring record
                cursor.execute("""
                    SELECT * FROM agent_portfolio_monitoring 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                latest = cursor.fetchone()
                
                if not latest:
                    return {'status': 'no_data'}
                
                # Get active positions count
                cursor.execute("""
                    SELECT COUNT(*) as active_positions 
                    FROM agent_live_signals 
                    WHERE signal_status = 'ACTIVE'
                """)
                
                positions_result = cursor.fetchone()
                active_positions = positions_result['active_positions'] if positions_result else 0
                
                return {
                    'timestamp': latest['timestamp'],
                    'portfolio_health': latest['portfolio_health'],
                    'total_risk_percent': float(latest['total_risk_percent']) if latest['total_risk_percent'] else 0,
                    'concentration_risk': float(latest['concentration_risk']) if latest['concentration_risk'] else 0,
                    'correlation_risk': latest['correlation_risk'],
                    'active_positions': active_positions,
                    'alerts_count': latest['alerts_count'],
                    'status': 'current'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get portfolio risk summary: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            self.return_connection(conn)

    def clean_old_monitoring_data(self, days_to_keep: int = 30) -> bool:
        """Clean old monitoring and correlation data"""
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                
                # Clean old monitoring data
                cursor.execute("""
                    DELETE FROM agent_portfolio_monitoring 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                """, (days_to_keep,))
                
                monitoring_deleted = cursor.rowcount
                
                # Clean old correlation data (if table exists)
                try:
                    cursor.execute("""
                        DELETE FROM agent_correlation_data 
                        WHERE created_at < NOW() - INTERVAL '%s days'
                    """, (days_to_keep,))
                    correlation_deleted = cursor.rowcount
                except:
                    correlation_deleted = 0
                
                conn.commit()
                
                self.logger.info(f"Cleaned {monitoring_deleted} monitoring records, {correlation_deleted} correlation records")
                return True
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to clean old monitoring data: {e}")
            return False
        finally:
            self.return_connection(conn)

    def update_position_risk_metrics(self, signal_id: int, risk_metrics: Dict) -> bool:
        """Update risk metrics for existing position"""
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                
                # Check if columns exist, add them if needed
                try:
                    cursor.execute("""
                        ALTER TABLE agent_live_signals 
                        ADD COLUMN IF NOT EXISTS correlation_risk VARCHAR(20),
                        ADD COLUMN IF NOT EXISTS portfolio_beta DECIMAL(4,2),
                        ADD COLUMN IF NOT EXISTS risk_updated_at TIMESTAMP
                    """)
                except:
                    pass  # Columns might already exist
                
                cursor.execute("""
                    UPDATE agent_live_signals 
                    SET correlation_risk = %s,
                        portfolio_beta = %s,
                        risk_updated_at = %s
                    WHERE id = %s
                """, (
                    risk_metrics.get('correlation_risk'),
                    risk_metrics.get('portfolio_beta'),
                    datetime.now(),
                    signal_id
                ))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to update position risk metrics: {e}")
            return False
        finally:
            self.return_connection(conn)

    def get_sector_allocation_summary(self) -> Dict:
        """Get current sector allocation summary"""
        
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                
                # Check if we have any active signals first
                cursor.execute("""
                    SELECT COUNT(*) as signal_count FROM agent_live_signals 
                    WHERE signal_status = 'ACTIVE'
                """)
                
                count_result = cursor.fetchone()
                if not count_result or count_result['signal_count'] == 0:
                    return {
                        'sectors': {},
                        'total_allocation': 0,
                        'sector_count': 0,
                        'max_sector_percent': 0
                    }
                
                # Try to get sector allocation with JOIN
                try:
                    cursor.execute("""
                        SELECT 
                            COALESCE(sct.sector, 'Unknown') as sector,
                            COUNT(als.id) as position_count,
                            SUM(COALESCE(als.recommended_position_size, 0)) as total_allocation,
                            AVG(COALESCE(als.overall_confidence, 0)) as avg_confidence
                        FROM agent_live_signals als
                        LEFT JOIN stocks_categories_table sct ON als.symbol = sct.symbol
                        WHERE als.signal_status = 'ACTIVE'
                        GROUP BY COALESCE(sct.sector, 'Unknown')
                        ORDER BY total_allocation DESC
                    """)
                    
                    results = cursor.fetchall()
                    
                except Exception as join_error:
                    # Fallback: just get signals without sector info
                    self.logger.warning(f"Sector JOIN failed, using fallback: {join_error}")
                    cursor.execute("""
                        SELECT 
                            'Unknown' as sector,
                            COUNT(id) as position_count,
                            SUM(COALESCE(recommended_position_size, 0)) as total_allocation,
                            AVG(COALESCE(overall_confidence, 0)) as avg_confidence
                        FROM agent_live_signals
                        WHERE signal_status = 'ACTIVE'
                    """)
                    
                    results = cursor.fetchall()
                
                sector_summary = {}
                total_allocation = 0
                
                for row in results:
                    allocation = float(row['total_allocation']) if row['total_allocation'] else 0
                    total_allocation += allocation
                    
                    sector_summary[row['sector']] = {
                        'position_count': row['position_count'],
                        'total_allocation': allocation,
                        'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0
                    }
                
                # Calculate percentages
                for sector_data in sector_summary.values():
                    sector_data['allocation_percent'] = round(
                        (sector_data['total_allocation'] / total_allocation * 100), 2
                    ) if total_allocation > 0 else 0
                
                return {
                    'sectors': sector_summary,
                    'total_allocation': round(total_allocation, 2),
                    'sector_count': len(sector_summary),
                    'max_sector_percent': max(
                        (data['allocation_percent'] for data in sector_summary.values()), default=0
                    )
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get sector allocation summary: {e}")
            return {'sectors': {}, 'total_allocation': 0, 'sector_count': 0, 'max_sector_percent': 0}
        finally:
            self.return_connection(conn)
    
    
    def _init_connection_pool(self):
        """Initialize connection pool for better performance"""
        try:
            self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=config.DB_CONNECTION_POOL_MAX,
                host=os.getenv('DATABASE_HOST', 'localhost'),
                port=os.getenv('DATABASE_PORT', '5435'),
                database=os.getenv('DATABASE_NAME', 'nexus_trading'),
                user=os.getenv('DATABASE_USER'),
                password=os.getenv('DATABASE_PASSWORD'),
                connect_timeout=config.DB_CONNECTION_TIMEOUT
            )
            self.logger.info(f"Connection pool initialized: {config.DB_CONNECTION_POOL_MAX} connections")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            # Fall back to regular connection
            pass

    def _get_pool_connection(self):
        """Get connection from pool if available, else regular connection"""
        if self._connection_pool:
            try:
                return self._connection_pool.getconn()
            except:
                pass
        return self.get_connection()

    def _put_pool_connection(self, conn):
        """Return connection to pool if using pool"""
        if self._connection_pool:
            try:
                self._connection_pool.putconn(conn)
                return
            except:
                pass
        if conn:
            conn.close()

    def _create_performance_indexes(self):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_agent_signals_symbol ON agent_live_signals (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_agent_signals_date ON agent_live_signals (signal_time)",
            "CREATE INDEX IF NOT EXISTS idx_stocks_categories_symbol ON stocks_categories_table (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol ON agent_technical_indicators (symbol, date)"
        ]
        
        conn = self._get_pool_connection()
        try:
            with conn.cursor() as cursor:
                for index_sql in indexes:
                    try:
                        cursor.execute(index_sql)
                    except:
                        pass  # Index might already exist
                conn.commit()
        except Exception as e:
            self.logger.warning(f"Index creation warning: {e}")
        finally:
            self._put_pool_connection(conn)

    def get_historical_data_optimized(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Optimized historical data retrieval with caching"""
        cache_key = f"hist_{symbol}_{limit}"
        
        # Check cache first
        if self._indicator_cache and cache_key in self._indicator_cache:
            cache_entry = self._indicator_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < config.CACHE_EXPIRY_MINUTES * 60:
                self._performance_stats['cache_hits'] += 1
                return cache_entry['data']
        
        start_time = time.time()
        
        # Use existing get_historical_data method instead of custom query
        try:
            df = self.get_historical_data(symbol, limit)
            
            # Cache result
            if self._indicator_cache and not df.empty:
                with self._cache_lock:
                    self._indicator_cache[cache_key] = {
                        'data': df.copy(),
                        'timestamp': datetime.now()
                    }
            
            query_time = time.time() - start_time
            self._update_performance_stats(query_time)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized data retrieval failed for {symbol}: {e}")
            return pd.DataFrame()
    


    def get_multiple_symbols_data(self, symbols: List[str], limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Batch retrieval for multiple symbols"""
        if not symbols:
            return {}
        
        start_time = time.time()
        results = {}
        
        # Use individual retrieval with optimized caching instead of complex batch query
        try:
            for symbol in symbols:
                df = self.get_historical_data_optimized(symbol, limit)
                results[symbol] = df
            
            query_time = time.time() - start_time
            self.logger.info(f"Batch retrieved {len(symbols)} symbols in {query_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Batch retrieval failed: {e}")
            # Fallback to individual retrieval
            for symbol in symbols:
                try:
                    results[symbol] = self.get_historical_data(symbol, limit)
                except:
                    results[symbol] = pd.DataFrame()
        
        return results

    def store_multiple_signals(self, signals_batch: List[Dict]) -> bool:
        """Bulk insert for better performance"""
        if not signals_batch:
            return True
        
        conn = self._get_pool_connection()
        try:
            with conn.cursor() as cursor:
                values = []
                for signal in signals_batch:
                    values.append((
                        signal.get('symbol'),
                        signal.get('signal_type'),
                        signal.get('confidence_score', 0),
                        signal.get('entry_price', 0),
                        signal.get('stop_loss', 0),
                        signal.get('target_price', 0),
                        signal.get('position_size', 0),
                        signal.get('reasoning', ''),
                        datetime.now()
                    ))
                
                cursor.executemany("""
                    INSERT INTO agent_live_signals 
                    (symbol, signal_type, confidence_score, entry_price, stop_loss, 
                     target_price, position_size, reasoning, signal_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, values)
                
                conn.commit()
                self.logger.info(f"Bulk stored {len(signals_batch)} signals")
                return True
                
        except Exception as e:
            self.logger.error(f"Bulk signal storage failed: {e}")
            return False
        finally:
            self._put_pool_connection(conn)

    def cleanup_cache(self):
        """Clean expired cache entries"""
        if not self._indicator_cache:
            return
        
        current_time = datetime.now()
        expired_keys = []
        
        with self._cache_lock:
            for key, entry in self._indicator_cache.items():
                if (current_time - entry['timestamp']).seconds > config.CACHE_EXPIRY_MINUTES * 60:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._indicator_cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

    def _update_performance_stats(self, query_time: float):
        """Update performance statistics"""
        # Initialize if not exists
        if not hasattr(self, '_performance_stats') or not self._performance_stats:
            self._performance_stats = {
                'queries_executed': 0,
                'cache_hits': 0,
                'avg_query_time': 0.0,
                'last_cleanup': datetime.now()
            }
        
        self._performance_stats['queries_executed'] += 1
        total_queries = self._performance_stats['queries_executed']
        current_avg = self._performance_stats['avg_query_time']
        
        # Calculate rolling average
        self._performance_stats['avg_query_time'] = (
            (current_avg * (total_queries - 1) + query_time) / total_queries
        )
    


    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        # Initialize performance stats if not exist
        if not hasattr(self, '_performance_stats') or not self._performance_stats:
            self._performance_stats = {
                'queries_executed': 0,
                'cache_hits': 0,
                'avg_query_time': 0.0,
                'last_cleanup': datetime.now()
            }
        
        cache_hit_rate = 0
        if self._performance_stats['queries_executed'] > 0:
            cache_hit_rate = (self._performance_stats['cache_hits'] / 
                            self._performance_stats['queries_executed']) * 100
        
        return {
            'queries_executed': self._performance_stats['queries_executed'],
            'cache_hits': self._performance_stats['cache_hits'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'avg_query_time': round(self._performance_stats['avg_query_time'], 3),
            'cache_size': len(self._indicator_cache) if self._indicator_cache else 0,
            'pool_status': 'active' if hasattr(self, '_connection_pool') and self._connection_pool else 'inactive'
        }

    def optimize_database(self):
        """Run database optimization tasks"""
        conn = self._get_pool_connection()
        try:
            with conn.cursor() as cursor:
                # Update table statistics
                cursor.execute("ANALYZE stocks_categories_table")
                cursor.execute("ANALYZE agent_live_signals")
                cursor.execute("ANALYZE agent_technical_indicators")
                
                # Cleanup old signals (keep last 30 days)
                cleanup_date = datetime.now() - timedelta(days=30)
                cursor.execute(
                    "DELETE FROM agent_live_signals WHERE signal_time < %s",
                    (cleanup_date,)
                )
                
                conn.commit()
                self.logger.info("Database optimization completed")
                
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
        finally:
            self._put_pool_connection(conn)

    def close_connections(self):
        """Clean shutdown of connection pool"""
        if self._connection_pool:
            try:
                self._connection_pool.closeall()
                self.logger.info("Connection pool closed")
            except:
                pass
    
    # ==========================================
# ADD THESE METHODS TO EnhancedDatabaseManager class
# ==========================================

    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for dashboard display"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Recent signals count
                cursor.execute("""
                    SELECT COUNT(*) FROM agent_live_signals 
                    WHERE signal_time >= NOW() - INTERVAL '24 hours'
                """)
                recent_signals = cursor.fetchone()[0]
                
                # Signal distribution
                cursor.execute("""
                    SELECT signal_type, COUNT(*) FROM agent_live_signals 
                    WHERE signal_time >= NOW() - INTERVAL '7 days'
                    GROUP BY signal_type
                """)
                signal_distribution = dict(cursor.fetchall())
                
                # Average confidence - use correct column name
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'agent_live_signals' 
                    AND column_name IN ('overall_confidence', 'technical_score', 'confidence_score')
                """)
                
                confidence_columns = [row[0] for row in cursor.fetchall()]
                if confidence_columns:
                    confidence_col = confidence_columns[0]  # Use first available
                    cursor.execute(f"""
                        SELECT AVG({confidence_col}) FROM agent_live_signals 
                        WHERE signal_time >= NOW() - INTERVAL '24 hours'
                    """)
                    avg_confidence = cursor.fetchone()[0] or 0
                else:
                    avg_confidence = 0.5  # Default
                
                return {
                    'recent_signals_24h': recent_signals,
                    'signal_distribution_7d': signal_distribution,
                    'avg_confidence_24h': float(avg_confidence),
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Dashboard metrics query failed: {e}")
            return {'error': str(e)}
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                stats = {}
                
                # Total signals
                cursor.execute("SELECT COUNT(*) FROM agent_live_signals")
                stats['total_signals'] = cursor.fetchone()[0]
                
                # Symbols analyzed
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM agent_live_signals")
                stats['unique_symbols'] = cursor.fetchone()[0]
                
                # Database size info
                cursor.execute("""
                    SELECT 
                        schemaname, 
                        tablename, 
                        pg_total_relation_size(schemaname||'.'||tablename) as size
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename LIKE 'agent_%'
                """)
                
                table_sizes = {}
                total_size = 0
                for row in cursor.fetchall():
                    size_mb = row[2] / (1024 * 1024)
                    table_sizes[row[1]] = round(size_mb, 2)
                    total_size += size_mb
                
                stats['table_sizes_mb'] = table_sizes
                stats['total_database_size_mb'] = round(total_size, 2)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"System statistics query failed: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self) -> Dict:
        """Clean up old data and return cleanup summary"""
        import config
        
        try:
            conn = self.get_connection()
            cleanup_summary = {}
            
            with conn.cursor() as cursor:
                # Clean old signals
                cutoff_date = datetime.now() - timedelta(days=config.DATA_RETENTION_DAYS)
                
                # Check if table exists first
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_live_signals'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    cursor.execute("""
                        DELETE FROM agent_live_signals 
                        WHERE signal_time < %s
                    """, (cutoff_date,))
                    
                    signals_cleaned = cursor.rowcount
                    cleanup_summary['signals_cleaned'] = signals_cleaned
                else:
                    cleanup_summary['signals_cleaned'] = 0
                
                # Clean old technical indicators if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_technical_indicators'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    cursor.execute("""
                        DELETE FROM agent_technical_indicators 
                        WHERE date < %s
                    """, (cutoff_date,))
                    
                    indicators_cleaned = cursor.rowcount
                    cleanup_summary['indicators_cleaned'] = indicators_cleaned
                else:
                    cleanup_summary['indicators_cleaned'] = 0
                
                conn.commit()
                
                cleanup_summary['cleanup_date'] = cutoff_date.isoformat()
                cleanup_summary['status'] = 'completed'
                
                self.logger.info(f"Data cleanup completed: {cleanup_summary}")
                return cleanup_summary
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def get_health_check_data(self) -> Dict:
        """Get data for system health checks"""
        try:
            health_data = {}
            
            # Test connection
            health_data['database_connected'] = self.test_connection()
            
            # Check table accessibility
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if main tables exist and are accessible
                tables_to_check = [
                    'stocks_categories_table',
                    'agent_live_signals',
                    'agent_technical_indicators'
                ]
                
                table_status = {}
                for table in tables_to_check:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                        table_status[table] = 'accessible'
                    except Exception as e:
                        table_status[table] = f'error: {str(e)[:50]}'
                
                health_data['table_status'] = table_status
                
                # Check recent data activity
                cursor.execute("""
                    SELECT COUNT(*) FROM agent_live_signals 
                    WHERE signal_time >= NOW() - INTERVAL '1 hour'
                """)
                recent_activity = cursor.fetchone()[0]
                health_data['recent_activity'] = recent_activity
                
            # Performance stats
            if hasattr(self, 'get_performance_stats'):
                health_data['performance'] = self.get_performance_stats()
            
            health_data['last_checked'] = datetime.now().isoformat()
            return health_data
            
        except Exception as e:
            return {
                'error': str(e),
                'database_connected': False,
                'last_checked': datetime.now().isoformat()
            }
    
    def backup_critical_data(self) -> Dict:
        """Create backup of critical data"""
        try:
            from pathlib import Path
            import json
            
            # Create backup directory
            backup_dir = Path('backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"system_backup_{timestamp}.json"
            
            # Get critical data
            conn = self.get_connection()
            backup_data = {}
            
            with conn.cursor() as cursor:
                # First check what columns actually exist
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'agent_live_signals'
                """)
                
                available_columns = [row[0] for row in cursor.fetchall()]
                self.logger.info(f"Available columns in agent_live_signals: {available_columns}")
                
                # Use existing columns or provide defaults
                if 'overall_confidence' in available_columns:
                    confidence_col = 'overall_confidence'
                elif 'technical_score' in available_columns:
                    confidence_col = 'technical_score'
                else:
                    confidence_col = '0.5 as confidence_score'  # Default value
                
                # Recent signals with correct column names
                cursor.execute(f"""
                    SELECT symbol, signal_type, {confidence_col}, signal_time 
                    FROM agent_live_signals 
                    WHERE signal_time >= NOW() - INTERVAL '7 days'
                    ORDER BY signal_time DESC
                    LIMIT 100
                """)
                
                signals = []
                for row in cursor.fetchall():
                    signals.append({
                        'symbol': row[0],
                        'signal_type': row[1],
                        'confidence_score': float(row[2]) if row[2] is not None else 0.5,
                        'signal_time': row[3].isoformat() if row[3] else datetime.now().isoformat()
                    })
                
                backup_data['recent_signals'] = signals
                backup_data['backup_timestamp'] = timestamp
                backup_data['record_count'] = len(signals)
            
            # Save backup
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return {
                'status': 'completed',
                'backup_file': str(backup_file),
                'records_backed_up': len(signals),
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_multiple_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get fundamental data for multiple symbols in one query"""
        if not symbols:
            return {}
        
        placeholders = ','.join(['%s'] * len(symbols))
        query = f"""
            SELECT * FROM stocks_categories_table 
            WHERE symbol IN ({placeholders})
        """
        
        results = self.execute_query(query, symbols, fetch_all=True)
        return {row['symbol']: row for row in results}
    
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """Get complete fundamental data for a symbol"""
        try:
            query = """
            SELECT * FROM stocks_categories_table 
            WHERE symbol = %s
            """
            
            result = self.execute_query(query, (symbol,))
            
            if result and isinstance(result, list) and len(result) > 0:
                # Convert Row to dict properly
                row_data = result[0]
                if hasattr(row_data, '_asdict'):
                    return row_data._asdict()
                elif hasattr(row_data, 'keys'):
                    return dict(row_data)
                else:
                    return {}
            else:
                self.logger.warning(f"No fundamental data found for {symbol}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return {}
        
    def return_connection(self, conn):
        """Return connection to pool or close it"""
        try:
            if hasattr(self, 'connection_pool') and self.connection_pool:
                self.connection_pool.putconn(conn)
            else:
                conn.close()
        except Exception as e:
            self.logger.warning(f"Error returning connection: {e}")
            try:
                conn.close()
            except:
                pass

    def get_connection(self):
        """Get database connection from pool or create new one"""
        try:
            if hasattr(self, 'connection_pool') and self.connection_pool:
                return self.connection_pool.getconn()
            else:
                # Fallback to direct connection
                import psycopg2
                return psycopg2.connect(**self.connection_params)
        except Exception as e:
            self.logger.error(f"Failed to get connection: {e}")
            raise