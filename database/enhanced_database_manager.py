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
    
    # Update the get_historical_data method to auto-create current quarter table

    def get_historical_data(self, symbol: str, start_date: datetime = None, 
                        end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Get historical OHLCV data from quarterly tables - with auto-create"""
        
        if not end_date:
            end_date = datetime.now(self.ist)
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Get available quarters
        quarters = self.get_available_quarters()
        
        # Auto-create current quarter table if it doesn't exist
        current_quarter = f"historical_data_3m_{end_date.year}_q{((end_date.month - 1) // 3) + 1}"
        if current_quarter not in quarters:
            self.logger.info(f"Creating missing quarter table: {current_quarter}")
            year = end_date.year
            quarter = ((end_date.month - 1) // 3) + 1
            if self.create_quarterly_table(year, quarter):
                quarters.append(current_quarter)
        
        if not quarters:
            self.logger.warning("No historical data tables found")
            return pd.DataFrame()
        
        all_data = []
        for quarter_table in quarters[:2]:  # Check only latest 2 quarters
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
                
                if len(all_data) >= limit:
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