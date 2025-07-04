"""
FILE: database/schema_creator.py
LOCATION: /database/ directory
PURPOSE: Complete Database Schema Creator for Nexus Trading System

DESCRIPTION:
- Creates all required database tables for the trading system
- Handles essential tables, pattern recognition tables, and backtesting tables
- Provides table verification and configuration initialization
- Maintains compatibility with existing database structure

DEPENDENCIES:
- psycopg2 (for PostgreSQL connections)
- logging (for error handling)

USAGE:
- Used during system setup to create database schema
- Called by main.py and integration scripts
- Provides table verification and management
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List
from .connection_config import get_connection_params


class SchemaCreator:
    """Complete Schema Creator for Nexus Trading System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection_params = get_connection_params()
        
    
    def create_essential_tables(self) -> bool:
        """Create essential tables for core functionality"""
        
        essential_tables = [
            'agent_symbol_integration',
            'agent_technical_indicators',
            'agent_live_signals',
            'agent_system_config'
        ]
        
        success_count = 0
        for table in essential_tables:
            if self._create_table(table):
                success_count += 1
        
        self.logger.info(f"Created {success_count}/{len(essential_tables)} essential tables")
        return success_count == len(essential_tables)
    
    def create_all_tables(self) -> bool:
        """Create all tables - essential, optional, and advanced"""
        
        # Start with essential tables
        if not self.create_essential_tables():
            self.logger.error("Failed to create essential tables")
            return False
        
        # Add optional tables
        optional_tables = [
            'agent_portfolio_positions',
            'agent_fundamental_data'
        ]
        
        for table in optional_tables:
            self._create_table(table)  # Don't fail if these don't work
        
        # Add advanced analysis tables
        self.create_pattern_backtest_tables()
        
        return True
    
    def create_pattern_backtest_tables(self) -> bool:
        """Create pattern recognition and backtesting tables"""
        
        pattern_tables = [
            'agent_pattern_signals',
            'agent_backtest_results', 
            'agent_backtest_trades'
        ]
        
        success_count = 0
        for table in pattern_tables:
            if self._create_table(table):
                success_count += 1
        
        self.logger.info(f"Created {success_count}/{len(pattern_tables)} pattern/backtest tables")
        return success_count == len(pattern_tables)
    
    def _create_table(self, table_name: str) -> bool:
        """Create individual table based on name"""
        
        table_schemas = {
            'agent_symbol_integration': self._get_symbol_integration_schema(),
            'agent_technical_indicators': self._get_technical_indicators_schema(),
            'agent_live_signals': self._get_live_signals_schema(),
            'agent_portfolio_positions': self._get_portfolio_positions_schema(),
            'agent_system_config': self._get_system_config_schema(),
            'agent_fundamental_data': self._get_fundamental_data_schema(),
            'agent_pattern_signals': self._get_pattern_signals_schema(),
            'agent_backtest_results': self._get_backtest_results_schema(),
            'agent_backtest_trades': self._get_backtest_trades_schema()
        }
        
        if table_name not in table_schemas:
            self.logger.error(f"Unknown table: {table_name}")
            return False
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table_name,))
                    
                    exists = cursor.fetchone()[0]
                    
                    if exists:
                        self.logger.debug(f"Table {table_name} already exists")
                        return True
                    
                    # Create table
                    cursor.execute(table_schemas[table_name])
                    conn.commit()
                    self.logger.info(f"Created table: {table_name}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to create table {table_name}: {e}")
            return False
    
    def _get_symbol_integration_schema(self) -> str:
        """Schema for agent_symbol_integration table"""
        return """
        CREATE TABLE agent_symbol_integration (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL UNIQUE,
            technical_analysis_enabled BOOLEAN DEFAULT true,
            fundamental_analysis_enabled BOOLEAN DEFAULT true,
            last_technical_update TIMESTAMP,
            total_signals_generated INTEGER DEFAULT 0,
            successful_signals INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_symbol_integration_symbol ON agent_symbol_integration(symbol);
        CREATE INDEX idx_symbol_integration_active ON agent_symbol_integration(is_active);
        """
    
    def _get_technical_indicators_schema(self) -> str:
        """Schema for agent_technical_indicators table"""
        return """
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
    
    def _get_live_signals_schema(self) -> str:
        """Schema for agent_live_signals table"""
        return """
        CREATE TABLE agent_live_signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            signal_uuid VARCHAR(100) UNIQUE NOT NULL,
            signal_type VARCHAR(10) NOT NULL,
            signal_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            entry_price DECIMAL(10,2) NOT NULL,
            stop_loss DECIMAL(10,2),
            target_price DECIMAL(10,2),
            recommended_position_size DECIMAL(12,2),
            risk_amount DECIMAL(12,2),
            technical_score DECIMAL(3,2),
            fundamental_score DECIMAL(3,2),
            sentiment_score DECIMAL(3,2),
            overall_confidence DECIMAL(3,2) NOT NULL,
            status VARCHAR(20) DEFAULT 'ACTIVE',
            primary_reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_live_signals_symbol ON agent_live_signals(symbol);
        CREATE INDEX idx_live_signals_status ON agent_live_signals(status);
        CREATE INDEX idx_live_signals_confidence ON agent_live_signals(overall_confidence);
        CREATE INDEX idx_live_signals_time ON agent_live_signals(signal_time);
        """
    
    def _get_portfolio_positions_schema(self) -> str:
        """Schema for agent_portfolio_positions table"""
        return """
        CREATE TABLE agent_portfolio_positions (
            id SERIAL PRIMARY KEY,
            signal_id INTEGER,
            symbol VARCHAR(20) NOT NULL,
            position_id VARCHAR(50) UNIQUE,
            entry_time TIMESTAMP WITHOUT TIME ZONE,
            entry_price DECIMAL(10,2),
            quantity INTEGER,
            position_value DECIMAL(12,2),
            current_price DECIMAL(10,2),
            unrealized_pnl DECIMAL(12,2),
            status VARCHAR(20) DEFAULT 'OPEN',
            exit_time TIMESTAMP WITHOUT TIME ZONE,
            exit_price DECIMAL(10,2),
            realized_pnl DECIMAL(12,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_portfolio_positions_symbol ON agent_portfolio_positions(symbol);
        CREATE INDEX idx_portfolio_positions_status ON agent_portfolio_positions(status);
        """
    
    def _get_system_config_schema(self) -> str:
        """Schema for agent_system_config table"""
        return """
        CREATE TABLE agent_system_config (
            id SERIAL PRIMARY KEY,
            config_key VARCHAR(100) UNIQUE NOT NULL,
            config_value TEXT,
            config_type VARCHAR(20) DEFAULT 'string',
            description TEXT,
            category VARCHAR(50),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_system_config_key ON agent_system_config(config_key);
        CREATE INDEX idx_system_config_category ON agent_system_config(category);
        """
    
    def _get_fundamental_data_schema(self) -> str:
        """Schema for agent_fundamental_data table"""
        return """
        CREATE TABLE agent_fundamental_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            analysis_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            fundamental_score DECIMAL(3,2),
            valuation_score DECIMAL(3,2),
            quality_score DECIMAL(3,2),
            growth_score DECIMAL(3,2),
            recommendation VARCHAR(20),
            reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, analysis_date)
        );
        
        CREATE INDEX idx_fundamental_data_symbol ON agent_fundamental_data(symbol);
        CREATE INDEX idx_fundamental_data_score ON agent_fundamental_data(fundamental_score);
        """
    
    def _get_pattern_signals_schema(self) -> str:
        """Schema for agent_pattern_signals table"""
        return """
        CREATE TABLE agent_pattern_signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            pattern_type VARCHAR(50) NOT NULL,
            pattern_subtype VARCHAR(20),
            confidence_score DECIMAL(4,3),
            key_levels TEXT,
            detected_date TIMESTAMP WITHOUT TIME ZONE,
            pattern_data TEXT,
            is_active BOOLEAN DEFAULT true,
            breakout_confirmed BOOLEAN DEFAULT false,
            breakout_date TIMESTAMP WITHOUT TIME ZONE,
            target_achieved BOOLEAN DEFAULT false,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_pattern_signals_symbol ON agent_pattern_signals(symbol);
        CREATE INDEX idx_pattern_signals_type ON agent_pattern_signals(pattern_type);
        CREATE INDEX idx_pattern_signals_confidence ON agent_pattern_signals(confidence_score);
        CREATE INDEX idx_pattern_signals_date ON agent_pattern_signals(detected_date);
        """

    def _get_backtest_results_schema(self) -> str:
        """Schema for agent_backtest_results table"""
        return """
        CREATE TABLE agent_backtest_results (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            start_date TIMESTAMP WITHOUT TIME ZONE,
            end_date TIMESTAMP WITHOUT TIME ZONE,
            initial_capital DECIMAL(12,2),
            final_capital DECIMAL(12,2),
            total_return DECIMAL(6,4),
            sharpe_ratio DECIMAL(6,3),
            max_drawdown DECIMAL(6,4),
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            win_rate DECIMAL(4,3),
            avg_win DECIMAL(10,2),
            avg_loss DECIMAL(10,2),
            profit_factor DECIMAL(6,2),
            largest_win DECIMAL(10,2),
            largest_loss DECIMAL(10,2),
            avg_holding_period_hours INTEGER,
            symbols_tested TEXT,
            strategy_parameters TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_backtest_results_strategy ON agent_backtest_results(strategy_name);
        CREATE INDEX idx_backtest_results_return ON agent_backtest_results(total_return);
        CREATE INDEX idx_backtest_results_sharpe ON agent_backtest_results(sharpe_ratio);
        CREATE INDEX idx_backtest_results_date ON agent_backtest_results(created_at);
        """

    def _get_backtest_trades_schema(self) -> str:
        """Schema for agent_backtest_trades table"""
        return """
        CREATE TABLE agent_backtest_trades (
            id SERIAL PRIMARY KEY,
            backtest_id INTEGER REFERENCES agent_backtest_results(id),
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            signal_type VARCHAR(10),
            timestamp TIMESTAMP WITHOUT TIME ZONE,
            price DECIMAL(10,2),
            shares INTEGER,
            trade_value DECIMAL(12,2),
            pnl DECIMAL(10,2),
            pnl_percent DECIMAL(6,3),
            commission DECIMAL(8,2),
            slippage DECIMAL(8,2),
            holding_period_hours INTEGER,
            exit_reason VARCHAR(50),
            confidence DECIMAL(4,3),
            source VARCHAR(20),
            pattern_type VARCHAR(50),
            technical_score DECIMAL(4,3),
            market_conditions VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_backtest_trades_backtest ON agent_backtest_trades(backtest_id);
        CREATE INDEX idx_backtest_trades_symbol ON agent_backtest_trades(symbol);
        CREATE INDEX idx_backtest_trades_pnl ON agent_backtest_trades(pnl);
        CREATE INDEX idx_backtest_trades_timestamp ON agent_backtest_trades(timestamp);
        """
    # Add this method to the SchemaCreator class in database/schema_creator.py

    def create_paper_trading_tables(self) -> bool:
        """Create additional tables for paper trading"""
        
        try:
            # Add executed_at column to existing signals table
            alter_signals_query = """
                ALTER TABLE agent_live_signals 
                ADD COLUMN IF NOT EXISTS executed_at TIMESTAMP WITHOUT TIME ZONE;
            """
            
            # Add paper trading specific columns to portfolio positions
            alter_positions_query = """
                ALTER TABLE agent_portfolio_positions 
                ADD COLUMN IF NOT EXISTS commission_paid DECIMAL(8,2) DEFAULT 0,
                ADD COLUMN IF NOT EXISTS slippage_amount DECIMAL(8,2) DEFAULT 0,
                ADD COLUMN IF NOT EXISTS execution_type VARCHAR(20) DEFAULT 'PAPER',
                ADD COLUMN IF NOT EXISTS max_gain DECIMAL(5,2) DEFAULT 0,
                ADD COLUMN IF NOT EXISTS max_loss DECIMAL(5,2) DEFAULT 0,
                ADD COLUMN IF NOT EXISTS holding_period_minutes INTEGER DEFAULT 0;
            """
            
            # Create paper trading performance tracking table
            performance_table_query = """
                CREATE TABLE IF NOT EXISTS agent_paper_performance (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    starting_capital DECIMAL(12,2),
                    ending_capital DECIMAL(12,2),
                    daily_pnl DECIMAL(10,2),
                    daily_return_percent DECIMAL(5,2),
                    trades_executed INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate DECIMAL(5,2) DEFAULT 0,
                    max_drawdown DECIMAL(5,2) DEFAULT 0,
                    sharpe_ratio DECIMAL(5,3) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_paper_performance_date ON agent_paper_performance(date);
            """
            
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(alter_signals_query)
                    cursor.execute(alter_positions_query)
                    cursor.execute(performance_table_query)
                    conn.commit()
                    
            self.logger.info("Paper trading tables created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create paper trading tables: {e}")
            return False
    
    def create_live_trading_tables(self) -> bool:
        """Create additional tables for live trading"""
        
        try:
            # Add live trading columns to existing signals table
            alter_signals_query = """
                ALTER TABLE agent_live_signals 
                ADD COLUMN IF NOT EXISTS execution_type VARCHAR(20) DEFAULT 'PAPER';
            """
            
            # Add live trading specific columns to portfolio positions
            alter_positions_query = """
                ALTER TABLE agent_portfolio_positions 
                ADD COLUMN IF NOT EXISTS order_id VARCHAR(50),
                ADD COLUMN IF NOT EXISTS order_status VARCHAR(20) DEFAULT 'PENDING',
                ADD COLUMN IF NOT EXISTS execution_type VARCHAR(20) DEFAULT 'PAPER',
                ADD COLUMN IF NOT EXISTS commission_paid DECIMAL(8,2) DEFAULT 0,
                ADD COLUMN IF NOT EXISTS slippage_amount DECIMAL(8,2) DEFAULT 0;
            """
            
            # Create live trading orders tracking table
            orders_table_query = """
                CREATE TABLE IF NOT EXISTS agent_live_orders (
                    id SERIAL PRIMARY KEY,
                    order_id VARCHAR(50) UNIQUE NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    signal_id INTEGER REFERENCES agent_live_signals(id),
                    order_type VARCHAR(20) NOT NULL,
                    transaction_type VARCHAR(10) NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DECIMAL(10,2),
                    order_status VARCHAR(20) DEFAULT 'PENDING',
                    filled_quantity INTEGER DEFAULT 0,
                    average_price DECIMAL(10,2) DEFAULT 0,
                    order_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    update_timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    exchange VARCHAR(10) DEFAULT 'NSE',
                    product VARCHAR(10) DEFAULT 'MIS',
                    validity VARCHAR(10) DEFAULT 'DAY',
                    tag VARCHAR(50) DEFAULT 'nexus_trading',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_live_orders_symbol ON agent_live_orders(symbol);
                CREATE INDEX IF NOT EXISTS idx_live_orders_status ON agent_live_orders(order_status);
            """
            
            # Create live trading performance tracking table
            live_performance_table_query = """
                CREATE TABLE IF NOT EXISTS agent_live_performance (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    starting_capital DECIMAL(12,2),
                    ending_capital DECIMAL(12,2),
                    live_pnl DECIMAL(10,2) DEFAULT 0,
                    paper_pnl DECIMAL(10,2) DEFAULT 0,
                    daily_return_percent DECIMAL(5,2),
                    live_trades_executed INTEGER DEFAULT 0,
                    paper_trades_executed INTEGER DEFAULT 0,
                    live_winning_trades INTEGER DEFAULT 0,
                    live_losing_trades INTEGER DEFAULT 0,
                    live_win_rate DECIMAL(5,2) DEFAULT 0,
                    execution_efficiency DECIMAL(5,2) DEFAULT 0,
                    api_calls_made INTEGER DEFAULT 0,
                    api_errors INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_live_performance_date ON agent_live_performance(date);
            """
            
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(alter_signals_query)
                    cursor.execute(alter_positions_query)
                    cursor.execute(orders_table_query)
                    cursor.execute(live_performance_table_query)
                    conn.commit()
                    
            self.logger.info("Live trading tables created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create live trading tables: {e}")
            return False
    
    def verify_essential_tables(self) -> Dict[str, bool]:
        """Verify essential tables exist"""
        
        essential_tables = [
            'stocks_categories_table',  # Your existing table
            'agent_symbol_integration',
            'agent_technical_indicators', 
            'agent_live_signals',
            'agent_system_config'
        ]
        
        table_status = {}
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    
                    for table in essential_tables:
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = %s
                            )
                        """, (table,))
                        
                        table_status[table] = cursor.fetchone()['exists']
        
        except Exception as e:
            self.logger.error(f"Failed to verify tables: {e}")
            
        return table_status
    
    def verify_pattern_backtest_tables(self) -> Dict[str, bool]:
        """Verify pattern recognition and backtesting tables exist"""
        
        pattern_tables = [
            'agent_pattern_signals',
            'agent_backtest_results',
            'agent_backtest_trades'
        ]
        
        table_status = {}
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    
                    for table in pattern_tables:
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = %s
                            )
                        """, (table,))
                        
                        table_status[table] = cursor.fetchone()['exists']
        
        except Exception as e:
            self.logger.error(f"Failed to verify pattern/backtest tables: {e}")
            
        return table_status
    
    def get_table_info(self, table_name: str) -> List[Dict]:
        """Get basic column information for a table"""
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                        LIMIT 20
                    """, (table_name,))
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Failed to get table info for {table_name}: {e}")
            return []
    
    def initialize_basic_config(self) -> bool:
        """Initialize basic system configuration"""
        
        basic_configs = [
            ('total_capital', '100000', 'float', 'Total trading capital', 'trading'),
            ('risk_per_trade', '2.0', 'float', 'Risk percentage per trade', 'risk'),
            ('max_positions', '5', 'integer', 'Maximum concurrent positions', 'trading'),
            ('min_confidence', '0.60', 'float', 'Minimum signal confidence', 'trading'),
            ('paper_trading', 'true', 'boolean', 'Paper trading mode', 'system'),
            ('enable_pattern_recognition', 'true', 'boolean', 'Enable pattern recognition', 'analysis'),
            ('enable_parameter_optimization', 'true', 'boolean', 'Enable parameter optimization', 'analysis'),
            ('enable_backtesting', 'true', 'boolean', 'Enable backtesting', 'analysis')
        ]
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    
                    for config_key, config_value, config_type, description, category in basic_configs:
                        cursor.execute("""
                            INSERT INTO agent_system_config 
                            (config_key, config_value, config_type, description, category)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (config_key) DO NOTHING
                        """, (config_key, config_value, config_type, description, category))
                    
                    conn.commit()
                    self.logger.info("Basic system configuration initialized")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize basic config: {e}")
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connection"""
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_all_tables_status(self) -> Dict[str, Dict]:
        """Get comprehensive status of all system tables"""
        
        all_tables = {
            'essential': ['agent_symbol_integration', 'agent_technical_indicators', 'agent_live_signals', 'agent_system_config'],
            'optional': ['agent_portfolio_positions', 'agent_fundamental_data'],
            'pattern_backtest': ['agent_pattern_signals', 'agent_backtest_results', 'agent_backtest_trades'],
            'existing': ['stocks_categories_table']
        }
        
        status = {}
        
        for category, tables in all_tables.items():
            status[category] = {}
            for table in tables:
                try:
                    with psycopg2.connect(**self.connection_params) as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_name = %s
                                )
                            """, (table,))
                            
                            status[category][table] = cursor.fetchone()[0]
                except Exception as e:
                    status[category][table] = False
                    
        return status
    
    def clean_old_data(self, days_to_keep: int = 30) -> bool:
        """Clean old data from analysis tables"""
        
        cleanup_tables = [
            ('agent_technical_indicators', 'created_at'),
            ('agent_pattern_signals', 'detected_date'),
            ('agent_backtest_trades', 'timestamp')
        ]
        
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cursor:
                    
                    cutoff_date = f"NOW() - INTERVAL '{days_to_keep} days'"
                    
                    for table, date_column in cleanup_tables:
                        cursor.execute(f"""
                            DELETE FROM {table} 
                            WHERE {date_column} < {cutoff_date}
                        """)
                        
                        deleted_rows = cursor.rowcount
                        self.logger.info(f"Cleaned {deleted_rows} old records from {table}")
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to clean old data: {e}")
            return False
    
    