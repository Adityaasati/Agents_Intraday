import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List

class SchemaCreator:
    """Simplified Schema Creator for Day 1 - Nexus Trading System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection_params = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', 5432)),
            'database': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD')
        }
    
    def create_essential_tables(self) -> bool:
        """Create only essential tables for Day 1"""
        
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
        """Create all tables - first essential, then optional"""
        
        # Start with essential tables
        if not self.create_essential_tables():
            self.logger.error("Failed to create essential tables")
            return False
        
        # Add optional tables for future days
        optional_tables = [
            'agent_portfolio_positions',
            'agent_fundamental_data'
        ]
        
        for table in optional_tables:
            self._create_table(table)  # Don't fail if these don't work
        
        return True
    
    def _create_table(self, table_name: str) -> bool:
        """Create individual table based on name"""
        
        table_schemas = {
            'agent_symbol_integration': self._get_symbol_integration_schema(),
            'agent_technical_indicators': self._get_technical_indicators_schema(),
            'agent_live_signals': self._get_live_signals_schema(),
            'agent_portfolio_positions': self._get_portfolio_positions_schema(),
            'agent_system_config': self._get_system_config_schema(),
            'agent_fundamental_data': self._get_fundamental_data_schema()
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
        """Schema for agent_symbol_integration table - simplified"""
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
        """Schema for agent_technical_indicators table - corrected with proper columns"""
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
        """Schema for agent_live_signals table - simplified"""
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
        """Schema for agent_portfolio_positions table - simplified"""
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
        """Schema for agent_fundamental_data table - simplified"""
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
            ('paper_trading', 'true', 'boolean', 'Paper trading mode', 'system')
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