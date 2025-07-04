-- Nexus Trading System - Performance Indexes
-- Run this SQL to improve query performance

-- Historical data indexes
CREATE INDEX IF NOT EXISTS idx_historical_symbol_timestamp 
ON historical_data_3m_2025_q3(symbol, timestamp);

CREATE INDEX IF NOT EXISTS idx_historical_symbol_date 
ON historical_data_3m_2025_q3(symbol, date);

-- Signal indexes
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
ON agent_live_signals(symbol, signal_time);

CREATE INDEX IF NOT EXISTS idx_signals_status 
ON agent_live_signals(status);

CREATE INDEX IF NOT EXISTS idx_signals_confidence 
ON agent_live_signals(overall_confidence);

-- Technical indicators indexes
CREATE INDEX IF NOT EXISTS idx_technical_symbol_time 
ON agent_technical_indicators(symbol, analysis_time);

-- Portfolio indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_symbol_status 
ON agent_portfolio_positions(symbol, status);

-- Fundamental data indexes
CREATE INDEX IF NOT EXISTS idx_fundamental_symbol_date 
ON agent_fundamental_data(symbol, analysis_date);

-- System config index
CREATE INDEX IF NOT EXISTS idx_config_key 
ON agent_system_config(config_key);

-- Analyze tables for query optimization
ANALYZE stocks_categories_table;
ANALYZE agent_live_signals;
ANALYZE agent_technical_indicators;
ANALYZE agent_portfolio_positions;
