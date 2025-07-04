-- Performance optimization indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historical_symbol_timestamp 
ON historical_data_3m_2025_q3(symbol, timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historical_symbol_date 
ON historical_data_3m_2025_q3(symbol, date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_time 
ON agent_live_signals(symbol, signal_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_status_time 
ON agent_live_signals(status, signal_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_technical_symbol_time 
ON agent_technical_indicators(symbol, analysis_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_status 
ON agent_portfolio_positions(status);

-- Analyze for query optimization
ANALYZE stocks_categories_table;
ANALYZE agent_live_signals;
ANALYZE agent_technical_indicators;