-- Day 3B: Dedicated Sentiment Database Schema

-- Table for storing sentiment analysis results
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

CREATE INDEX idx_sentiment_analysis_symbol_time ON agent_sentiment_analysis(symbol, analysis_time);
CREATE INDEX idx_sentiment_analysis_score ON agent_sentiment_analysis(sentiment_score);

-- Table for storing individual news articles and their sentiment
CREATE TABLE IF NOT EXISTS agent_news_sentiment (
    id SERIAL PRIMARY KEY,
    sentiment_analysis_id INTEGER REFERENCES agent_sentiment_analysis(id),
    article_title TEXT NOT NULL,
    article_content TEXT,
    article_source VARCHAR(50),
    publication_date TIMESTAMP WITHOUT TIME ZONE,
    sentiment_score DECIMAL(3,2),
    confidence DECIMAL(3,2),
    impact_score DECIMAL(3,2),
    event_type VARCHAR(20),
    relevance_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_sentiment_analysis_id ON agent_news_sentiment(sentiment_analysis_id);
CREATE INDEX idx_news_sentiment_source ON agent_news_sentiment(article_source);

-- Table for market-wide sentiment
CREATE TABLE IF NOT EXISTS agent_market_sentiment (
    id SERIAL PRIMARY KEY,
    analysis_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    market_sentiment_score DECIMAL(3,2) NOT NULL,
    articles_analyzed INTEGER DEFAULT 0,
    primary_themes TEXT[],
    data_sources TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(analysis_time)
);

CREATE INDEX idx_market_sentiment_time ON agent_market_sentiment(analysis_time);

-- View for latest sentiment by symbol
CREATE OR REPLACE VIEW latest_sentiment_by_symbol AS
SELECT DISTINCT ON (symbol)
    symbol,
    sentiment_score,
    confidence,
    momentum_score,
    trend_direction,
    analysis_time
FROM agent_sentiment_analysis
ORDER BY symbol, analysis_time DESC;