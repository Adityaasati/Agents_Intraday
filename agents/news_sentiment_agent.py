"""
FILE: agents/news_sentiment_agent.py
PURPOSE: News Sentiment Analysis Agent - Day 3A Implementation

DESCRIPTION:
- Fetches news from RSS feeds (Economic Times, MoneyControl)
- Analyzes sentiment using Claude API
- Provides sentiment scores for trading signals
- Basic event impact analysis

DEPENDENCIES:
- utils/news_fetcher.py
- utils/claude_sentiment.py
- database/enhanced_database_manager.py

USAGE:
- Called by signal_agent.py for sentiment scoring
- Replaces 0.5 placeholder sentiment scores
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz

class NewsSentimentAgent:
    """News sentiment analysis for trading signals"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize sentiment components
        self._init_sentiment_components()
    

    def _init_sentiment_components(self):
        """Initialize enhanced sentiment components - MAINTAIN both fetcher and aggregator"""
        
        # Keep both news_fetcher and news_aggregator for backward compatibility
        self.news_fetcher = None
        self.news_aggregator = None
        self.claude_sentiment = None
        self.event_analyzer = None
        self.sentiment_momentum = None
        
        try:
            from utils.news_fetcher import NewsFetcher
            self.news_fetcher = NewsFetcher()
            self.logger.debug("NewsFetcher initialized")
        except Exception as e:
            self.logger.debug(f"NewsFetcher not available: {e}")
        
        try:
            from utils.news_aggregator import NewsAggregator
            self.news_aggregator = NewsAggregator()
            self.logger.debug("NewsAggregator initialized")
        except Exception as e:
            self.logger.debug(f"NewsAggregator not available: {e}")
        
        try:
            from utils.claude_sentiment import ClaudeSentiment
            self.claude_sentiment = ClaudeSentiment()
            self.logger.debug("ClaudeSentiment initialized")
        except Exception as e:
            self.logger.debug(f"ClaudeSentiment not available: {e}")
        
        try:
            from utils.event_analyzer import EventAnalyzer
            self.event_analyzer = EventAnalyzer()
            self.logger.debug("EventAnalyzer initialized")
        except Exception as e:
            self.logger.debug(f"EventAnalyzer not available: {e}")
        
        try:
            from utils.sentiment_momentum import SentimentMomentum
            self.sentiment_momentum = SentimentMomentum(self.db_manager)
            self.logger.debug("SentimentMomentum initialized")
        except Exception as e:
            self.logger.debug(f"SentimentMomentum not available: {e}")

    

 

    def analyze_symbol_sentiment(self, symbol: str, hours_back: int = 24) -> Dict:
        """Enhanced sentiment analysis - USES existing method names"""
        
        if not self._components_available():
            return {'symbol': symbol, 'sentiment_score': 0.5, 'status': 'unavailable'}
        
        try:
            # Get fundamental data for sector info
            fundamental_data = self.db_manager.get_fundamental_data(symbol)
            sector = fundamental_data.get('sector', 'Unknown')
            
            # Use news_aggregator if available, otherwise fall back to news_fetcher
            if self.news_aggregator:
                news_articles = self.news_aggregator.get_enhanced_news(
                    symbol=symbol, sector=sector, hours_back=hours_back
                )
            elif self.news_fetcher:
                news_articles = self.news_fetcher.get_symbol_news(symbol, hours_back)
            else:
                news_articles = []
            
            if not news_articles:
                return {'symbol': symbol, 'sentiment_score': 0.5, 'status': 'no_news'}
            
            # Analyze events (if available)
            if self.event_analyzer:
                event_analysis = self.event_analyzer.analyze_events(news_articles[:10])
            else:
                event_analysis = {'total_impact': 1.0, 'primary_event': {'event_type': 'general'}}
            
            # Analyze sentiment with Claude
            sentiment_results = []
            for article in news_articles[:8]:  # Limit for API efficiency
                sentiment = self.claude_sentiment.analyze_text(
                    article.get('title', ''),
                    article.get('content', '')
                )
                if sentiment:
                    sentiment_results.append(sentiment)
            
            if not sentiment_results:
                return {'symbol': symbol, 'sentiment_score': 0.5, 'status': 'analysis_failed'}
            
            # Calculate enhanced sentiment score
            base_sentiment = self._calculate_weighted_sentiment(sentiment_results)
            event_impact = event_analysis.get('total_impact', 1.0)
            adjusted_sentiment = self._apply_event_impact(base_sentiment, event_impact)
            
            # Get momentum analysis (if available)
            if self.sentiment_momentum:
                momentum_data = self.sentiment_momentum.calculate_momentum(symbol, adjusted_sentiment)
                final_sentiment = self._apply_momentum_adjustment(adjusted_sentiment, momentum_data)
            else:
                momentum_data = {'momentum_score': 0, 'trend_direction': 'neutral'}
                final_sentiment = adjusted_sentiment
            
            # MAINTAIN existing method name: _store_sentiment_data (not _store_enhanced_sentiment)
            self._store_sentiment_data(symbol, sentiment_results, final_sentiment)
            
            return {
                'symbol': symbol,
                'sentiment_score': round(final_sentiment, 3),
                'base_sentiment': round(base_sentiment, 3),
                'event_impact': round(event_impact, 2),
                'momentum_score': momentum_data.get('momentum_score', 0),
                'trend_direction': momentum_data.get('trend_direction', 'neutral'),
                'primary_event': event_analysis.get('primary_event', {}).get('event_type', 'general'),
                'articles_analyzed': len(sentiment_results),
                'status': 'success',
                'last_update': datetime.now(self.ist)
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced sentiment analysis failed for {symbol}: {e}")
            return {'symbol': symbol, 'sentiment_score': 0.5, 'status': 'error'}




    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment - USES existing method patterns"""
        
        if not self._components_available():
            return {'market_sentiment': 0.5, 'status': 'unavailable'}
        
        try:
            # Use news_aggregator if available, otherwise fall back to news_fetcher
            if self.news_aggregator:
                market_news = self.news_aggregator.get_market_news(hours_back=12)
            elif self.news_fetcher:
                market_news = self.news_fetcher.get_market_news(hours_back=12)
            else:
                market_news = []
            
            if not market_news:
                return {'market_sentiment': 0.5, 'status': 'no_news'}
            
            sentiment_scores = []
            for article in market_news[:10]:  # Top 10 market news
                sentiment = self.claude_sentiment.analyze_text(
                    article.get('title', ''),
                    article.get('content', '')
                )
                if sentiment and 'sentiment_score' in sentiment:
                    sentiment_scores.append(sentiment['sentiment_score'])
            
            if sentiment_scores:
                market_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            else:
                market_sentiment = 0.5
            
            return {
                'market_sentiment': round(market_sentiment, 3),
                'articles_analyzed': len(sentiment_scores),
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Market sentiment analysis failed: {e}")
            return {'market_sentiment': 0.5, 'status': 'error'}
    
    
    def _calculate_weighted_sentiment(self, sentiment_results: List[Dict]) -> float:
        """Calculate weighted sentiment score"""
        
        if not sentiment_results:
            return 0.5
        
        total_weight = 0
        weighted_sum = 0
        
        for result in sentiment_results:
            score = result.get('sentiment_score', 0.5)
            confidence = result.get('confidence', 0.5)
            impact = result.get('impact_score', 0.5)
            
            # Weight by confidence and impact
            weight = confidence * impact
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _store_sentiment_data(self, symbol: str, sentiment_results: List[Dict], final_score: float):
        """Store sentiment analysis results"""
        
        try:
            sentiment_data = {
                'symbol': symbol,
                'analysis_time': datetime.now(self.ist),
                'sentiment_score': final_score,
                'articles_count': len(sentiment_results),
                'raw_results': sentiment_results[:3]  # Store top 3 for debugging
            }
            
            # Store in database (simple approach for Day 3A)
            self.db_manager.store_sentiment_data(sentiment_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to store sentiment data for {symbol}: {e}")
    
    
    def _components_available(self) -> bool:
        """Check if sentiment components are available - MAINTAIN existing logic"""
        return (self.news_fetcher is not None or self.news_aggregator is not None) and self.claude_sentiment is not None
    