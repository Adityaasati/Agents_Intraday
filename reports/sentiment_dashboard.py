"""
FILE: reports/sentiment_dashboard.py
PURPOSE: Sentiment Analysis Reporting - Day 3B Implementation

DESCRIPTION:
- Generates sentiment analysis reports
- Provides sentiment trends and insights
- Market and symbol-level sentiment overview

USAGE:
- Called for sentiment reporting and analysis
- Provides structured sentiment insights
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz

class SentimentDashboard:
    """Sentiment analysis reporting and insights"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def generate_sentiment_report(self, symbols: List[str] = None) -> Dict:
        """Generate comprehensive sentiment report"""
        
        try:
            if not symbols:
                symbols = self._get_active_symbols()
            
            # Get sentiment data for symbols
            symbol_sentiments = []
            for symbol in symbols[:20]:  # Limit for performance
                sentiment_data = self._get_symbol_sentiment_summary(symbol)
                if sentiment_data:
                    symbol_sentiments.append(sentiment_data)
            
            # Market overview
            market_overview = self._generate_market_overview(symbol_sentiments)
            
            # Trend analysis
            trend_analysis = self._generate_trend_analysis(symbols[:10])
            
            # Event analysis
            event_summary = self._generate_event_summary(symbol_sentiments)
            
            report = {
                'report_time': datetime.now(self.ist),
                'symbols_analyzed': len(symbol_sentiments),
                'market_overview': market_overview,
                'trend_analysis': trend_analysis,
                'event_summary': event_summary,
                'top_positive_sentiment': self._get_top_sentiments(symbol_sentiments, positive=True),
                'top_negative_sentiment': self._get_top_sentiments(symbol_sentiments, positive=False),
                'momentum_leaders': self._get_momentum_leaders(symbol_sentiments)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate sentiment report: {e}")
            return {'error': str(e)}
    
    def get_symbol_sentiment_trend(self, symbol: str, days: int = 7) -> Dict:
        """Get sentiment trend for specific symbol"""
        
        try:
            history = self.db_manager.get_sentiment_history(symbol, days)
            
            if len(history) < 2:
                return {'symbol': symbol, 'trend': 'insufficient_data'}
            
            # Calculate trend metrics
            sentiments = [h['sentiment_score'] for h in history]
            trend_direction = 'rising' if sentiments[-1] > sentiments[0] else 'falling'
            
            volatility = max(sentiments) - min(sentiments)
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                'symbol': symbol,
                'trend_direction': trend_direction,
                'current_sentiment': sentiments[-1],
                'avg_sentiment': round(avg_sentiment, 3),
                'volatility': round(volatility, 3),
                'data_points': len(history),
                'latest_update': history[-1]['analysis_time']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get sentiment trend for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _get_active_symbols(self) -> List[str]:
        """Get list of active symbols with recent sentiment data"""
        
        try:
            query = """
            SELECT DISTINCT symbol FROM agent_sentiment_analysis 
            WHERE analysis_time > %s 
            ORDER BY symbol
            """
            
            cutoff = datetime.now(self.ist) - timedelta(days=1)
            results = self.db_manager.execute_query(query, (cutoff,))
            
            return [r['symbol'] for r in results]
            
        except Exception:
            return []
    
    def _get_symbol_sentiment_summary(self, symbol: str) -> Optional[Dict]:
        """Get sentiment summary for symbol"""
        
        try:
            latest = self.db_manager.get_latest_sentiment(symbol)
            
            if not latest:
                return None
            
            return {
                'symbol': symbol,
                'sentiment_score': latest['sentiment_score'],
                'confidence': latest.get('confidence', 0.5),
                'momentum_score': latest.get('momentum_score', 0),
                'trend_direction': latest.get('trend_direction', 'neutral'),
                'primary_event': latest.get('primary_event_type', 'general'),
                'last_update': latest['analysis_time']
            }
            
        except Exception:
            return None
    
    def _generate_market_overview(self, symbol_sentiments: List[Dict]) -> Dict:
        """Generate market sentiment overview"""
        
        if not symbol_sentiments:
            return {'status': 'no_data'}
        
        sentiments = [s['sentiment_score'] for s in symbol_sentiments]
        
        return {
            'average_sentiment': round(sum(sentiments) / len(sentiments), 3),
            'bullish_count': len([s for s in sentiments if s > 0.6]),
            'bearish_count': len([s for s in sentiments if s < 0.4]),
            'neutral_count': len([s for s in sentiments if 0.4 <= s <= 0.6]),
            'sentiment_range': {
                'max': round(max(sentiments), 3),
                'min': round(min(sentiments), 3)
            }
        }
    
    def _generate_trend_analysis(self, symbols: List[str]) -> Dict:
        """Generate trend analysis across symbols"""
        
        trends = {'rising': 0, 'falling': 0, 'neutral': 0}
        
        for symbol in symbols:
            trend_data = self.get_symbol_sentiment_trend(symbol, days=3)
            direction = trend_data.get('trend_direction', 'neutral')
            
            if direction in trends:
                trends[direction] += 1
            else:
                trends['neutral'] += 1
        
        return trends
    
    def _generate_event_summary(self, symbol_sentiments: List[Dict]) -> Dict:
        """Generate event impact summary"""
        
        events = {}
        
        for sentiment in symbol_sentiments:
            event_type = sentiment.get('primary_event', 'general')
            events[event_type] = events.get(event_type, 0) + 1
        
        return events
    
    def _get_top_sentiments(self, symbol_sentiments: List[Dict], positive: bool = True) -> List[Dict]:
        """Get top positive or negative sentiments"""
        
        sorted_sentiments = sorted(
            symbol_sentiments, 
            key=lambda x: x['sentiment_score'], 
            reverse=positive
        )
        
        return sorted_sentiments[:5]
    
    def _get_momentum_leaders(self, symbol_sentiments: List[Dict]) -> List[Dict]:
        """Get symbols with strongest momentum"""
        
        momentum_symbols = [s for s in symbol_sentiments if abs(s.get('momentum_score', 0)) > 0.1]
        
        return sorted(
            momentum_symbols,
            key=lambda x: abs(x.get('momentum_score', 0)),
            reverse=True
        )[:5]