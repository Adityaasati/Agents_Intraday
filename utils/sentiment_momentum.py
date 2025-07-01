"""
FILE: utils/sentiment_momentum.py
PURPOSE: Sentiment Momentum Tracking - Day 3B Implementation

DESCRIPTION:
- Tracks sentiment changes over time
- Calculates sentiment trends and momentum
- Provides momentum-based adjustments

USAGE:
- Called by news_sentiment_agent.py for trend analysis
- Enhances sentiment with momentum context
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class SentimentMomentum:
    """Track and analyze sentiment momentum"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
    
    def calculate_momentum(self, symbol: str, current_sentiment: float) -> Dict:
        """Calculate sentiment momentum for symbol"""
        
        try:
            # Get historical sentiment
            historical_data = self._get_historical_sentiment(symbol, days_back=7)
            
            if len(historical_data) < 2:
                return {
                    'momentum_score': 0.0,
                    'trend_direction': 'neutral',
                    'momentum_strength': 'weak',
                    'data_points': len(historical_data)
                }
            
            # Calculate momentum metrics
            momentum_score = self._calculate_momentum_score(historical_data, current_sentiment)
            trend_direction = self._get_trend_direction(momentum_score)
            momentum_strength = self._get_momentum_strength(historical_data)
            
            return {
                'momentum_score': round(momentum_score, 3),
                'trend_direction': trend_direction,
                'momentum_strength': momentum_strength,
                'data_points': len(historical_data),
                'sentiment_range': self._get_sentiment_range(historical_data)
            }
            
        except Exception as e:
            self.logger.error(f"Momentum calculation failed for {symbol}: {e}")
            return {
                'momentum_score': 0.0,
                'trend_direction': 'neutral',
                'momentum_strength': 'weak',
                'data_points': 0
            }
    
    def _get_historical_sentiment(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """Get historical sentiment data"""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            query = """
            SELECT config_value, updated_at FROM agent_system_config 
            WHERE config_key LIKE %s 
            AND category = 'sentiment'
            AND updated_at > %s
            ORDER BY updated_at ASC
            """
            
            results = self.db_manager.execute_query(
                query, (f"sentiment_{symbol}_%", cutoff_time)
            )
            
            import json
            historical_data = []
            
            for result in results:
                try:
                    sentiment_data = json.loads(result['config_value'])
                    sentiment_data['timestamp'] = result['updated_at']
                    historical_data.append(sentiment_data)
                except json.JSONDecodeError:
                    continue
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical sentiment: {e}")
            return []
    
    def _calculate_momentum_score(self, historical_data: List[Dict], current_sentiment: float) -> float:
        """Calculate momentum score based on trend"""
        
        if len(historical_data) < 2:
            return 0.0
        
        # Get sentiment values
        sentiments = [data['sentiment_score'] for data in historical_data]
        sentiments.append(current_sentiment)
        
        # Calculate simple momentum
        if len(sentiments) >= 3:
            recent_trend = sentiments[-1] - sentiments[-3]
            momentum = recent_trend * 2  # Amplify for momentum
        else:
            momentum = sentiments[-1] - sentiments[0]
        
        return max(-1.0, min(1.0, momentum))  # Clamp to [-1, 1]
    
    def _get_trend_direction(self, momentum_score: float) -> str:
        """Get trend direction from momentum score"""
        
        if momentum_score > 0.1:
            return 'bullish'
        elif momentum_score < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_momentum_strength(self, historical_data: List[Dict]) -> str:
        """Get momentum strength"""
        
        if len(historical_data) < 3:
            return 'weak'
        
        # Calculate volatility as proxy for strength
        sentiments = [data['sentiment_score'] for data in historical_data]
        volatility = max(sentiments) - min(sentiments)
        
        if volatility > 0.3:
            return 'strong'
        elif volatility > 0.15:
            return 'medium'
        else:
            return 'weak'
    
    def _get_sentiment_range(self, historical_data: List[Dict]) -> Dict:
        """Get sentiment range statistics"""
        
        if not historical_data:
            return {'min': 0.5, 'max': 0.5, 'avg': 0.5}
        
        sentiments = [data['sentiment_score'] for data in historical_data]
        
        return {
            'min': round(min(sentiments), 3),
            'max': round(max(sentiments), 3),
            'avg': round(sum(sentiments) / len(sentiments), 3)
        }