"""
FILE: utils/claude_sentiment.py
PURPOSE: Claude AI Sentiment Analysis Integration

DESCRIPTION:
- Integrates with Claude API for news sentiment analysis
- Provides structured sentiment scoring (0-1 scale)
- Basic impact assessment for trading relevance

DEPENDENCIES:
- requests (for API calls)
- os (for API key)

USAGE:
- Called by news_sentiment_agent.py
- Analyzes news headlines and content for trading sentiment
"""

import os
import requests
import json
import logging
from typing import Dict, Optional

class ClaudeSentiment:
    """Claude AI sentiment analysis for trading news"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('CLAUDE_API_KEY')
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        if not self.api_key:
            self.logger.warning("CLAUDE_API_KEY not found - sentiment analysis disabled")
    
    def analyze_text(self, title: str, content: str = '') -> Optional[Dict]:
        """Analyze sentiment of news title and content"""
        
        if not self.api_key:
            return None
        
        try:
            # Prepare text for analysis
            text_to_analyze = f"Title: {title}\nContent: {content[:300]}"  # Limit content
            
            # Create sentiment analysis prompt
            prompt = self._create_sentiment_prompt(text_to_analyze)
            
            # Call Claude API
            response = self._call_claude_api(prompt)
            
            if response:
                return self._parse_sentiment_response(response)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return None
    
    def _create_sentiment_prompt(self, text: str) -> str:
        """Create sentiment analysis prompt for Claude"""
        
        return f"""Analyze the sentiment of this financial news for stock trading:

{text}

Provide a JSON response with:
1. sentiment_score: float 0-1 (0=very bearish, 0.5=neutral, 1=very bullish)
2. confidence: float 0-1 (how confident in the analysis)
3. impact_score: float 0-1 (relevance for trading decisions)
4. reasoning: brief explanation

Be objective and focus on market impact. Respond only with valid JSON."""
    
    def _call_claude_api(self, prompt: str) -> Optional[str]:
        """Call Claude API with sentiment prompt"""
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-haiku-20240307',  # Use fast, cost-effective model
                'max_tokens': 200,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('content', [{}])[0].get('text', '')
            else:
                self.logger.warning(f"Claude API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            return None
    
    def _parse_sentiment_response(self, response_text: str) -> Optional[Dict]:
        """Parse Claude's sentiment response"""
        
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                sentiment_data = json.loads(json_text)
                
                # Validate required fields
                required_fields = ['sentiment_score', 'confidence', 'impact_score']
                if all(field in sentiment_data for field in required_fields):
                    
                    # Ensure values are in valid range
                    sentiment_data['sentiment_score'] = max(0.0, min(1.0, float(sentiment_data['sentiment_score'])))
                    sentiment_data['confidence'] = max(0.0, min(1.0, float(sentiment_data['confidence'])))
                    sentiment_data['impact_score'] = max(0.0, min(1.0, float(sentiment_data['impact_score'])))
                    
                    return sentiment_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to parse sentiment response: {e}")
            return None