"""
FILE: utils/event_analyzer.py
PURPOSE: Event Impact Analysis - Day 3B Implementation

DESCRIPTION:
- Detects earnings, results, policy announcements
- Analyzes event impact on sentiment
- Provides event-based sentiment weighting

USAGE:
- Called by news_sentiment_agent.py for event detection
- Enhances sentiment analysis with event context
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class EventAnalyzer:
    """Analyze news events for trading impact"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.event_patterns = {
            'earnings': [
                r'earnings?', r'quarterly results?', r'q[1-4] results?',
                r'profit', r'revenue', r'net income'
            ],
            'policy': [
                r'rbi', r'reserve bank', r'monetary policy', r'interest rate',
                r'government policy', r'budget', r'regulation'
            ],
            'corporate': [
                r'merger', r'acquisition', r'dividend', r'split',
                r'bonus', r'rights issue', r'ipo'
            ]
        }
        
        self.impact_multipliers = {
            'earnings': 1.5,
            'policy': 1.3,
            'corporate': 1.2,
            'default': 1.0
        }
    
    def analyze_events(self, articles: List[Dict]) -> Dict:
        """Analyze events in news articles"""
        
        events_detected = []
        total_impact = 0
        
        for article in articles:
            event_info = self._detect_events(article)
            if event_info['event_type'] != 'general':
                events_detected.append(event_info)
                total_impact += event_info['impact_score']
        
        return {
            'events_detected': events_detected,
            'event_count': len(events_detected),
            'total_impact': min(total_impact, 2.0),  # Cap impact
            'primary_event': events_detected[0] if events_detected else None
        }
    
    def _detect_events(self, article: Dict) -> Dict:
        """Detect event type in single article"""
        
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return {
                        'event_type': event_type,
                        'impact_score': self.impact_multipliers[event_type],
                        'article_title': article.get('title', ''),
                        'detection_pattern': pattern
                    }
        
        return {
            'event_type': 'general',
            'impact_score': self.impact_multipliers['default'],
            'article_title': article.get('title', ''),
            'detection_pattern': 'none'
        }