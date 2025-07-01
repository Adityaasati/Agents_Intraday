"""
FILE: utils/news_fetcher.py
PURPOSE: News Fetching Utility - RSS feed integration

DESCRIPTION:
- Fetches news from Economic Times and MoneyControl RSS feeds
- Filters news by symbol relevance
- Basic content extraction and cleaning

DEPENDENCIES:
- requests (for RSS feed fetching)
- Basic XML parsing

USAGE:
- Called by news_sentiment_agent.py
- Provides structured news data for sentiment analysis
"""

import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re

class NewsFetcher:
    """Simple RSS news fetcher for sentiment analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # RSS feed URLs
        self.rss_feeds = {
            'economic_times': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'moneycontrol': 'https://www.moneycontrol.com/rss/business.xml'
        }
        
        self.timeout = 10
    
    def get_symbol_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Get news articles related to a specific symbol"""
        
        try:
            all_articles = self._fetch_all_news()
            
            # Filter by symbol relevance
            relevant_articles = []
            for article in all_articles:
                if self._is_relevant_to_symbol(article, symbol):
                    relevant_articles.append(article)
            
            # Filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_articles = [a for a in relevant_articles if a.get('pub_date', datetime.min) > cutoff_time]
            
            return recent_articles[:10]  # Limit to 10 most relevant
            
        except Exception as e:
            self.logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []
    

    def get_market_news(self, hours_back: int = 12) -> List[Dict]:
        """Get general market news - TIMEZONE FIXED"""
        
        try:
            all_articles = self._fetch_all_news()
            
            # Filter by market relevance
            market_articles = []
            market_keywords = ['market', 'sensex', 'nifty', 'trading', 'stocks', 'equity', 'economy']
            
            for article in all_articles:
                title_lower = article.get('title', '').lower()
                if any(keyword in title_lower for keyword in market_keywords):
                    market_articles.append(article)
            
            # FIXED: Handle timezone-aware vs naive datetime comparison
            cutoff_time = datetime.now()  # Remove timezone to match article dates
            cutoff_time = cutoff_time - timedelta(hours=hours_back)
            
            recent_articles = []
            for article in market_articles:
                pub_date = article.get('pub_date')
                if pub_date and pub_date != datetime.min:
                    # Ensure both are naive datetimes for comparison
                    if pub_date.tzinfo is not None:
                        pub_date = pub_date.replace(tzinfo=None)
                    
                    if pub_date > cutoff_time:
                        recent_articles.append(article)
                else:
                    # Include articles without dates
                    recent_articles.append(article)
            
            return recent_articles[:15]
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market news: {e}")
            return []
    
    def _fetch_all_news(self) -> List[Dict]:
        """Fetch news from all RSS feeds"""
        
        all_articles = []
        
        for source, url in self.rss_feeds.items():
            try:
                articles = self._fetch_rss_feed(url, source)
                all_articles.extend(articles)
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source}: {e}")
        
        # Sort by publication date (newest first)
        all_articles.sort(key=lambda x: x.get('pub_date', datetime.min), reverse=True)
        
        return all_articles
    
    def _fetch_rss_feed(self, url: str, source: str) -> List[Dict]:
        """Fetch and parse single RSS feed"""
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            articles = []
            
            # Parse RSS items
            for item in root.findall('.//item')[:20]:  # Limit to 20 per feed
                article = self._parse_rss_item(item, source)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.warning(f"RSS fetch failed for {url}: {e}")
            return []
    
    def _parse_rss_item(self, item: ET.Element, source: str) -> Optional[Dict]:
        """Parse single RSS item"""
        
        try:
            title = self._get_element_text(item, 'title')
            description = self._get_element_text(item, 'description')
            pub_date_str = self._get_element_text(item, 'pubDate')
            link = self._get_element_text(item, 'link')
            
            if not title:
                return None
            
            # Parse publication date
            pub_date = self._parse_date(pub_date_str)
            
            # Clean content
            content = self._clean_text(description) if description else ''
            
            return {
                'title': self._clean_text(title),
                'content': content,
                'pub_date': pub_date,
                'source': source,
                'link': link
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse RSS item: {e}")
            return None
    
    def _is_relevant_to_symbol(self, article: Dict, symbol: str) -> bool:
        """Check if article is relevant to symbol"""
        
        # Get company name from database
        try:
            # This would be enhanced with actual company name lookup
            symbol_keywords = [symbol.lower()]
            
            # Check title and content
            text_to_check = f"{article.get('title', '')} {article.get('content', '')}".lower()
            
            return any(keyword in text_to_check for keyword in symbol_keywords)
            
        except Exception:
            return False
    
    def _get_element_text(self, parent: ET.Element, tag: str) -> Optional[str]:
        """Safely get element text"""
        element = parent.find(tag)
        return element.text if element is not None and element.text else None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse RSS date string"""
        
        if not date_str:
            return datetime.min
        
        try:
            # Try common RSS date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            return datetime.min
            
        except Exception:
            return datetime.min
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        if not text:
            return ''
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:500]  # Limit length for API efficiency