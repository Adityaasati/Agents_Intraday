"""
FILE: utils/news_aggregator.py
PURPOSE: Enhanced News Aggregation - Day 3B Implementation

DESCRIPTION:
- Multiple news source integration
- Sector-specific news filtering
- Advanced relevance scoring
- News deduplication

USAGE:
- Enhanced version of news_fetcher.py
- Provides richer news data for sentiment analysis
"""

import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re

class NewsAggregator:
    """Enhanced news aggregation with multiple sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.rss_feeds = {
            'economic_times': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'moneycontrol': 'https://www.moneycontrol.com/rss/business.xml',
            
        }
        
        self.sector_keywords = {
            'Banking': ['bank', 'banking', 'hdfc', 'icici', 'sbi'],
            'IT': ['infosys', 'tcs', 'wipro', 'tech', 'software'],
            'Auto': ['maruti', 'hyundai', 'automotive', 'vehicle'],
            'Pharma': ['pharma', 'drug', 'medicine', 'healthcare'],
            'Energy': ['oil', 'gas', 'energy', 'power', 'reliance']
        }
        
        self.timeout = 10
    
    def get_enhanced_news(self, symbol: str = None, sector: str = None, hours_back: int = 24) -> List[Dict]:
        """Get enhanced news with multiple sources and filtering"""
        
        try:
            # Fetch from all sources
            all_articles = self._fetch_all_sources()
            
            # Apply filters
            if symbol:
                all_articles = self._filter_by_symbol(all_articles, symbol)
            
            if sector:
                sector_articles = self._filter_by_sector(all_articles, sector)
                all_articles.extend(sector_articles)
            
            # Time filter
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_articles = [a for a in all_articles if a.get('pub_date', datetime.min) > cutoff_time]
            
            # Remove duplicates and score relevance
            unique_articles = self._deduplicate_articles(recent_articles)
            scored_articles = self._score_relevance(unique_articles, symbol, sector)
            
            # Sort by relevance score
            scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return scored_articles[:15]  # Top 15 most relevant
            
        except Exception as e:
            self.logger.error(f"Enhanced news fetch failed: {e}")
            return []
    
    def get_sector_news(self, sector: str, hours_back: int = 12) -> List[Dict]:
        """Get sector-specific news"""
        
        try:
            all_articles = self._fetch_all_sources()
            sector_articles = self._filter_by_sector(all_articles, sector)
            
            # Time filter
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_articles = [a for a in sector_articles if a.get('pub_date', datetime.min) > cutoff_time]
            
            # Score and sort
            scored_articles = self._score_relevance(recent_articles, sector=sector)
            scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return scored_articles[:10]
            
        except Exception as e:
            self.logger.error(f"Sector news fetch failed for {sector}: {e}")
            return []
    
    def _fetch_all_sources(self) -> List[Dict]:
        """Fetch from all RSS sources"""
        
        all_articles = []
        
        for source, url in self.rss_feeds.items():
            try:
                articles = self._fetch_rss_feed(url, source)
                all_articles.extend(articles)
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {source}: {e}")
        
        return all_articles
    
    def _fetch_rss_feed(self, url: str, source: str) -> List[Dict]:
        """Fetch single RSS feed"""
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            articles = []
            
            for item in root.findall('.//item')[:15]:  # Limit per source
                article = self._parse_rss_item(item, source)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.warning(f"RSS fetch failed for {url}: {e}")
            return []
    
    def _parse_rss_item(self, item: ET.Element, source: str) -> Optional[Dict]:
        """Parse RSS item with enhanced data"""
        
        try:
            title = self._get_element_text(item, 'title')
            description = self._get_element_text(item, 'description')
            pub_date_str = self._get_element_text(item, 'pubDate')
            link = self._get_element_text(item, 'link')
            
            if not title:
                return None
            
            pub_date = self._parse_date(pub_date_str)
            content = self._clean_text(description) if description else ''
            
            return {
                'title': self._clean_text(title),
                'content': content,
                'pub_date': pub_date,
                'source': source,
                'link': link,
                'relevance_score': 0  # Will be calculated later
            }
            
        except Exception:
            return None
    
    def _filter_by_symbol(self, articles: List[Dict], symbol: str) -> List[Dict]:
        """Filter articles by symbol relevance"""
        
        relevant_articles = []
        symbol_lower = symbol.lower()
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            if symbol_lower in text:
                relevant_articles.append(article)
        
        return relevant_articles
    
    def _filter_by_sector(self, articles: List[Dict], sector: str) -> List[Dict]:
        """Filter articles by sector relevance"""
        
        if sector not in self.sector_keywords:
            return []
        
        keywords = self.sector_keywords[sector]
        sector_articles = []
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            if any(keyword in text for keyword in keywords):
                sector_articles.append(article)
        
        return sector_articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '')
            title_normalized = re.sub(r'[^\w\s]', '', title.lower())
            
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_articles.append(article)
        
        return unique_articles
    
    def _score_relevance(self, articles: List[Dict], symbol: str = None, sector: str = None) -> List[Dict]:
        """Score article relevance"""
        
        for article in articles:
            score = 0.5  # Base score
            
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            text = f"{title} {content}"
            
            # Symbol relevance
            if symbol and symbol.lower() in text:
                score += 0.3
                if symbol.lower() in title:
                    score += 0.2  # Title mentions are more important
            
            # Sector relevance
            if sector and sector in self.sector_keywords:
                keywords = self.sector_keywords[sector]
                keyword_matches = sum(1 for keyword in keywords if keyword in text)
                score += min(0.2, keyword_matches * 0.05)
            
            # Recency boost
            if article.get('pub_date'):
                hours_old = (datetime.now() - article['pub_date']).total_seconds() / 3600
                if hours_old < 6:
                    score += 0.1
            
            article['relevance_score'] = min(1.0, score)
        
        return articles
    
    def _get_element_text(self, parent: ET.Element, tag: str) -> Optional[str]:
        """Safely get element text"""
        element = parent.find(tag)
        return element.text if element is not None and element.text else None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse RSS date string"""
        
        if not date_str:
            return datetime.min
        
        try:
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
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:400]  # Limit length

    def get_market_news(self, hours_back: int = 12) -> List[Dict]:
        """Get general market news - MISSING METHOD ADDED"""
        
        try:
            all_articles = self._fetch_all_sources()
            
            # Filter by market relevance
            market_articles = []
            market_keywords = ['market', 'sensex', 'nifty', 'trading', 'stocks', 'equity', 'economy']
            
            for article in all_articles:
                title_lower = article.get('title', '').lower()
                if any(keyword in title_lower for keyword in market_keywords):
                    market_articles.append(article)
            
            # Time filter with timezone handling
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_articles = []
            
            for article in market_articles:
                pub_date = article.get('pub_date')
                if pub_date and pub_date != datetime.min:
                    # Handle timezone-aware vs naive datetime comparison
                    if pub_date.tzinfo is not None:
                        pub_date = pub_date.replace(tzinfo=None)
                    
                    if pub_date > cutoff_time:
                        recent_articles.append(article)
                else:
                    # Include articles without dates
                    recent_articles.append(article)
            
            return recent_articles[:15]  # Limit to 15 market articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market news: {e}")
            return []