"""
FILE: agents/fundamental_agent.py
PURPOSE: Fundamental Analysis Agent - Day 2 Implementation

DESCRIPTION:
- Analyzes fundamental strength using stocks_categories_table (30 columns)
- Calculates quality, valuation, growth scores
- Provides sector comparison and relative analysis
- Generates fundamental buy/sell signals

DEPENDENCIES:
- database/enhanced_database_manager.py
- config.py

USAGE:
- Called by signal_agent.py for fundamental scoring
- Used for sector rotation analysis
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pytz

import config

class FundamentalAgent:
    """Fundamental Analysis Agent for quality and valuation analysis"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def analyze_symbol_fundamentals(self, symbol: str) -> Dict:
        """Complete fundamental analysis for a symbol"""
        
        try:
            # Get fundamental data from stocks_categories_table
            fundamental_data = self.db_manager.get_fundamental_data(symbol)
            
            if not fundamental_data:
                return {'symbol': symbol, 'error': 'no_fundamental_data'}
            
            # Calculate component scores
            valuation_score = self._calculate_valuation_score(fundamental_data)
            quality_score = self._calculate_quality_score(fundamental_data)
            growth_score = self._calculate_growth_score(fundamental_data)
            
            # Combined fundamental score
            fundamental_score = (
                valuation_score * 0.3 +
                quality_score * 0.4 +
                growth_score * 0.3
            )
            
            # Generate recommendation
            recommendation = self._get_recommendation(fundamental_score, fundamental_data)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(self.ist),
                'fundamental_score': round(fundamental_score, 3),
                'valuation_score': round(valuation_score, 3),
                'quality_score': round(quality_score, 3),
                'growth_score': round(growth_score, 3),
                'recommendation': recommendation,
                'category': fundamental_data.get('category', 'B'),
                'sector': fundamental_data.get('sector', 'Unknown'),
                'market_cap_type': fundamental_data.get('market_cap_type', 'Mid_Cap'),
                'reasoning': self._generate_reasoning(valuation_score, quality_score, growth_score)
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental analysis failed for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def sector_relative_analysis(self, symbol: str) -> Dict:
        """Compare symbol to sector peers"""
        
        try:
            symbol_data = self.db_manager.get_fundamental_data(symbol)
            if not symbol_data:
                return {'error': 'no_data'}
            
            sector = symbol_data.get('sector')
            if not sector:
                return {'error': 'no_sector'}
            
            # Get sector peers
            sector_symbols = self.db_manager.get_symbols_from_categories(
                limit=50, market_cap_types=[symbol_data.get('market_cap_type')]
            )
            
            sector_peers = [s for s in sector_symbols if s.get('sector') == sector and s.get('symbol') != symbol]
            
            if len(sector_peers) < 3:
                return {'error': 'insufficient_peers'}
            
            # Calculate sector averages
            sector_metrics = self._calculate_sector_averages(sector_peers)
            
            # Compare symbol to sector
            comparison = self._compare_to_sector(symbol_data, sector_metrics)
            
            return {
                'symbol': symbol,
                'sector': sector,
                'peer_count': len(sector_peers),
                'sector_metrics': sector_metrics,
                'comparison': comparison,
                'relative_strength': self._calculate_relative_strength(comparison)
            }
            
        except Exception as e:
            self.logger.error(f"Sector analysis failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_valuation_score(self, data: Dict) -> float:
        """Calculate valuation score (0-1, higher is better value)"""
        
        score = 0.5  # Start with neutral
        
        try:
            pe_ratio = data.get('pe_ratio')
            pb_ratio = data.get('pb_ratio')
            
            # PE Ratio scoring (lower is better)
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    score += 0.2
                elif pe_ratio < 25:
                    score += 0.1
                elif pe_ratio > 40:
                    score -= 0.2
            
            # PB Ratio scoring (lower is better for value)
            if pb_ratio and pb_ratio > 0:
                if pb_ratio < 1.5:
                    score += 0.15
                elif pb_ratio < 3:
                    score += 0.05
                elif pb_ratio > 5:
                    score -= 0.15
            
            # Dividend yield (higher is better for value)
            dividend_yield = data.get('dividend_yield')
            if dividend_yield and dividend_yield > 0:
                if dividend_yield > 3:
                    score += 0.1
                elif dividend_yield > 1.5:
                    score += 0.05
            
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    def _calculate_quality_score(self, data: Dict) -> float:
        """Calculate quality score (0-1, higher is better quality)"""
        
        score = 0.5  # Start with neutral
        
        try:
            # ROE scoring (higher is better)
            roe = data.get('roe_ratio')
            if roe and roe > 0:
                if roe > 20:
                    score += 0.25
                elif roe > 15:
                    score += 0.15
                elif roe > 10:
                    score += 0.05
                elif roe < 5:
                    score -= 0.15
            
            # ROCE scoring (higher is better)
            roce = data.get('roce_ratio')
            if roce and roce > 0:
                if roce > 20:
                    score += 0.2
                elif roce > 15:
                    score += 0.1
                elif roce < 10:
                    score -= 0.1
            
            # Category adjustment
            category = data.get('category', 'B')
            if category == 'A':
                score += 0.1
            elif category == 'C':
                score -= 0.1
            
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    def _calculate_growth_score(self, data: Dict) -> float:
        """Calculate growth score (0-1, higher is better growth)"""
        
        score = 0.5  # Start with neutral
        
        try:
            # Revenue growth
            revenue_growth = data.get('revenue_growth_ttm')
            if revenue_growth is not None:
                if revenue_growth > 20:
                    score += 0.2
                elif revenue_growth > 10:
                    score += 0.1
                elif revenue_growth < 0:
                    score -= 0.15
            
            # Profit growth
            profit_growth = data.get('profit_growth_ttm')
            if profit_growth is not None:
                if profit_growth > 25:
                    score += 0.25
                elif profit_growth > 15:
                    score += 0.15
                elif profit_growth < 0:
                    score -= 0.2
            
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    def _calculate_sector_averages(self, sector_peers: List[Dict]) -> Dict:
        """Calculate sector average metrics"""
        
        metrics = {'pe_ratio': [], 'pb_ratio': [], 'roe_ratio': [], 'roce_ratio': []}
        
        for peer in sector_peers:
            for metric in metrics:
                value = peer.get(metric)
                if value and value > 0:
                    metrics[metric].append(value)
        
        averages = {}
        for metric, values in metrics.items():
            if values:
                averages[f'sector_avg_{metric}'] = sum(values) / len(values)
        
        return averages
    
    def _compare_to_sector(self, symbol_data: Dict, sector_metrics: Dict) -> Dict:
        """Compare symbol metrics to sector averages"""
        
        comparison = {}
        
        metrics_to_compare = ['pe_ratio', 'pb_ratio', 'roe_ratio', 'roce_ratio']
        
        for metric in metrics_to_compare:
            symbol_value = symbol_data.get(metric)
            sector_avg = sector_metrics.get(f'sector_avg_{metric}')
            
            if symbol_value and sector_avg and symbol_value > 0 and sector_avg > 0:
                ratio = symbol_value / sector_avg
                
                if metric in ['pe_ratio', 'pb_ratio']:  # Lower is better
                    comparison[metric] = 'better' if ratio < 0.9 else 'worse' if ratio > 1.1 else 'average'
                else:  # Higher is better
                    comparison[metric] = 'better' if ratio > 1.1 else 'worse' if ratio < 0.9 else 'average'
        
        return comparison
    
    def _calculate_relative_strength(self, comparison: Dict) -> str:
        """Calculate overall relative strength vs sector"""
        
        better_count = sum(1 for v in comparison.values() if v == 'better')
        worse_count = sum(1 for v in comparison.values() if v == 'worse')
        
        if better_count > worse_count:
            return 'outperformer'
        elif worse_count > better_count:
            return 'underperformer'
        else:
            return 'inline'
    
    def _get_recommendation(self, fundamental_score: float, data: Dict) -> str:
        """Generate buy/sell recommendation"""
        
        if fundamental_score >= 0.7:
            return 'strong_buy'
        elif fundamental_score >= 0.6:
            return 'buy'
        elif fundamental_score >= 0.4:
            return 'hold'
        elif fundamental_score >= 0.3:
            return 'sell'
        else:
            return 'strong_sell'
    
    def _generate_reasoning(self, valuation_score: float, quality_score: float, growth_score: float) -> str:
        """Generate human-readable reasoning"""
        
        reasons = []
        
        if valuation_score > 0.6:
            reasons.append("attractive valuation")
        elif valuation_score < 0.4:
            reasons.append("expensive valuation")
        
        if quality_score > 0.6:
            reasons.append("high quality metrics")
        elif quality_score < 0.4:
            reasons.append("weak quality metrics")
        
        if growth_score > 0.6:
            reasons.append("strong growth")
        elif growth_score < 0.4:
            reasons.append("poor growth")
        
        return "; ".join(reasons) if reasons else "neutral fundamentals"