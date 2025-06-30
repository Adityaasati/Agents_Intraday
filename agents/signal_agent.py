"""
FILE: agents/signal_agent.py
PURPOSE: Signal Generation Agent - Master coordinator combining all analysis

DESCRIPTION:
- Combines technical, fundamental analysis scores
- Generates final buy/sell signals with confidence
- Applies category-based adjustments
- Manages signal filtering and ranking

DEPENDENCIES:
- agents/technical_agent.py
- agents/fundamental_agent.py
- config.py

USAGE:
- Main signal coordinator called by main system
- Combines all agent outputs for final decisions
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pytz

import config
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent

class SignalAgent:
    """Master signal generation coordinator"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize sub-agents
        self.technical_agent = TechnicalAgent(db_manager)
        self.fundamental_agent = FundamentalAgent(db_manager)
    
    def generate_signals(self, symbols: List[str], limit: int = 10) -> List[Dict]:
        """Generate trading signals for symbol list"""
        
        signals = []
        
        for symbol in symbols[:limit]:
            try:
                signal = self._analyze_single_symbol(symbol)
                if signal and 'error' not in signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Signal generation failed for {symbol}: {e}")
        
        # Filter and rank signals
        filtered_signals = self.filter_signals_by_quality(signals)
        ranked_signals = self.rank_signals_by_opportunity(filtered_signals)
        
        return ranked_signals
    
    def generate_live_signals(self, symbols: List[str] = None) -> List[Dict]:
        """Generate real-time signals for live trading"""
        
        if not symbols:
            # Get active symbols from database
            symbols_data = self.db_manager.get_symbols_from_categories(
                limit=20, 
                categories=['A', 'B'],
                market_cap_types=['Large_Cap', 'Mid_Cap']
            )
            symbols = [s['symbol'] for s in symbols_data]
        
        signals = self.generate_signals(symbols, limit=10)
        
        # Store high-confidence signals in database
        for signal in signals:
            if signal.get('overall_confidence', 0) >= config.MIN_CONFIDENCE_THRESHOLD:
                self._store_live_signal(signal)
        
        return signals
    
    def _analyze_single_symbol(self, symbol: str) -> Optional[Dict]:
        """Complete analysis for single symbol"""
        
        try:
            # Technical analysis
            technical_analysis = self.technical_agent.analyze_symbol(symbol)
            if 'error' in technical_analysis:
                self.logger.warning(f"Technical analysis failed for {symbol}: {technical_analysis['error']}")
                return None
            
            # Fundamental analysis
            fundamental_analysis = self.fundamental_agent.analyze_symbol_fundamentals(symbol)
            if 'error' in fundamental_analysis:
                self.logger.warning(f"Fundamental analysis failed for {symbol}: {fundamental_analysis['error']}")
                # Continue with technical only
                fundamental_score = 0.5
                fundamental_analysis = {'fundamental_score': 0.5}
            else:
                fundamental_score = fundamental_analysis.get('fundamental_score', 0.5)
            
            # Extract scores
            technical_score = technical_analysis.get('technical_score', 0.5)
            sentiment_score = 0.5  # Placeholder for Day 3
            
            # Calculate overall confidence
            category = technical_analysis.get('category', 'B')
            overall_confidence = config.calculate_final_confidence(
                technical_score, fundamental_score, sentiment_score, category
            )
            
            # Generate signal if confidence meets threshold
            signal_type, signal_strength = self._determine_signal_type(overall_confidence, technical_analysis)
            
            if signal_type == 'NONE':
                return None
            
            # Calculate price targets
            entry_price = technical_analysis.get('entry_price', 0)
            stop_loss = technical_analysis.get('stop_loss', 0)
            target_price = technical_analysis.get('target_price', 0)
            
            # Create signal
            signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_time': datetime.now(self.ist),
                'overall_confidence': round(overall_confidence, 3),
                'technical_score': round(technical_score, 3),
                'fundamental_score': round(fundamental_score, 3),
                'sentiment_score': round(sentiment_score, 3),
                'signal_strength': signal_strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'category': category,
                'sector': technical_analysis.get('sector', 'Unknown'),
                'market_cap_type': technical_analysis.get('market_cap_type', 'Mid_Cap'),
                'volatility_category': technical_analysis.get('volatility_category', 'Medium'),
                'primary_reasoning': self._generate_primary_reasoning(technical_analysis, fundamental_analysis),
                'supporting_factors': self._get_supporting_factors(technical_analysis, fundamental_analysis),
                'risk_factors': self._get_risk_factors(technical_analysis, fundamental_analysis)
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Symbol analysis failed for {symbol}: {e}")
            return None
    
    def _determine_signal_type(self, confidence: float, technical_analysis: Dict) -> tuple:
        """Determine signal type and strength"""
        
        if confidence < config.MIN_CONFIDENCE_THRESHOLD:
            return 'NONE', 'weak'
        
        # Check technical signals
        buy_signal = technical_analysis.get('buy_signal', False)
        sell_signal = technical_analysis.get('sell_signal', False)
        
        if buy_signal and confidence >= 0.7:
            strength = 'very_strong' if confidence >= 0.85 else 'strong'
            return 'BUY', strength
        elif buy_signal:
            return 'BUY', 'medium'
        elif sell_signal and confidence >= 0.7:
            strength = 'very_strong' if confidence >= 0.85 else 'strong'  
            return 'SELL', strength
        elif sell_signal:
            return 'SELL', 'medium'
        
        return 'NONE', 'weak'
    
    def filter_signals_by_quality(self, signals: List[Dict], min_confidence: float = None) -> List[Dict]:
        """Filter signals by quality criteria"""
        
        if min_confidence is None:
            min_confidence = config.MIN_CONFIDENCE_THRESHOLD
        
        filtered = []
        
        for signal in signals:
            # Confidence filter
            if signal.get('overall_confidence', 0) < min_confidence:
                continue
            
            # Category filter - prefer A and B category stocks
            if signal.get('category') == 'C':
                continue
            
            # Market cap filter - avoid small caps for now
            if signal.get('market_cap_type') == 'Small_Cap':
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def rank_signals_by_opportunity(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by opportunity score"""
        
        for signal in signals:
            # Calculate opportunity score
            confidence = signal.get('overall_confidence', 0)
            technical_score = signal.get('technical_score', 0)
            fundamental_score = signal.get('fundamental_score', 0)
            
            # Bonus for category A stocks
            category_bonus = 0.1 if signal.get('category') == 'A' else 0
            
            # Bonus for large cap stocks (lower risk)
            cap_bonus = 0.05 if signal.get('market_cap_type') == 'Large_Cap' else 0
            
            opportunity_score = (confidence * 0.6 + 
                               (technical_score + fundamental_score) / 2 * 0.4 + 
                               category_bonus + cap_bonus)
            
            signal['opportunity_score'] = round(opportunity_score, 3)
        
        # Sort by opportunity score (descending)
        return sorted(signals, key=lambda x: x.get('opportunity_score', 0), reverse=True)
    
    def _generate_primary_reasoning(self, technical: Dict, fundamental: Dict) -> str:
        """Generate primary reasoning for signal"""
        
        reasons = []
        
        # Technical reasoning
        if technical.get('buy_signal'):
            reasons.append("Technical buy signal")
        elif technical.get('sell_signal'):
            reasons.append("Technical sell signal")
        
        # Fundamental reasoning
        fund_score = fundamental.get('fundamental_score', 0.5)
        if fund_score > 0.6:
            reasons.append("Strong fundamentals")
        elif fund_score < 0.4:
            reasons.append("Weak fundamentals")
        
        # Technical details
        tech_reasoning = technical.get('reasoning', '')
        if tech_reasoning:
            reasons.append(tech_reasoning[:50])  # Keep it short
        
        return "; ".join(reasons[:3])  # Limit to 3 main reasons
    
    def _get_supporting_factors(self, technical: Dict, fundamental: Dict) -> List[str]:
        """Get supporting factors for signal"""
        
        factors = []
        
        # Technical factors
        indicators = technical.get('indicators', {})
        if indicators.get('volume_signal') in ['high', 'spike']:
            factors.append("High volume confirmation")
        
        if indicators.get('ma_trend') == 'bullish':
            factors.append("Bullish trend")
        
        # Fundamental factors
        if fundamental.get('quality_score', 0) > 0.6:
            factors.append("High quality metrics")
        
        if fundamental.get('valuation_score', 0) > 0.6:
            factors.append("Attractive valuation")
        
        return factors[:5]  # Limit to 5 factors
    
    def _get_risk_factors(self, technical: Dict, fundamental: Dict) -> List[str]:
        """Get risk factors for signal"""
        
        risks = []
        
        # Technical risks
        if technical.get('volatility_category') == 'High':
            risks.append("High volatility")
        
        # Fundamental risks
        if fundamental.get('growth_score', 0.5) < 0.4:
            risks.append("Poor growth prospects")
        
        if technical.get('market_cap_type') == 'Small_Cap':
            risks.append("Small cap liquidity risk")
        
        return risks[:3]  # Limit to 3 main risks
    
    def _store_live_signal(self, signal: Dict) -> bool:
        """Store live signal in database"""
        
        try:
            signal_uuid = f"{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            signal_data = {
                'symbol': signal['symbol'],
                'signal_uuid': signal_uuid,
                'signal_type': signal['signal_type'],
                'signal_time': signal['signal_time'],
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'target_price': signal.get('target_price'),
                'overall_confidence': signal['overall_confidence'],
                'technical_score': signal['technical_score'],
                'fundamental_score': signal['fundamental_score'],
                'sentiment_score': signal['sentiment_score'],
                'status': 'ACTIVE',
                'primary_reasoning': signal.get('primary_reasoning', ''),
                'created_at': datetime.now(self.ist)
            }
            
            return self.db_manager.store_live_signal(signal_data)
            
        except Exception as e:
            self.logger.error(f"Failed to store signal for {signal['symbol']}: {e}")
            return False
    
    def get_signal_summary(self, signals: List[Dict]) -> Dict:
        """Get summary of generated signals"""
        
        if not signals:
            return {'total_signals': 0}
        
        summary = {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s.get('signal_type') == 'BUY']),
            'sell_signals': len([s for s in signals if s.get('signal_type') == 'SELL']),
            'avg_confidence': round(sum(s.get('overall_confidence', 0) for s in signals) / len(signals), 3),
            'strong_signals': len([s for s in signals if s.get('signal_strength') in ['strong', 'very_strong']]),
            'categories': {},
            'sectors': {}
        }
        
        # Category breakdown
        for signal in signals:
            cat = signal.get('category', 'Unknown')
            summary['categories'][cat] = summary['categories'].get(cat, 0) + 1
        
        # Sector breakdown
        for signal in signals:
            sector = signal.get('sector', 'Unknown')
            summary['sectors'][sector] = summary['sectors'].get(sector, 0) + 1
        
        return summary