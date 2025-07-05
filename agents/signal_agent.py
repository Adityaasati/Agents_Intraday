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
from .base_agent import BaseAgent
import config
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent

class SignalAgent(BaseAgent):
    """Master signal generation coordinator"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        
        # Initialize sub-agents
        self.technical_agent = TechnicalAgent(db_manager)
        self.fundamental_agent = FundamentalAgent(db_manager)
        
        try:
            from agents.news_sentiment_agent import NewsSentimentAgent
            self.sentiment_agent = NewsSentimentAgent(db_manager)
        except ImportError:
            self.logger.warning("News sentiment agent not available")
            self.sentiment_agent = None
    
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
    
    

    def generate_signal(self, symbol: str) -> Dict:
        """Generate signal for a single symbol (used by validation)"""
        
        try:
            import config
            from datetime import datetime
            
            # Get technical analysis
            technical_result = self.technical_agent.analyze_symbol(symbol)
            if 'error' in technical_result:
                return {'error': f'Technical analysis failed: {technical_result["error"]}'}
            
            # Get fundamental analysis  
            fundamental_result = self.fundamental_agent.analyze_symbol_fundamentals(symbol)
            if 'error' in fundamental_result:
                self.logger.warning(f"Fundamental analysis failed for {symbol}, using default score")
                fundamental_result = {'fundamental_score': 0.5}
            
            # Get sentiment analysis (optional)
            sentiment_score = 0.5  # Default neutral
            if self.sentiment_agent:
                try:
                    sentiment_result = self.sentiment_agent.analyze_symbol_sentiment(symbol)
                    sentiment_score = sentiment_result.get('sentiment_score', 0.5)
                except:
                    self.logger.warning(f"Sentiment analysis failed for {symbol}, using neutral")
            
            # Extract scores
            technical_score = technical_result.get('overall_score', 0.5)
            fundamental_score = fundamental_result.get('fundamental_score', 0.5)
            
            # Calculate overall confidence using existing pattern
            overall_confidence = self.calculate_confidence_score(
                technical_score, fundamental_score, sentiment_score
            )
            
            # Determine signal type
            signal_type = 'BUY' if overall_confidence > config.MIN_CONFIDENCE_THRESHOLD else 'HOLD'
            
            # Get symbol data for pricing
            symbol_data = self.db_manager.get_fundamental_data(symbol)
            if not symbol_data:
                return {'error': f'Symbol data not found for {symbol}'}
            
            # Get latest price from your historical data tables
            latest_data = self.db_manager.get_historical_data(symbol, 1)
            if latest_data.empty:
                # Fallback to stocks_categories_table current_price
                if symbol_data and symbol_data.get('current_price'):
                    latest_price = float(symbol_data['current_price'])
                else:
                    return {'error': f'No price data found for {symbol}'}
            else:
                latest_price = float(latest_data.iloc[0]['close'])
            
            # Calculate basic position sizing (default to avoid NULL constraint)
            default_position_value = 10000.0  # Default 10k position
            default_shares = int(default_position_value / latest_price)
            
            signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_time': datetime.now(self.ist),
                'entry_price': latest_price,
                'overall_confidence': overall_confidence,
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'sentiment_score': sentiment_score,
                'stop_loss': latest_price * 0.95,  # 5% stop loss
                'target_price': latest_price * 1.10,  # 10% target
                'recommended_shares': default_shares,
                'recommended_position_value': default_position_value,
                'recommended_position_size': default_position_value,  # Add this field to avoid NULL constraint
                'primary_reasoning': technical_result.get('primary_signal', 'Technical analysis signal'),
                'category': symbol_data.get('category', 'B'),
                'sector': symbol_data.get('sector', 'Unknown')
            }

            # Now calculate proper position sizing with complete signal
            try:
                from agents.risk_agent import RiskAgent
                risk_agent = RiskAgent(self.db_manager)
                
                position_size_info = risk_agent.calculate_position_size(signal)
                if 'error' not in position_size_info:
                    # Update signal with calculated position sizing
                    signal.update({
                        'recommended_shares': position_size_info.get('recommended_shares', default_shares),
                        'recommended_position_value': position_size_info.get('recommended_position_value', default_position_value),
                        'recommended_position_size': position_size_info.get('recommended_position_value', default_position_value)
                    })
            except Exception as e:
                self.logger.warning(f"Position sizing calculation failed for {symbol}: {e}")
                # Keep default values
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def calculate_confidence_score(self, technical_score: float, fundamental_score: float, sentiment_score: float) -> float:
        """Calculate overall confidence score using existing weighting"""
        
        try:
            import config
            
            # Use existing weighting pattern from documentation
            weights = {
                'technical': 0.50,    # 50% technical
                'fundamental': 0.30,  # 30% fundamental  
                'sentiment': 0.20     # 20% sentiment
            }
            
            # Calculate weighted score
            confidence = (
                technical_score * weights['technical'] +
                fundamental_score * weights['fundamental'] +
                sentiment_score * weights['sentiment']
            )
            
            # Apply category adjustments if available
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            return round(confidence, 3)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default neutral confidence
    def _get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score for symbol - Day 3A implementation"""
        
        if not self.sentiment_agent:
            return 0.5  # Fallback to neutral
        
        try:
            # Try to get recent cached sentiment first
            recent_sentiment = self.db_manager.get_recent_sentiment(symbol, hours_back=6)
            
            if recent_sentiment:
                return recent_sentiment.get('sentiment_score', 0.5)
            
            # Analyze fresh sentiment
            sentiment_result = self.sentiment_agent.analyze_symbol_sentiment(symbol, hours_back=24)
            
            return sentiment_result.get('sentiment_score', 0.5)
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
            return 0.5
    
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
            sentiment_score = self._get_sentiment_score(symbol)
            
            # Calculate overall confidence
            category = technical_analysis.get('category', 'B')
            overall_confidence = self.calculate_confidence_score(technical_score, fundamental_score, sentiment_score)
            
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
        """Store live signal in database with proper field mapping"""
        
        try:
            signal_uuid = f"{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Ensure all required fields are present with defaults
            signal_data = {
                'symbol': signal['symbol'],
                'signal_uuid': signal_uuid,
                'signal_type': signal['signal_type'],
                'signal_time': signal['signal_time'],
                'entry_price': signal.get('entry_price', 0.0),
                'stop_loss': signal.get('stop_loss', 0.0),
                'target_price': signal.get('target_price', 0.0),
                'overall_confidence': signal.get('overall_confidence', 0.5),
                'technical_score': signal.get('technical_score', 0.5),
                'fundamental_score': signal.get('fundamental_score', 0.5),
                'sentiment_score': signal.get('sentiment_score', 0.5),
                'recommended_position_size': signal.get('recommended_position_size', 10000.0),  # Required field
                'status': 'ACTIVE',
                'primary_reasoning': signal.get('primary_reasoning', 'Generated signal'),
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
    
    # Add these methods to the existing SignalAgent class in agents/signal_agent.py

    def auto_execute_signals(self, signals: List[Dict] = None) -> Dict:
        try:
            import config
            
            if not config.AUTO_EXECUTE_SIGNALS or not config.PAPER_TRADING_MODE:
                return {'status': 'disabled', 'executed': 0}
            
            if signals is None:
                signals = self.get_executable_signals()
            
            from agents.portfolio_agent import PortfolioAgent
            portfolio_agent = PortfolioAgent(self.db_manager)
            
            executed_count = 0
            for signal in signals:
                result = portfolio_agent.execute_paper_trade(signal)
                if 'error' not in result:
                    executed_count += 1
                    self._mark_signal_executed(signal.get('id'))
            
            return {'status': 'completed', 'executed': executed_count}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_executable_signals(self) -> List[Dict]:
        """Get signals ready for execution (standardized)"""
        
        try:
            import config
            
            query = """
                SELECT * FROM agent_live_signals 
                WHERE overall_confidence >= %s 
                AND status = 'ACTIVE'
                AND (executed_at IS NULL)
                ORDER BY overall_confidence DESC, signal_time DESC
                LIMIT %s
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        config.MIN_CONFIDENCE_THRESHOLD,
                        config.MAX_ORDERS_PER_DAY
                    ))
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            self.logger.error(f"Failed to get executable signals: {e}")
            return []
    
    
    def generate_trade_orders(self, symbols: List[str] = None) -> Dict:
        """Generate signals and auto-convert to trade orders (consolidated method)"""
        
        try:
            import config
            
            # Determine symbols to process
            if symbols:
                target_symbols = symbols
            else:
                # Get symbols based on trading mode
                if config.LIVE_TRADING_MODE and config.TRADE_MODE:
                    target_symbols = config.LIVE_TRADING_APPROVED_SYMBOLS[:5]
                else:
                    symbols_data = self.db_manager.get_symbols_from_categories(
                        categories=['A', 'B'], limit=10
                    )
                    target_symbols = [s['symbol'] for s in symbols_data]
            
            # Generate signals using existing batch method
            signals = self.generate_signals(target_symbols, limit=len(target_symbols))
            
            # Filter executable signals
            executable_signals = [
                s for s in signals 
                if s.get('overall_confidence', 0) >= config.MIN_CONFIDENCE_THRESHOLD
            ]
            
            # Auto-execute if enabled
            execution_result = {'executed': 0}
            if config.AUTO_EXECUTE_SIGNALS and executable_signals:
                execution_result = self.auto_execute_signals(executable_signals)
            
            return {
                'signals_generated': len(signals),
                'executable_signals': len(executable_signals),
                'execution_result': execution_result,
                'signals': executable_signals,
                'trading_mode': 'LIVE' if (config.LIVE_TRADING_MODE and config.TRADE_MODE) else 'PAPER'
            }
            
        except Exception as e:
            self.logger.error(f"Trade order generation failed: {e}")
            return {'error': str(e)}
    
    
    def _validate_execution_criteria(self, signal: Dict) -> bool:
        """Validate if signal meets execution criteria"""
        
        try:
            import config
            
            # Check confidence threshold
            if signal.get('overall_confidence', 0) < config.MIN_CONFIDENCE_THRESHOLD:
                return False
            
            # Check if symbol already has open position
            from agents.portfolio_agent import PortfolioAgent
            portfolio_agent = PortfolioAgent(self.db_manager)
            open_positions = portfolio_agent._get_open_paper_positions()
            
            symbol = signal.get('symbol')
            if any(pos.get('symbol') == symbol for pos in open_positions):
                return False
            
            # Check daily loss limit
            portfolio_summary = portfolio_agent.get_paper_portfolio_summary()
            daily_loss = abs(min(0, portfolio_summary.get('unrealized_pnl', 0)))
            
            if daily_loss >= config.PAPER_MAX_LOSS_PER_DAY:
                return False
            
            # Check position count limit
            if len(open_positions) >= config.PAPER_MAX_POSITIONS:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Execution criteria validation failed: {e}")
            return False
    
    def get_signal_performance_summary(self) -> Dict:
        """Get signal performance summary (fixed query)"""
        
        try:
            # Updated query to handle missing executed_at column gracefully
            query = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN executed_at IS NOT NULL THEN 1 END) as executed_signals,
                    AVG(overall_confidence) as avg_confidence,
                    COUNT(CASE WHEN overall_confidence >= 0.7 AND executed_at IS NOT NULL THEN 1 END) as high_conf_executed
                FROM agent_live_signals 
                WHERE signal_time >= CURRENT_DATE - INTERVAL '30 days'
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    
                    if result:
                        total_signals, executed_signals, avg_confidence, high_conf_executed = result
                        
                        # Calculate performance metrics
                        execution_rate = (executed_signals / total_signals * 100) if total_signals > 0 else 0
                        high_conf_success_rate = (high_conf_executed / executed_signals * 100) if executed_signals > 0 else 0
                        
                        return {
                            'total_signals': total_signals or 0,
                            'executed_signals': executed_signals or 0,
                            'execution_rate': round(execution_rate, 1),
                            'high_confidence_success_rate': round(high_conf_success_rate, 1),
                            'avg_confidence': round(avg_confidence or 0, 3)
                        }
                    
                    return {'error': 'No data available'}
                    
        except Exception as e:
            self.logger.error(f"Signal performance summary failed: {e}")
            return {'error': str(e)}

    
    # def execute_live_signals(self, signals: List[Dict] = None) -> Dict:
    #     """Execute approved signals with trade mode control"""
        
    #     try:
    #         import config
            
    #         if not config.LIVE_TRADING_MODE:
    #             return self.auto_execute_signals(signals)
            
    #         if not self._is_live_trading_allowed():
    #             return {'status': 'disabled', 'reason': 'Live trading conditions not met'}
            
    #         # Get signals to execute
    #         if signals is None:
    #             signals = self.get_live_executable_signals()
            
    #         from agents.portfolio_agent import PortfolioAgent
    #         portfolio_agent = PortfolioAgent(self.db_manager)
            
    #         executed_count = 0
    #         signal_only_count = 0
    #         execution_results = []
            
    #         for signal in signals:
    #             if self._validate_live_execution_criteria(signal):
    #                 result = portfolio_agent.execute_live_trade(signal)
                    
    #                 if result.get('status') == 'live_executed':
    #                     executed_count += 1
    #                     execution_results.append({
    #                         'symbol': signal.get('symbol'),
    #                         'status': 'live_executed',
    #                         'order_id': result.get('order_id'),
    #                         'execution_type': 'LIVE'
    #                     })
                        
    #                     # Mark signal as executed
    #                     self._mark_signal_executed(signal.get('id'), 'LIVE')
                    
    #                 elif result.get('status') == 'signal_only':
    #                     signal_only_count += 1
    #                     execution_results.append({
    #                         'symbol': signal.get('symbol'),
    #                         'status': 'signal_generated',
    #                         'confidence': result.get('signal_confidence'),
    #                         'would_execute': result.get('would_execute'),
    #                         'reason': 'TRADE_MODE disabled',
    #                         'execution_type': 'SIGNAL_ONLY'
    #                     })
                        
    #                     # Mark signal as generated (not executed)
    #                     self._mark_signal_executed(signal.get('id'), 'SIGNAL_ONLY')
                    
    #                 else:
    #                     execution_results.append({
    #                         'symbol': signal.get('symbol'),
    #                         'status': 'failed',
    #                         'error': result.get('error', 'Unknown error')
    #                     })
            
    #         if config.TRADE_MODE:
    #             self.logger.info(f"Live execution: {executed_count} trades executed")
    #         else:
    #             self.logger.info(f"Signal generation: {signal_only_count} signals generated (TRADE_MODE=no)")
            
    #         return {
    #             'status': 'completed',
    #             'live_executed': executed_count,
    #             'signals_generated': signal_only_count,
    #             'total_signals': len(signals),
    #             'results': execution_results,
    #             'execution_type': 'LIVE' if config.TRADE_MODE else 'SIGNAL_ONLY'
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Live signal execution failed: {e}")
    #         return self.auto_execute_signals(signals)

    # def get_live_executable_signals(self) -> List[Dict]:
    #     """Get signals ready for live execution with approved symbols filter"""
        
    #     try:
    #         import config
            
    #         # Create approved symbols list for SQL IN clause
    #         approved_symbols_str = "', '".join(config.LIVE_TRADING_APPROVED_SYMBOLS)
            
    #         query = f"""
    #             SELECT * FROM agent_live_signals 
    #             WHERE overall_confidence >= %s 
    #             AND created_at >= CURRENT_DATE 
    #             AND executed_at IS NULL
    #             AND symbol IN ('{approved_symbols_str}')
    #             ORDER BY overall_confidence DESC, created_at DESC
    #             LIMIT %s
    #         """
            
    #         limit = min(config.LIVE_MAX_POSITIONS, len(config.LIVE_TRADING_APPROVED_SYMBOLS))
            
    #         with self.db_manager.get_connection() as conn:
    #             with conn.cursor() as cursor:
    #                 cursor.execute(query, (config.LIVE_MIN_CONFIDENCE_THRESHOLD, limit))
    #                 columns = [desc[0] for desc in cursor.description]
    #                 rows = cursor.fetchall()
    #                 return [dict(zip(columns, row)) for row in rows]
                    
    #     except Exception as e:
    #         self.logger.error(f"Failed to get live executable signals: {e}")
    #         return []

    # def generate_live_trade_orders(self, symbols: List[str] = None) -> Dict:
    #     """Generate signals with trade mode awareness"""
        
    #     try:
    #         import config
            
    #         if not config.LIVE_TRADING_MODE:
    #             return self.generate_trade_orders(symbols)
            
    #         # Generate fresh signals
    #         if symbols:
    #             signals = []
    #             for symbol in symbols:
    #                 signal = self.generate_signal(symbol)
    #                 if signal and 'error' not in signal:
    #                     signals.append(signal)
    #         else:
    #             # Conservative symbol selection for live trading
    #             approved_symbols = config.LIVE_TRADING_APPROVED_SYMBOLS[:5]  # Top 5 approved
    #             signals = [self.generate_signal(symbol) for symbol in approved_symbols]
    #             signals = [s for s in signals if s and 'error' not in s]
            
    #         # Filter for live trading (higher confidence threshold)
    #         executable_signals = [
    #             s for s in signals 
    #             if s.get('overall_confidence', 0) >= config.LIVE_MIN_CONFIDENCE_THRESHOLD
    #         ]
            
    #         # Execute live trades or generate signals only
    #         execution_result = self.execute_live_signals(executable_signals)
            
    #         return {
    #             'signals_generated': len(signals),
    #             'live_executable_signals': len(executable_signals),
    #             'execution_result': execution_result,
    #             'signals': executable_signals,
    #             'execution_type': 'LIVE' if config.TRADE_MODE else 'SIGNAL_ONLY',
    #             'trade_mode': 'enabled' if config.TRADE_MODE else 'disabled'
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Live trade order generation failed: {e}")
    #         return self.generate_trade_orders(symbols)

    # def _is_live_trading_allowed(self) -> bool:
    #     """Check if live trading is allowed based on current conditions"""
        
    #     try:
    #         import config
    #         from agents.portfolio_agent import PortfolioAgent
            
    #         # Check if market is open
    #         portfolio_agent = PortfolioAgent(self.db_manager)
    #         if not portfolio_agent._is_market_open():
    #             return False
            
    #         # Check daily loss limits
    #         live_summary = portfolio_agent.get_live_portfolio_summary()
    #         if 'error' in live_summary:
    #             return False
            
    #         daily_loss = abs(min(0, live_summary.get('unrealized_pnl', 0)))
    #         if daily_loss >= config.LIVE_MAX_LOSS_PER_DAY:
    #             return False
            
    #         # Check position limits
    #         if live_summary.get('open_positions', 0) >= config.LIVE_MAX_POSITIONS:
    #             return False
            
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"Live trading permission check failed: {e}")
    #         return False

    # def _validate_live_execution_criteria(self, signal: Dict) -> bool:
    #     """Enhanced validation with approved symbols and trade mode"""
        
    #     try:
    #         import config
    #         from datetime import datetime, timedelta
            
    #         # Check if symbol is approved for live trading
    #         symbol = signal.get('symbol')
    #         if symbol not in config.LIVE_TRADING_APPROVED_SYMBOLS:
    #             return False
            
    #         # Higher confidence threshold for live trading
    #         if signal.get('overall_confidence', 0) < config.LIVE_MIN_CONFIDENCE_THRESHOLD:
    #             return False
            
    #         # Check signal freshness (only execute recent signals live)
    #         if isinstance(signal.get('created_at'), str):
    #             signal_time = datetime.fromisoformat(signal.get('created_at'))
    #         else:
    #             signal_time = signal.get('created_at', datetime.now())
            
    #         if datetime.now() - signal_time > timedelta(hours=1):
    #             return False
            
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"Live execution criteria validation failed: {e}")
    #         return False

    def _mark_signal_executed(self, signal_id: int, execution_type: str = 'PAPER'):
        """Mark signal as executed (standardized connection)"""
        
        try:
            from datetime import datetime
            
            query = """
                UPDATE agent_live_signals 
                SET executed_at = %s, execution_type = %s, status = 'EXECUTED'
                WHERE id = %s
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (datetime.now(), execution_type, signal_id))
                    
            self.logger.info(f"Signal {signal_id} marked as executed ({execution_type})")
            
        except Exception as e:
            self.logger.error(f"Failed to mark signal {signal_id} as executed: {e}")

    # def get_multiple_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
    #     """Get fundamental data for multiple symbols in one query"""
    #     if not symbols:
    #         return {}
        
    #     placeholders = ','.join(['%s'] * len(symbols))
    #     query = f"""
    #         SELECT * FROM stocks_categories_table 
    #         WHERE symbol IN ({placeholders})
    #     """
        
    #     results = self.execute_query(query, symbols, fetch_all=True)
    #     return {row['symbol']: row for row in results}