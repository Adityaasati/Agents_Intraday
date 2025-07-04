"""
FILE: agents/risk_agent.py
PURPOSE: Risk Management Agent - Position sizing and risk control

DESCRIPTION:
- Calculates position sizes based on risk parameters
- Determines stop loss levels using technical and volatility data
- Validates portfolio risk limits and diversification
- Manages risk-reward ratios and portfolio correlation

DEPENDENCIES:
- config.py (for risk parameters)
- database/enhanced_database_manager.py

USAGE:
- Called by signal_agent.py for position sizing
- Used by portfolio_agent.py for risk validation
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz
import pandas as pd
import config
from .base_agent import BaseAgent



class RiskAgent(BaseAgent):
    """Risk management and position sizing agent"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
    
    def calculate_position_size(self, signal: Dict, total_capital: float = None, 
                              risk_percent: float = None) -> Dict:
        """Calculate position size based on risk parameters"""
        
        if total_capital is None:
            total_capital = config.TOTAL_CAPITAL
        
        if risk_percent is None:
            risk_percent = config.RISK_PER_TRADE
        
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            symbol = signal.get('symbol', '')
            
            if not entry_price or not stop_loss or entry_price <= stop_loss:
                return {'error': 'invalid_prices'}
            
            # Base position size calculation
            risk_amount = total_capital * (risk_percent / 100)
            price_risk = entry_price - stop_loss
            base_shares = int(risk_amount / price_risk)
            base_position_value = base_shares * entry_price
            
            # Apply multipliers
            volatility_mult = self._get_volatility_multiplier(signal)
            market_cap_mult = self._get_market_cap_multiplier(signal)
            category_mult = self._get_category_multiplier(signal)
            confidence_mult = signal.get('overall_confidence', 0.6)
            
            # Final adjustments
            total_multiplier = volatility_mult * market_cap_mult * category_mult * confidence_mult
            adjusted_position_value = base_position_value * total_multiplier
            
            # Apply limits
            max_position_value = total_capital * (config.MAX_POSITION_SIZE_PERCENT / 100)
            final_position_value = min(adjusted_position_value, max_position_value)
            final_position_value = max(final_position_value, config.MIN_POSITION_SIZE)
            
            final_shares = int(final_position_value / entry_price)
            final_position_value = final_shares * entry_price
            final_risk_amount = final_shares * price_risk
            
            return {
                'symbol': symbol,
                'recommended_shares': final_shares,
                'recommended_position_value': round(final_position_value, 2),
                'actual_risk_amount': round(final_risk_amount, 2),
                'risk_percent_of_capital': round((final_risk_amount / total_capital) * 100, 2),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'price_risk_per_share': round(price_risk, 2),
                'total_multiplier': round(total_multiplier, 3),
                'volatility_multiplier': volatility_mult,
                'market_cap_multiplier': market_cap_mult,
                'category_multiplier': category_mult,
                'confidence_multiplier': round(confidence_mult, 3)
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed for {signal.get('symbol', 'Unknown')}: {e}")
            return {'error': str(e)}
    
    def determine_stop_loss(self, symbol: str, entry_price: float, method: str = 'atr') -> Dict:
        """Determine stop loss based on specified method"""
        
        try:
            # Get fundamental data for volatility category
            fundamental_data = self.db_manager.get_fundamental_data(symbol)
            volatility_category = fundamental_data.get('volatility_category', 'Medium')
            
            if method == 'atr':
                # ATR-based stop loss (preferred)
                return self._calculate_atr_stop_loss(symbol, entry_price, volatility_category)
            elif method == 'percentage':
                # Percentage-based stop loss (fallback)
                return self._calculate_percentage_stop_loss(entry_price, volatility_category)
            elif method == 'support':
                # Support level based stop loss
                return self._calculate_support_stop_loss(symbol, entry_price)
            else:
                return {'error': 'invalid_method'}
                
        except Exception as e:
            self.logger.error(f"Stop loss calculation failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def portfolio_risk_check(self, current_positions: List[Dict], new_signal: Dict) -> Dict:
        """Check if new position violates portfolio risk limits"""
        
        try:
            # Sector concentration check
            sector_check = self._check_sector_concentration(current_positions, new_signal)
            
            # Position size check
            size_check = self._check_position_size_limit(current_positions, new_signal)
            
            # Maximum positions check
            position_count_check = len(current_positions) < config.MAX_POSITIONS_LIVE
            
            # Overall portfolio risk
            portfolio_risk = self._calculate_portfolio_risk(current_positions, new_signal)
            
            risk_check = {
                'sector_concentration_ok': sector_check['within_limit'],
                'position_size_ok': size_check['within_limit'],
                'position_count_ok': position_count_check,
                'portfolio_risk_ok': portfolio_risk < config.MAX_DRAWDOWN,
                'current_sector_allocation': sector_check['current_allocation'],
                'current_position_count': len(current_positions),
                'estimated_portfolio_risk': round(portfolio_risk, 2),
                'recommendation': 'APPROVE' if all([
                    sector_check['within_limit'],
                    size_check['within_limit'],
                    position_count_check,
                    portfolio_risk < config.MAX_DRAWDOWN
                ]) else 'REJECT'
            }
            
            return risk_check
            
        except Exception as e:
            self.logger.error(f"Portfolio risk check failed: {e}")
            return {'error': str(e)}
    
    def _get_volatility_multiplier(self, signal: Dict) -> float:
        """Get position size multiplier based on volatility"""
        volatility_category = signal.get('volatility_category', 'Medium')
        return config.VOLATILITY_POSITION_MULTIPLIER.get(volatility_category, 1.0)
    
    def _get_market_cap_multiplier(self, signal: Dict) -> float:
        """Get position size multiplier based on market cap"""
        market_cap_type = signal.get('market_cap_type', 'Mid_Cap')
        return config.MARKET_CAP_MULTIPLIER.get(market_cap_type, 1.0)
    
    def _get_category_multiplier(self, signal: Dict) -> float:
        """Get position size multiplier based on stock category"""
        category = signal.get('category', 'B')
        return config.CATEGORY_MULTIPLIER.get(category, 1.0)
    
    def _calculate_atr_stop_loss(self, symbol: str, entry_price: float, volatility_category: str) -> Dict:
        """Calculate ATR-based stop loss"""
        
        try:
            # Get recent historical data for ATR calculation
            from datetime import timedelta
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=30)
            
            df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            
            if df.empty or len(df) < 14:
                # Fallback to percentage method
                return self._calculate_percentage_stop_loss(entry_price, volatility_category)
            
            # Calculate ATR manually
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                return self._calculate_percentage_stop_loss(entry_price, volatility_category)
            
            # Apply volatility-based ATR multiplier
            atr_multiplier = config.get_stop_loss_multiplier(volatility_category)
            stop_loss = entry_price - (atr * atr_multiplier)
            
            return {
                'stop_loss': round(stop_loss, 2),
                'method': 'atr',
                'atr_value': round(atr, 2),
                'atr_multiplier': atr_multiplier,
                'risk_per_share': round(entry_price - stop_loss, 2)
            }
            
        except Exception as e:
            self.logger.warning(f"ATR calculation failed for {symbol}: {e}")
            return self._calculate_percentage_stop_loss(entry_price, volatility_category)
    
    def _calculate_percentage_stop_loss(self, entry_price: float, volatility_category: str) -> Dict:
        """Calculate percentage-based stop loss (fallback method)"""
        
        # Adjust percentage based on volatility
        if volatility_category == 'Low':
            stop_percent = 1.0
        elif volatility_category == 'High':
            stop_percent = 2.5
        else:
            stop_percent = config.STOP_LOSS_PERCENT
        
        stop_loss = entry_price * (1 - stop_percent / 100)
        
        return {
            'stop_loss': round(stop_loss, 2),
            'method': 'percentage',
            'stop_percent': stop_percent,
            'risk_per_share': round(entry_price - stop_loss, 2)
        }
    
    def _calculate_support_stop_loss(self, symbol: str, entry_price: float) -> Dict:
        """Calculate support level based stop loss"""
        
        try:
            # This would use support levels from technical analysis
            # For now, fallback to percentage method
            return self._calculate_percentage_stop_loss(entry_price, 'Medium')
            
        except Exception:
            return self._calculate_percentage_stop_loss(entry_price, 'Medium')
    
    def _check_sector_concentration(self, current_positions: List[Dict], new_signal: Dict) -> Dict:
        """Check sector concentration limits"""
        
        new_sector = new_signal.get('sector', 'Unknown')
        
        # Calculate current sector allocation
        sector_allocations = {}
        total_value = 0
        
        for position in current_positions:
            sector = position.get('sector', 'Unknown')
            value = position.get('position_value', 0)
            sector_allocations[sector] = sector_allocations.get(sector, 0) + value
            total_value += value
        
        # Add new position
        new_position_value = new_signal.get('recommended_position_value', 0)
        current_sector_value = sector_allocations.get(new_sector, 0)
        new_sector_value = current_sector_value + new_position_value
        new_total_value = total_value + new_position_value
        
        new_sector_percent = (new_sector_value / new_total_value * 100) if new_total_value > 0 else 0
        
        return {
            'within_limit': new_sector_percent <= config.MAX_SECTOR_ALLOCATION,
            'current_allocation': round(new_sector_percent, 2),
            'limit': config.MAX_SECTOR_ALLOCATION,
            'sector': new_sector
        }
    
    def _check_position_size_limit(self, current_positions: List[Dict], new_signal: Dict) -> Dict:
        """Check individual position size limits"""
        
        new_position_value = new_signal.get('recommended_position_value', 0)
        max_position_value = config.TOTAL_CAPITAL * (config.MAX_POSITION_SIZE_PERCENT / 100)
        
        return {
            'within_limit': new_position_value <= max_position_value,
            'position_value': new_position_value,
            'limit': max_position_value,
            'percentage_of_capital': round((new_position_value / config.TOTAL_CAPITAL) * 100, 2)
        }
    
    def _calculate_portfolio_risk(self, current_positions: List[Dict], new_signal: Dict) -> float:
        """Calculate estimated portfolio risk (simplified)"""
        
        try:
            total_risk = 0
            
            # Current positions risk
            for position in current_positions:
                position_risk = position.get('actual_risk_amount', 0)
                total_risk += position_risk
            
            # Add new position risk
            new_risk = new_signal.get('actual_risk_amount', 0)
            total_risk += new_risk
            
            # Portfolio risk as percentage of capital
            portfolio_risk_percent = (total_risk / config.TOTAL_CAPITAL) * 100
            
            return portfolio_risk_percent
            
        except Exception:
            return 0.0
    
    # Add these methods to existing agents/risk_agent.py file
    # Insert at the end of the RiskAgent class, before the existing helper methods

    def calculate_portfolio_correlation(self, current_positions: List[Dict], new_symbol: str) -> Dict:
        """Calculate correlation matrix for portfolio including new symbol"""
        
        try:
            symbols = [pos.get('symbol') for pos in current_positions if pos.get('symbol')]
            if new_symbol not in symbols:
                symbols.append(new_symbol)
            
            if len(symbols) < 2:
                return {'correlation_risk': 'low', 'max_correlation': 0.0, 'correlated_symbols': []}
            
            # Get price data for correlation calculation
            correlations = {}
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    try:
                        corr_value = self._calculate_symbol_correlation(symbol1, symbol2)
                        correlations[f"{symbol1}-{symbol2}"] = corr_value
                    except:
                        correlations[f"{symbol1}-{symbol2}"] = 0.0
            
            if not correlations:
                return {'correlation_risk': 'low', 'max_correlation': 0.0, 'correlated_symbols': []}
            
            max_corr = max(abs(corr) for corr in correlations.values())
            high_corr_pairs = [pair for pair, corr in correlations.items() if abs(corr) > 0.7]
            
            risk_level = 'high' if max_corr > 0.8 else 'medium' if max_corr > 0.6 else 'low'
            
            return {
                'correlation_risk': risk_level,
                'max_correlation': round(max_corr, 3),
                'correlated_symbols': high_corr_pairs,
                'correlation_matrix': {k: round(v, 3) for k, v in correlations.items()}
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            return {'correlation_risk': 'unknown', 'max_correlation': 0.0, 'correlated_symbols': []}

    def calculate_enhanced_position_size(self, signal: Dict, current_positions: List[Dict]) -> Dict:
        """Enhanced position sizing with portfolio context and market regime"""
        
        try:
            # Get base position size
            base_result = self.calculate_position_size(signal)
            if 'error' in base_result:
                return base_result
            
            # Portfolio context adjustments
            symbol = signal.get('symbol', '')
            
            # Correlation adjustment
            correlation_data = self.calculate_portfolio_correlation(current_positions, symbol)
            corr_multiplier = self._get_correlation_multiplier(correlation_data)
            
            # Sector concentration adjustment
            sector_multiplier = self._get_sector_multiplier(current_positions, signal)
            
            # Market regime adjustment
            regime_multiplier = self._get_market_regime_multiplier(symbol)
            
            # Apply enhanced multipliers
            enhanced_shares = int(base_result['recommended_shares'] * corr_multiplier * sector_multiplier * regime_multiplier)
            enhanced_value = enhanced_shares * signal.get('entry_price', 0)
            
            # Risk limits
            max_position_value = config.TOTAL_CAPITAL * 0.15  # 15% max per position
            if enhanced_value > max_position_value:
                enhanced_shares = int(max_position_value / signal.get('entry_price', 1))
                enhanced_value = enhanced_shares * signal.get('entry_price', 0)
            
            enhanced_result = base_result.copy()
            enhanced_result.update({
                'recommended_shares': enhanced_shares,
                'recommended_position_value': round(enhanced_value, 2),
                'correlation_multiplier': round(corr_multiplier, 3),
                'sector_multiplier': round(sector_multiplier, 3),
                'regime_multiplier': round(regime_multiplier, 3),
                'enhancement_applied': True
            })
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Enhanced position sizing failed: {e}")
            return self.calculate_position_size(signal)  # Fallback to basic

    def get_portfolio_risk_metrics(self, current_positions: List[Dict]) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        
        try:
            if not current_positions:
                return {'total_risk': 0, 'risk_per_position': [], 'concentration_risk': 0}
            
            # Individual position risks
            position_risks = []
            total_risk = 0
            
            for pos in current_positions:
                risk_amount = pos.get('actual_risk_amount', 0)
                position_value = pos.get('position_value', 0)
                
                position_risk = {
                    'symbol': pos.get('symbol'),
                    'risk_amount': risk_amount,
                    'risk_percent': round((risk_amount / config.TOTAL_CAPITAL) * 100, 2),
                    'position_percent': round((position_value / config.TOTAL_CAPITAL) * 100, 2)
                }
                position_risks.append(position_risk)
                total_risk += risk_amount
            
            # Sector concentration risk
            sectors = {}
            for pos in current_positions:
                sector = self._get_symbol_sector(pos.get('symbol'))
                sectors[sector] = sectors.get(sector, 0) + pos.get('position_value', 0)
            
            max_sector_exposure = max(sectors.values()) if sectors else 0
            concentration_risk = (max_sector_exposure / config.TOTAL_CAPITAL) * 100
            
            # Correlation risk
            symbols = [pos.get('symbol') for pos in current_positions]
            correlation_data = self.calculate_portfolio_correlation(current_positions, '')
            
            return {
                'total_risk': round(total_risk, 2),
                'total_risk_percent': round((total_risk / config.TOTAL_CAPITAL) * 100, 2),
                'risk_per_position': position_risks,
                'concentration_risk': round(concentration_risk, 2),
                'correlation_risk': correlation_data.get('correlation_risk', 'unknown'),
                'max_correlation': correlation_data.get('max_correlation', 0),
                'sector_distribution': {k: round(v, 2) for k, v in sectors.items()},
                'risk_budget_used': round((total_risk / (config.TOTAL_CAPITAL * 0.2)) * 100, 2)  # 20% max portfolio risk
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio risk metrics calculation failed: {e}")
            return {'total_risk': 0, 'error': str(e)}

    # Helper methods for enhanced calculations

    def _calculate_symbol_correlation(self, symbol1: str, symbol2: str, days: int = 30) -> float:
        """Calculate correlation between two symbols"""
        
        try:
            from datetime import timedelta
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=days)
            
            df1 = self.db_manager.get_historical_data(symbol1, start_date, end_date)
            df2 = self.db_manager.get_historical_data(symbol2, start_date, end_date)
            
            if df1.empty or df2.empty or len(df1) < 10 or len(df2) < 10:
                return 0.0
            
            # Calculate returns
            df1['returns'] = df1['close'].pct_change().dropna()
            df2['returns'] = df2['close'].pct_change().dropna()
            
            # Align dates and calculate correlation
            common_dates = set(df1['date']).intersection(set(df2['date']))
            if len(common_dates) < 10:
                return 0.0
            
            returns1 = df1[df1['date'].isin(common_dates)]['returns'].values
            returns2 = df2[df2['date'].isin(common_dates)]['returns'].values
            
            if len(returns1) != len(returns2) or len(returns1) < 10:
                return 0.0
            
            correlation = float(pd.Series(returns1).corr(pd.Series(returns2)))
            return correlation if not pd.isna(correlation) else 0.0
            
        except Exception:
            return 0.0

    def _get_correlation_multiplier(self, correlation_data: Dict) -> float:
        """Get position size multiplier based on correlation"""
        
        max_corr = correlation_data.get('max_correlation', 0)
        
        if max_corr > 0.8:
            return 0.6  # Reduce position size significantly
        elif max_corr > 0.6:
            return 0.8  # Moderate reduction
        else:
            return 1.0  # No adjustment

    def _get_sector_multiplier(self, current_positions: List[Dict], new_signal: Dict) -> float:
        """Get position size multiplier based on sector concentration"""
        
        try:
            new_sector = self._get_symbol_sector(new_signal.get('symbol'))
            
            # Calculate current sector allocation
            sector_value = 0
            total_value = 0
            
            for pos in current_positions:
                pos_value = pos.get('position_value', 0)
                total_value += pos_value
                
                if self._get_symbol_sector(pos.get('symbol')) == new_sector:
                    sector_value += pos_value
            
            if total_value == 0:
                return 1.0
            
            current_sector_percent = (sector_value / total_value) * 100
            
            if current_sector_percent > 25:
                return 0.5  # Reduce significantly if sector overweight
            elif current_sector_percent > 20:
                return 0.8  # Moderate reduction
            else:
                return 1.0
                
        except Exception:
            return 1.0

    def _get_market_regime_multiplier(self, symbol: str) -> float:
        """Get position size multiplier based on market regime"""
        
        try:
            from agents.technical_agent import TechnicalAgent
            tech_agent = TechnicalAgent(self.db_manager)
            
            regime_summary = tech_agent.get_market_regime_summary([symbol])
            regime = regime_summary.get('dominant_regime', 'sideways')
            confidence = regime_summary.get('regime_confidence', 0.5)
            
            if regime == 'bull' and confidence > 0.7:
                return 1.2  # Increase position size in strong bull market
            elif regime == 'bear' and confidence > 0.7:
                return 0.7  # Reduce position size in strong bear market
            else:
                return 1.0  # Normal sizing for sideways/uncertain markets
                
        except Exception:
            return 1.0

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        
        try:
            fundamental_data = self.db_manager.get_fundamental_data(symbol)
            return fundamental_data.get('sector', 'Unknown')
        except:
            return 'Unknown'
    
    def _get_normal_ppf(self, confidence_level: float) -> float:
        """Get normal distribution percentile point function (fallback for scipy)"""
        
        try:
            from scipy import stats
            return stats.norm.ppf(1 - confidence_level/100)
        except ImportError:
            # Fallback approximation for common confidence levels
            confidence_mappings = {
                90.0: -1.282,  # 90% confidence
                95.0: -1.645,  # 95% confidence
                99.0: -2.326   # 99% confidence
            }
            
            return confidence_mappings.get(confidence_level, -1.645)
    # Add these methods to existing agents/risk_agent.py file
    # Insert at the end of the RiskAgent class

    def calculate_optimal_stop_loss(self, symbol: str, entry_price: float, signal_data: Dict = None) -> Dict:
        """Calculate optimal stop loss using multiple strategies"""
        
        try:
            # Calculate all stop loss types
            stop_strategies = {}
            
            # Trailing stop
            trailing_stop = self._calculate_trailing_stop(entry_price)
            stop_strategies['trailing'] = trailing_stop
            
            # Volatility-based stop (ATR)
            atr_stop = self._calculate_atr_stop(symbol, entry_price)
            stop_strategies['volatility'] = atr_stop
            
            # Technical level stop
            technical_stop = self._calculate_technical_stop(symbol, entry_price)
            stop_strategies['technical'] = technical_stop
            
            # Time-based stop
            if config.TIME_STOP_ENABLED:
                time_stop = self._calculate_time_stop(entry_price)
                stop_strategies['time'] = time_stop
            
            # Select optimal stop based on priority and risk-reward
            optimal_stop = self._select_optimal_stop(stop_strategies, entry_price)
            
            return {
                'optimal_stop_loss': optimal_stop['stop_price'],
                'strategy_used': optimal_stop['strategy'],
                'risk_reward_ratio': optimal_stop['risk_reward'],
                'all_strategies': stop_strategies,
                'confidence': optimal_stop.get('confidence', 0.7)
            }
            
        except Exception as e:
            self.logger.error(f"Optimal stop loss calculation failed: {e}")
            # Fallback to simple percentage stop
            fallback_stop = entry_price * (1 - config.TRAILING_STOP_INITIAL_PERCENT / 100)
            return {
                'optimal_stop_loss': fallback_stop,
                'strategy_used': 'fallback',
                'risk_reward_ratio': 2.0,
                'error': str(e)
            }

    def update_dynamic_stops(self, positions: List[Dict]) -> List[Dict]:
        """Update stop losses dynamically based on profit/market conditions"""
        
        updated_positions = []
        
        for position in positions:
            try:
                symbol = position.get('symbol')
                entry_price = position.get('entry_price', 0)
                current_stop = position.get('stop_loss', 0)
                current_price = self._get_current_price(symbol)
                
                if not all([symbol, entry_price, current_price]):
                    updated_positions.append(position)
                    continue
                
                # Calculate profit percentage
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Apply profit protection
                new_stop = self._apply_profit_protection(entry_price, current_price, current_stop, profit_pct)
                
                # Apply trailing stop logic
                trailing_stop = self._update_trailing_stop(entry_price, current_price, current_stop)
                
                # Use the more favorable stop (higher for long positions)
                final_stop = max(new_stop, trailing_stop)
                
                # Ensure stop doesn't move against position
                if final_stop > current_stop:
                    position['stop_loss'] = final_stop
                    position['stop_updated'] = datetime.now(self.ist)
                    position['stop_reason'] = self._get_stop_update_reason(profit_pct)
                
                updated_positions.append(position)
                
            except Exception as e:
                self.logger.warning(f"Failed to update stop for {position.get('symbol')}: {e}")
                updated_positions.append(position)
        
        return updated_positions

    def calculate_risk_parity_weights(self, symbols: List[str], lookback_months: int = None) -> Dict:
        """Calculate risk parity weights for portfolio"""
        
        if lookback_months is None:
            lookback_months = config.OPTIMIZATION_LOOKBACK_MONTHS
        
        try:
            # Get historical data for all symbols
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=lookback_months * 30)
            
            returns_data = {}
            for symbol in symbols:
                df = self.db_manager.get_historical_data(symbol, start_date, end_date)
                if not df.empty and len(df) > 20:
                    df['returns'] = df['close'].pct_change().dropna()
                    returns_data[symbol] = df['returns'].std() * (252 ** 0.5)  # Annualized volatility
            
            if len(returns_data) < 2:
                # Fallback to equal weights
                equal_weight = 1.0 / len(symbols)
                return {symbol: equal_weight for symbol in symbols}
            
            # Calculate risk parity weights
            total_inv_vol = sum(1/vol for vol in returns_data.values())
            risk_parity_weights = {
                symbol: (1/vol) / total_inv_vol 
                for symbol, vol in returns_data.items()
            }
            
            # Apply maximum weight constraint
            max_weight = config.MAX_SINGLE_ASSET_WEIGHT / 100
            constrained_weights = self._apply_weight_constraints(risk_parity_weights, max_weight)
            
            return constrained_weights
            
        except Exception as e:
            self.logger.error(f"Risk parity calculation failed: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}

    def calculate_portfolio_var(self, positions: List[Dict], confidence_level: float = None) -> Dict:
        """Calculate portfolio Value at Risk (VaR)"""
        
        if confidence_level is None:
            confidence_level = config.VAR_CONFIDENCE_LEVEL
        
        try:
            if not positions:
                return {'var_amount': 0, 'var_percent': 0, 'confidence': confidence_level}
            
            # Calculate portfolio value and daily returns
            total_value = sum(pos.get('position_value', 0) for pos in positions)
            
            # Estimate portfolio volatility (simplified)
            portfolio_vol = self._estimate_portfolio_volatility(positions)
            
            # Calculate VaR using normal distribution approximation
            z_score = self._get_normal_ppf(confidence_level)

            daily_var = abs(z_score * portfolio_vol * total_value)
            var_percent = (daily_var / total_value) * 100
            
            return {
                'var_amount': round(daily_var, 2),
                'var_percent': round(var_percent, 2),
                'confidence_level': confidence_level,
                'portfolio_value': total_value,
                'portfolio_volatility': round(portfolio_vol * 100, 2)
            }
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return {'var_amount': 0, 'var_percent': 0, 'error': str(e)}

    # Helper methods for stop loss optimization

    def _calculate_trailing_stop(self, entry_price: float) -> Dict:
        """Calculate trailing stop loss"""
        
        stop_percent = config.TRAILING_STOP_INITIAL_PERCENT
        stop_price = entry_price * (1 - stop_percent / 100)
        
        return {
            'stop_price': stop_price,
            'stop_percent': stop_percent,
            'strategy': 'trailing',
            'confidence': 0.8
        }

    def _calculate_atr_stop(self, symbol: str, entry_price: float) -> Dict:
        """Calculate ATR-based stop loss"""
        
        try:
            # Get recent data for ATR calculation
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=30)
            df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            
            if df.empty or len(df) < config.ATR_CALCULATION_PERIODS:
                # Fallback to percentage stop
                return self._calculate_trailing_stop(entry_price)
            
            # Calculate ATR
            df['tr'] = df[['high', 'low', 'close']].apply(self._calculate_true_range, axis=1)
            atr = df['tr'].rolling(config.ATR_CALCULATION_PERIODS).mean().iloc[-1]
            
            # Calculate stop based on ATR
            atr_stop_distance = atr * config.ATR_STOP_MULTIPLIER
            stop_price = entry_price - atr_stop_distance
            stop_percent = (atr_stop_distance / entry_price) * 100
            
            # Apply maximum stop limit
            if stop_percent > config.VOLATILITY_STOP_MAX_PERCENT:
                stop_percent = config.VOLATILITY_STOP_MAX_PERCENT
                stop_price = entry_price * (1 - stop_percent / 100)
            
            return {
                'stop_price': stop_price,
                'stop_percent': stop_percent,
                'atr_value': atr,
                'strategy': 'volatility',
                'confidence': 0.9
            }
            
        except Exception:
            return self._calculate_trailing_stop(entry_price)

    def _calculate_technical_stop(self, symbol: str, entry_price: float) -> Dict:
        """Calculate technical level-based stop loss"""
        
        try:
            # Get recent data for support level calculation
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=config.TECHNICAL_STOP_LOOKBACK_DAYS)
            df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            
            if df.empty or len(df) < 10:
                return self._calculate_trailing_stop(entry_price)
            
            # Find support level (simplified)
            recent_lows = df['low'].rolling(3).min()
            support_level = recent_lows.min()
            
            # Apply buffer below support
            buffer = support_level * (config.SUPPORT_RESISTANCE_BUFFER / 100)
            stop_price = support_level - buffer
            
            # Ensure stop is reasonable (not too far from entry)
            stop_percent = ((entry_price - stop_price) / entry_price) * 100
            if stop_percent > 25:  # If stop is too far, use trailing stop
                return self._calculate_trailing_stop(entry_price)
            
            return {
                'stop_price': stop_price,
                'stop_percent': stop_percent,
                'support_level': support_level,
                'strategy': 'technical',
                'confidence': 0.7
            }
            
        except Exception:
            return self._calculate_trailing_stop(entry_price)

    def _calculate_time_stop(self, entry_price: float) -> Dict:
        """Calculate time-based stop loss"""
        
        # Time stop uses trailing stop price but different strategy
        trailing_data = self._calculate_trailing_stop(entry_price)
        
        return {
            'stop_price': trailing_data['stop_price'],
            'stop_percent': trailing_data['stop_percent'],
            'time_limit_days': config.TIME_STOP_DAYS,
            'strategy': 'time',
            'confidence': 0.6
        }

    def _select_optimal_stop(self, strategies: Dict, entry_price: float) -> Dict:
        """Select optimal stop from available strategies"""
        
        # Filter valid strategies
        valid_strategies = {k: v for k, v in strategies.items() if 'stop_price' in v}
        
        if not valid_strategies:
            return self._calculate_trailing_stop(entry_price)
        
        # Score strategies based on priority and risk-reward
        best_strategy = None
        best_score = -1
        
        for strategy_name, strategy_data in valid_strategies.items():
            # Get priority (lower number = higher priority)
            priority = config.STOP_LOSS_STRATEGY_PRIORITY.get(strategy_name, 5)
            confidence = strategy_data.get('confidence', 0.5)
            
            # Calculate risk-reward (assuming 2:1 target)
            stop_price = strategy_data['stop_price']
            risk = entry_price - stop_price
            reward = risk * 2  # Assume 2:1 ratio
            risk_reward = reward / risk if risk > 0 else 0
            
            # Score: higher is better
            score = confidence * (1 / priority) * min(risk_reward / 2, 1)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_data.copy()
                best_strategy['strategy'] = strategy_name
                best_strategy['risk_reward'] = risk_reward
        
        return best_strategy or self._calculate_trailing_stop(entry_price)

    def _apply_profit_protection(self, entry_price: float, current_price: float, 
                            current_stop: float, profit_pct: float) -> float:
        """Apply profit protection logic"""
        
        new_stop = current_stop
        
        for profit_threshold, stop_profit_pct in config.PROFIT_PROTECTION_LEVELS.items():
            if profit_pct >= profit_threshold:
                protected_stop = entry_price * (1 + stop_profit_pct / 100)
                new_stop = max(new_stop, protected_stop)
        
        return new_stop

    def _update_trailing_stop(self, entry_price: float, current_price: float, current_stop: float) -> float:
        """Update trailing stop based on current price"""
        
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Determine trailing percentage based on profit
        if profit_pct > config.TRAILING_STOP_TIGHTENING_THRESHOLD:
            # Tighten stop as profit increases
            trailing_pct = max(
                config.TRAILING_STOP_MIN_PERCENT,
                config.TRAILING_STOP_INITIAL_PERCENT - (profit_pct - config.TRAILING_STOP_TIGHTENING_THRESHOLD) * 0.3
            )
        else:
            trailing_pct = config.TRAILING_STOP_INITIAL_PERCENT
        
        trailing_stop = current_price * (1 - trailing_pct / 100)
        return max(current_stop, trailing_stop)

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (simplified)"""
        
        try:
            # Get latest price from database
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=1)
            df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            
            if not df.empty:
                return float(df.iloc[-1]['close'])
            return 0.0
            
        except Exception:
            return 0.0

    def _calculate_true_range(self, row) -> float:
        """Calculate True Range for ATR"""
        
        high = row['high']
        low = row['low']
        prev_close = row['close']  # Simplified
        
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    def _estimate_portfolio_volatility(self, positions: List[Dict]) -> float:
        """Estimate portfolio volatility (simplified)"""
        
        try:
            # Simple equal-weighted volatility estimate
            total_weight = 0
            weighted_vol = 0
            
            for position in positions:
                volatility_cat = position.get('volatility_category', 'Medium')
                weight = position.get('position_value', 0)
                
                # Estimate volatility by category
                vol_mapping = {'Low': 0.15, 'Medium': 0.25, 'High': 0.40}
                daily_vol = vol_mapping.get(volatility_cat, 0.25) / (252 ** 0.5)  # Daily volatility
                
                weighted_vol += daily_vol * weight
                total_weight += weight
            
            return weighted_vol / total_weight if total_weight > 0 else 0.02
            
        except Exception:
            return 0.02  # Default 2% daily volatility

    def _apply_weight_constraints(self, weights: Dict, max_weight: float) -> Dict:
        """Apply maximum weight constraints to portfolio"""
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Apply max weight constraint
        constrained_weights = {}
        excess_weight = 0
        
        for symbol, weight in normalized_weights.items():
            if weight > max_weight:
                constrained_weights[symbol] = max_weight
                excess_weight += weight - max_weight
            else:
                constrained_weights[symbol] = weight
        
        # Redistribute excess weight proportionally
        if excess_weight > 0:
            eligible_symbols = [s for s, w in constrained_weights.items() if w < max_weight]
            if eligible_symbols:
                for symbol in eligible_symbols:
                    additional_weight = excess_weight / len(eligible_symbols)
                    constrained_weights[symbol] = min(
                        constrained_weights[symbol] + additional_weight, 
                        max_weight
                    )
        
        return constrained_weights

    def _get_stop_update_reason(self, profit_pct: float) -> str:
        """Get reason for stop update"""
        
        if profit_pct >= config.BREAKEVEN_PROFIT_THRESHOLD:
            return 'profit_protection'
        elif profit_pct > 0:
            return 'trailing_stop'
        else:
            return 'dynamic_adjustment'
    def validate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                               target_price: float) -> bool:
        """Validate risk-reward ratio meets minimum requirement"""
        if entry_price <= 0 or stop_loss <= 0 or target_price <= 0:
            return False
        
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        if risk == 0:
            return False
        
        ratio = reward / risk
        return ratio >= config.MIN_RISK_REWARD_RATIO
    