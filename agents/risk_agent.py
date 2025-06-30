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
from datetime import datetime
import pytz

import config

class RiskAgent:
    """Risk management and position sizing agent"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.ist = pytz.timezone('Asia/Kolkata')
    
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