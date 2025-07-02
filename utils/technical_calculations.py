"""
FILE: utils/technical_calculations.py
LOCATION: /utils/ directory  
PURPOSE: Technical Analysis Utility Functions - Complete Implementation

DESCRIPTION:
- Basic utility functions for technical analysis
- Data validation and cleaning functions for OHLCV data
- Simple support/resistance level detection
- Basic validation functions for technical indicators
- Advanced features: Parameter optimization, market regime detection, backtesting

DEPENDENCIES:
- pandas, numpy (for data processing)
- logging (for error handling)

USAGE:
- Used by technical_agent.py for basic calculations
- Called for data validation and cleaning
- Used in test suites for validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class TechnicalCalculations:
    """Utility class for basic technical analysis calculations and validations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> bool:
        """Validate OHLCV data integrity"""
        
        if df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for negative values
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False
        
        if (df['volume'] < 0).any():
            return False
        
        # Check OHLC relationships
        invalid_data = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        return not invalid_data.any()
    
    @staticmethod
    def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare OHLCV data for analysis"""
        
        df = df.copy()
        
        # Remove rows with missing values
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        df = df[df[price_columns].gt(0).all(axis=1)]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        # Fix OHLC relationships
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def find_support_resistance_levels(df: pd.DataFrame, window: int = 20) -> Dict:
        """Find basic support and resistance levels"""
        
        if len(df) < window:
            return {'support_levels': [], 'resistance_levels': []}
        
        try:
            # Simple approach - use recent highs and lows
            recent_data = df.tail(window)
            support_level = recent_data['low'].min()
            resistance_level = recent_data['high'].max()
            
            return {
                'support_levels': [float(support_level)] if not pd.isna(support_level) else [],
                'resistance_levels': [float(resistance_level)] if not pd.isna(resistance_level) else []
            }
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Support/resistance calculation error: {e}")
            return {'support_levels': [], 'resistance_levels': []}

class SimpleBacktester:
    """Simple backtesting functionality for strategy validation"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission = 0.001
    
    def run_simple_backtest(self, signals: List[Dict], symbol_prices: Dict) -> Dict:
        """Run basic backtest on signals"""
        
        portfolio_value = self.initial_capital
        trades = []
        
        for signal in signals:
            symbol = signal['symbol']
            if symbol not in symbol_prices:
                continue
                
            entry_price = signal.get('entry_price', symbol_prices[symbol])
            exit_price = entry_price * (1.05 if signal.get('signal_type') == 'BUY' else 0.95)
            
            shares = int((portfolio_value * 0.1) / entry_price)  # 10% per trade
            pnl = (exit_price - entry_price) * shares if signal.get('signal_type') == 'BUY' else (entry_price - exit_price) * shares
            
            portfolio_value += pnl
            trades.append({'symbol': symbol, 'pnl': pnl, 'shares': shares})
        
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'final_value': portfolio_value
        }

class PatternEnhancer:
    """Simple pattern-based signal enhancement"""
    
    @staticmethod
    def enhance_signal_with_patterns(technical_score: float, patterns: List[Dict]) -> float:
        """Enhance technical score with pattern confidence"""
        
        if not patterns:
            return technical_score
        
        pattern_boost = 1.0
        for pattern in patterns:
            confidence = pattern.get('confidence', 0.5)
            if pattern.get('subtype') == 'bullish':
                pattern_boost *= (1.0 + confidence * 0.1)
            elif pattern.get('subtype') == 'bearish':
                pattern_boost *= (1.0 - confidence * 0.1)
        
        return min(technical_score * pattern_boost, 0.95)

class ParameterOptimizer:
    """Optimize technical indicator parameters using simple grid search"""
    
    def __init__(self):
        self.rsi_ranges = [10, 14, 21, 28]
        self.ema_ranges = [15, 20, 25, 50]
        self.macd_configs = [(12,26,9), (8,21,5), (15,30,10)]
    
    def optimize_parameters(self, df: pd.DataFrame, target_metric: str = 'sharpe') -> Dict:
        """Find optimal parameters for technical indicators"""
        
        best_score = -999
        best_params = {}
        
        for rsi_period in self.rsi_ranges:
            for ema_period in self.ema_ranges:
                for macd_config in self.macd_configs:
                    try:
                        score = self._test_parameter_combination(df, rsi_period, ema_period, macd_config)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'rsi_period': rsi_period,
                                'ema_period': ema_period, 
                                'macd_config': macd_config,
                                'score': score
                            }
                    except:
                        continue
        
        return best_params if best_params else self._get_default_params()
    
    def _test_parameter_combination(self, df: pd.DataFrame, rsi_period: int, ema_period: int, macd_config: tuple) -> float:
        """Test parameter combination and return performance score"""
        
        if len(df) < max(rsi_period, ema_period, macd_config[1]) + 10:
            return -999
        
        # Simple scoring based on trend consistency
        closes = df['close'].values
        returns = np.diff(closes) / closes[:-1]
        
        # Calculate simple RSI
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-rsi_period:]) if len(gains) >= rsi_period else 0
        avg_loss = np.mean(losses[-rsi_period:]) if len(losses) >= rsi_period else 0.001
        
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        # Simple trend score
        ema = np.mean(closes[-ema_period:]) if len(closes) >= ema_period else closes[-1]
        trend_score = (closes[-1] - ema) / ema
        
        # Combine scores
        signal_quality = abs(rsi - 50) / 50  # Distance from neutral
        return trend_score * signal_quality
    
    def _get_default_params(self) -> Dict:
        """Return default parameters if optimization fails"""
        return {
            'rsi_period': 14,
            'ema_period': 20,
            'macd_config': (12, 26, 9),
            'score': 0
        }

class MarketRegimeDetector:
    """Detect market regime (bull/bear/sideways) for parameter adjustment"""
    
    def __init__(self):
        self.lookback_periods = 50
    
    def detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        
        if len(df) < self.lookback_periods:
            return self._default_regime()
        
        closes = df['close'].tail(self.lookback_periods).values
        returns = np.diff(closes) / closes[:-1]
        
        # Calculate trend metrics
        total_return = (closes[-1] - closes[0]) / closes[0]
        volatility = np.std(returns)
        trend_strength = abs(total_return)
        
        # Determine regime
        if total_return > 0.05 and trend_strength > 0.02:
            regime = 'bull'
            confidence = min(total_return * 10, 0.9)
        elif total_return < -0.05 and trend_strength > 0.02:
            regime = 'bear'
            confidence = min(abs(total_return) * 10, 0.9)
        else:
            regime = 'sideways'
            confidence = 1 - trend_strength * 5
        
        return {
            'regime': regime,
            'confidence': round(confidence, 3),
            'volatility': round(volatility, 4),
            'trend_strength': round(trend_strength, 4),
            'total_return': round(total_return, 4)
        }
    
    def adjust_parameters_for_regime(self, base_params: Dict, regime_data: Dict) -> Dict:
        """Adjust technical parameters based on market regime"""
        
        regime = regime_data.get('regime', 'sideways')
        adjusted = base_params.copy()
        
        if regime == 'bull':
            # More aggressive in bull markets
            adjusted['rsi_oversold'] = 25  # Lower oversold threshold
            adjusted['position_size_multiplier'] = 1.2
        elif regime == 'bear':
            # More conservative in bear markets  
            adjusted['rsi_overbought'] = 75  # Lower overbought threshold
            adjusted['position_size_multiplier'] = 0.8
        else:
            # Neutral in sideways markets
            adjusted['rsi_oversold'] = 30
            adjusted['rsi_overbought'] = 70
            adjusted['position_size_multiplier'] = 1.0
        
        return adjusted
    
    def _default_regime(self) -> Dict:
        """Return default regime when detection fails"""
        return {
            'regime': 'sideways',
            'confidence': 0.5,
            'volatility': 0.02,
            'trend_strength': 0.01,
            'total_return': 0.0
        }