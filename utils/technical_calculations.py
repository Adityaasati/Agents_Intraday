"""
FILE: utils/technical_calculations.py
LOCATION: /utils/ directory  
PURPOSE: Technical Analysis Utility Functions - Helper functions for advanced technical analysis

DESCRIPTION:
- Utility class containing advanced technical analysis calculations
- Data validation and cleaning functions for OHLCV data
- Support/resistance level detection algorithms
- Chart pattern recognition (head & shoulders, double top/bottom)
- Volume profile analysis and market strength indicators
- Validation functions for technical indicators

DEPENDENCIES:
- pandas, numpy (for data processing)
- logging (for error handling)

USAGE:
- Used by technical_agent.py for advanced calculations
- Called for data validation and cleaning
- Provides pattern recognition capabilities
- Used in test suites for validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class TechnicalCalculations:
    """Utility class for technical analysis calculations and validations"""
    
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
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate price returns"""
        return prices.pct_change(periods=periods)
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = TechnicalCalculations.calculate_returns(prices)
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def detect_price_gaps(df: pd.DataFrame, threshold_percent: float = 2.0) -> pd.Series:
        """Detect price gaps between sessions"""
        
        if len(df) < 2:
            return pd.Series(dtype=bool)
        
        gap_up = (df['open'] / df['close'].shift(1) - 1) * 100 > threshold_percent
        gap_down = (df['open'] / df['close'].shift(1) - 1) * 100 < -threshold_percent
        
        return gap_up | gap_down
    
    @staticmethod
    def find_support_resistance_levels(df: pd.DataFrame, window: int = 20, 
                                     min_touches: int = 2) -> Dict:
        """Find dynamic support and resistance levels"""
        
        if len(df) < window * 2:
            return {'support_levels': [], 'resistance_levels': []}
        
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Find local highs and lows
        local_highs = df['high'] == highs
        local_lows = df['low'] == lows
        
        # Get resistance levels (local highs)
        resistance_prices = df.loc[local_highs, 'high'].values
        resistance_levels = TechnicalCalculations._cluster_levels(
            resistance_prices, min_touches
        )
        
        # Get support levels (local lows)
        support_prices = df.loc[local_lows, 'low'].values
        support_levels = TechnicalCalculations._cluster_levels(
            support_prices, min_touches
        )
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    @staticmethod
    def _cluster_levels(prices: np.ndarray, min_touches: int = 2, 
                       tolerance_percent: float = 1.0) -> List[float]:
        """Cluster price levels that are close to each other"""
        
        if len(prices) == 0:
            return []
        
        # Remove NaN values
        prices = prices[~np.isnan(prices)]
        if len(prices) == 0:
            return []
        
        levels = []
        prices_sorted = np.sort(prices)
        
        i = 0
        while i < len(prices_sorted):
            current_price = prices_sorted[i]
            if current_price <= 0:  # Skip invalid prices
                i += 1
                continue
                
            cluster = [current_price]
            
            # Find all prices within tolerance
            j = i + 1
            while j < len(prices_sorted):
                price_diff = abs(prices_sorted[j] - current_price) / current_price * 100
                if price_diff <= tolerance_percent:
                    cluster.append(prices_sorted[j])
                    j += 1
                else:
                    break
            
            # If cluster has enough touches, add average as support/resistance level
            if len(cluster) >= min_touches:
                levels.append(float(np.mean(cluster)))
            
            i = j if j > i else i + 1
        
        return levels
    
    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float) -> Dict:
        """Calculate pivot points for intraday trading"""
        
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'resistance_1': 2 * pivot - low,
            'support_1': 2 * pivot - high,
            'resistance_2': pivot + (high - low),
            'support_2': pivot - (high - low),
            'resistance_3': high + 2 * (pivot - low),
            'support_3': low - 2 * (high - pivot)
        }
    
    @staticmethod
    def identify_chart_patterns(df: pd.DataFrame) -> List[Dict]:
        """Identify basic chart patterns with robust error handling"""
        
        patterns = []
        
        if df.empty or len(df) < 10:
            return patterns
        
        try:
            # Head and Shoulders pattern (simplified)
            if len(df) >= 20:
                recent_highs = df['high'].tail(20)
                if len(recent_highs) >= 5:
                    max_idx = recent_highs.idxmax()
                    
                    # Calculate shoulders with bounds checking
                    left_shoulder_data = recent_highs.loc[:max_idx-3] if max_idx >= 3 else pd.Series()
                    right_shoulder_data = recent_highs.loc[max_idx+3:] if max_idx < len(recent_highs)-3 else pd.Series()
                    
                    left_shoulder = left_shoulder_data.max() if not left_shoulder_data.empty else 0
                    right_shoulder = right_shoulder_data.max() if not right_shoulder_data.empty else 0
                    head = recent_highs.loc[max_idx]
                    
                    if left_shoulder > 0 and right_shoulder > 0 and head > 0:
                        shoulder_avg = (left_shoulder + right_shoulder) / 2
                        if (head > shoulder_avg * 1.02 and 
                            abs(left_shoulder - right_shoulder) / shoulder_avg < 0.05):
                            patterns.append({
                                'pattern': 'head_and_shoulders',
                                'type': 'bearish',
                                'confidence': 0.7,
                                'target': round(shoulder_avg, 2)
                            })
            
            # Double Top/Bottom patterns (simplified)
            if len(df) >= 20:
                highs = df['high'].tail(20)
                
                # Use rolling max to find peaks
                if len(highs) >= 3:
                    peak_mask = highs.rolling(window=3, center=True).max() == highs
                    peak_values = highs[peak_mask].values
                    
                    if len(peak_values) >= 2:
                        last_two_peaks = peak_values[-2:]
                        peak_avg = np.mean(last_two_peaks)
                        if peak_avg > 0 and abs(last_two_peaks[0] - last_two_peaks[1]) / peak_avg < 0.02:
                            patterns.append({
                                'pattern': 'double_top',
                                'type': 'bearish',
                                'confidence': 0.6,
                                'target': round(peak_avg * 0.95, 2)
                            })
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Pattern identification error: {e}")
        
        return patterns
    
    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> Dict:
        """Calculate additional momentum indicators"""
        
        indicators = {}
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Stochastic Oscillator
            if len(df) >= 14:
                lowest_low = low.rolling(window=14).min()
                highest_high = high.rolling(window=14).max()
                k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                d_percent = k_percent.rolling(window=3).mean()
                
                indicators['stoch_k'] = k_percent.iloc[-1] if not k_percent.empty else None
                indicators['stoch_d'] = d_percent.iloc[-1] if not d_percent.empty else None
            
            # Williams %R
            if len(df) >= 14:
                highest_high_14 = high.rolling(window=14).max()
                lowest_low_14 = low.rolling(window=14).min()
                williams_r = -100 * ((highest_high_14 - close) / (highest_high_14 - lowest_low_14))
                
                indicators['williams_r'] = williams_r.iloc[-1] if not williams_r.empty else None
            
            # Rate of Change (ROC)
            if len(df) >= 12:
                roc = ((close / close.shift(12)) - 1) * 100
                indicators['roc_12'] = roc.iloc[-1] if not roc.empty else None
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Momentum indicators calculation error: {e}")
        
        return indicators
    
    @staticmethod
    def analyze_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
        """Analyze volume profile for price levels"""
        
        if len(df) < 10:
            return {}
        
        try:
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # Calculate volume at each price level
            volume_profile = {}
            total_volume = 0
            
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                bin_center = (bin_low + bin_high) / 2
                
                # Find bars where price traded in this range
                in_range = (df['low'] <= bin_high) & (df['high'] >= bin_low)
                bin_volume = df.loc[in_range, 'volume'].sum()
                
                volume_profile[bin_center] = bin_volume
                total_volume += bin_volume
            
            # Find VWAP and high volume nodes
            if total_volume > 0:
                vwap = sum(price * volume for price, volume in volume_profile.items()) / total_volume
                max_volume_price = max(volume_profile.keys(), key=lambda x: volume_profile[x])
                
                return {
                    'vwap': vwap,
                    'poc': max_volume_price,  # Point of Control
                    'volume_profile': volume_profile
                }
        
        except Exception as e:
            logging.getLogger(__name__).warning(f"Volume profile analysis error: {e}")
        
        return {}
    
    @staticmethod
    def calculate_market_strength(df: pd.DataFrame) -> Dict:
        """Calculate overall market strength indicators"""
        
        if len(df) < 20:
            return {}
        
        try:
            close = df['close']
            volume = df['volume']
            
            # Accumulation/Distribution Line
            money_flow_multiplier = ((close - df['low']) - (df['high'] - close)) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * volume
            ad_line = money_flow_volume.cumsum()
            
            # On Balance Volume
            price_change = close.diff()
            obv_direction = np.where(price_change > 0, volume, 
                                   np.where(price_change < 0, -volume, 0))
            obv = pd.Series(obv_direction).cumsum()
            
            # Volume Price Trend
            vpt = ((close.pct_change()) * volume).cumsum()
            
            return {
                'ad_line': ad_line.iloc[-1] if not ad_line.empty else None,
                'obv': obv.iloc[-1] if not obv.empty else None,
                'vpt': vpt.iloc[-1] if not vpt.empty else None,
                'volume_trend': 'increasing' if volume.tail(5).mean() > volume.tail(10).mean() else 'decreasing'
            }
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Market strength calculation error: {e}")
            return {}
    
    @staticmethod
    def validate_technical_indicators(indicators: Dict) -> Dict:
        """Validate and clean technical indicators"""
        
        validated = {}
        
        for key, value in indicators.items():
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                continue
            
            # Round numeric values for consistency
            if isinstance(value, (int, float)):
                if key.endswith('_ratio') or key.endswith('_score'):
                    validated[key] = round(float(value), 4)
                elif 'price' in key or 'level' in key:
                    validated[key] = round(float(value), 2)
                else:
                    validated[key] = round(float(value), 3)
            else:
                validated[key] = value
        
        return validated
    
    @staticmethod
    def get_indicator_status(indicators: Dict, symbol: str) -> Dict:
        """Get status summary of technical indicators"""
        
        status = {
            'symbol': symbol,
            'indicators_calculated': len(indicators),
            'rsi_status': 'unknown',
            'trend_status': 'unknown',
            'volume_status': 'unknown',
            'momentum_status': 'unknown',
            'overall_health': 'unknown'
        }
        
        try:
            # RSI status
            rsi_14 = indicators.get('rsi_14')
            if rsi_14:
                if rsi_14 < 30:
                    status['rsi_status'] = 'oversold'
                elif rsi_14 > 70:
                    status['rsi_status'] = 'overbought'
                else:
                    status['rsi_status'] = 'neutral'
            
            # Trend status
            ma_trend = indicators.get('ma_trend')
            if ma_trend:
                status['trend_status'] = ma_trend
            
            # Volume status
            volume_signal = indicators.get('volume_signal')
            if volume_signal:
                status['volume_status'] = volume_signal
            
            # Momentum status
            macd_signal_type = indicators.get('macd_signal_type')
            if macd_signal_type:
                status['momentum_status'] = macd_signal_type
            
            # Overall health
            technical_score = indicators.get('technical_score')
            if technical_score:
                if technical_score >= 0.7:
                    status['overall_health'] = 'strong'
                elif technical_score >= 0.5:
                    status['overall_health'] = 'moderate'
                else:
                    status['overall_health'] = 'weak'
        
        except Exception as e:
            logging.getLogger(__name__).warning(f"Indicator status calculation error: {e}")
        
        return status

# Convenience functions for external use
def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLCV data"""
    return TechnicalCalculations.clean_ohlcv_data(df)

def get_support_resistance(df: pd.DataFrame) -> Dict:
    """Get support and resistance levels"""
    return TechnicalCalculations.find_support_resistance_levels(df)

def analyze_chart_patterns(df: pd.DataFrame) -> List[Dict]:
    """Analyze chart patterns"""
    return TechnicalCalculations.identify_chart_patterns(df)

def calculate_additional_indicators(df: pd.DataFrame) -> Dict:
    """Calculate additional momentum indicators"""
    return TechnicalCalculations.calculate_momentum_indicators(df)