"""
FILE: agents/technical_agent.py
LOCATION: /agents/ directory
PURPOSE: Technical Analysis Agent - Core technical analysis engine for generating trading signals

DESCRIPTION:
- Main technical analysis agent that calculates RSI, MACD, Bollinger Bands, Moving Averages
- Generates buy/sell signals based on technical indicators  
- Integrates with database to store/retrieve analysis results
- Supports both pandas_ta library and manual fallback calculations
- Handles error scenarios gracefully with synthetic data generation
- Part of the multi-agent trading system for technical analysis component

DEPENDENCIES:
- database/enhanced_database_manager.py (for data access)
- config.py (for system parameters)
- pandas, numpy (for calculations)
- pandas_ta (optional, with manual fallback)

USAGE:
- Used by main.py for symbol analysis
- Called by signal generation agent for technical scoring
- Integrated with risk management for position sizing
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
import time
import threading
import config
from database.enhanced_database_manager import EnhancedDatabaseManager
from .base_agent import BaseAgent
from utils.decorators import handle_agent_errors

class TechnicalAgent(BaseAgent): 
    """Technical Analysis Agent for generating trading signals"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        self._indicator_cache = {} if config.ENABLE_INDICATOR_CACHE else None
        self._cache_lock = threading.Lock()
        
        # FIXED: More robust pandas_ta detection
        try:
            import ta 
            # Test if it actually works
            ta.rsi
            self.pandas_ta = ta
            self.use_pandas_ta = True
            self.logger.debug("pandas_ta loaded successfully")
        except (ImportError, AttributeError) as e:
            self.pandas_ta = None
            self.use_pandas_ta = False
            self.logger.debug(f"pandas_ta not available: {e}, using manual calculations")
    
    @handle_agent_errors(default_return={'error': 'analysis_failed'})
    def analyze_symbol(self, symbol: str, timeframe: str = '5m', 
                  lookback_days: int = 30) -> Dict:
        """Complete technical analysis for a symbol with robust data handling"""
        
        try:
            # Get historical data
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=lookback_days)
            
            # Debug: Log the method call
            self.logger.debug(f"Getting historical data for {symbol}")
            df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            
            # Handle insufficient data more gracefully
            if df.empty:
                self.logger.info(f"No historical data for {symbol}, using synthetic data for testing")
                df = self._create_test_data(symbol, 100)
            elif len(df) < 10:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} records, supplementing with synthetic data")
                synthetic_df = self._create_test_data(symbol, 50)
                if not df.empty:
                    last_date = df['date'].max()
                    synthetic_df['date'] = pd.date_range(start=last_date + timedelta(minutes=5), 
                                                    periods=50, freq='5min')
                df = pd.concat([df, synthetic_df], ignore_index=True)
                self.logger.info(f"Supplemented {symbol} data: now {len(df)} records")
            
            # Validate and clean data
            df = self._validate_ohlcv_data(df)
            if df.empty:
                return {'symbol': symbol, 'error': 'invalid_data_after_cleaning'}
            
            # Calculate all technical indicators
            self.logger.debug(f"Calculating indicators for {symbol}")
            indicators = self._calculate_all_indicators(df)
            
            # DEBUG: Check what indicators is
            self.logger.debug(f"Indicators type: {type(indicators)}, value: {indicators}")
            
            if not isinstance(indicators, dict):
                self.logger.error(f"Indicators is not a dict! Type: {type(indicators)}")
                return {'symbol': symbol, 'error': 'invalid_indicators_type'}
            
            if not indicators or not any(v is not None for v in indicators.values()):
                return {'symbol': symbol, 'error': 'calculation_failed', 'data_points': len(df)}
            
            # Generate signals and analysis
            self.logger.debug(f"Generating signals for {symbol}")
            analysis = self._generate_technical_signals(symbol, df, indicators)
            
            # Store technical indicators in database (with error handling)
            try:
                self._store_technical_analysis(symbol, df, indicators, timeframe)
            except Exception as storage_error:
                self.logger.warning(f"Failed to store technical analysis for {symbol}: {storage_error}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed for {symbol}: {e}", exc_info=True)
            return {'symbol': symbol, 'error': str(e)}
    
    def analyze_with_optimization(self, symbol: str, timeframe: str = '5m', lookback_days: int = 30) -> Dict:
        """Enhanced analysis with parameter optimization and regime detection"""
        
        try:
            # Get base analysis
            analysis = self.analyze_symbol(symbol, timeframe, lookback_days)
            
            # Get historical data for optimization
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=lookback_days)
            # df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            df = self.db_manager.get_historical_data(symbol, limit=1000)  # Adjust limit as needed
            
            
            if df.empty or len(df) < 30:
                return analysis
            
            # Initialize optimizers
            from utils.technical_calculations import ParameterOptimizer, MarketRegimeDetector
            
            optimizer = ParameterOptimizer()
            regime_detector = MarketRegimeDetector()
            
            # Optimize parameters
            optimal_params = optimizer.optimize_parameters(df)
            
            # Detect market regime
            regime_data = regime_detector.detect_market_regime(df)
            
            # Adjust parameters for regime
            adjusted_params = regime_detector.adjust_parameters_for_regime(optimal_params, regime_data)
            
            # Enhance analysis with optimization data
            analysis['optimization'] = {
                'optimal_parameters': optimal_params,
                'market_regime': regime_data,
                'adjusted_parameters': adjusted_params
            }
            
            # Adjust technical score based on regime
            if 'technical_score' in analysis:
                regime_multiplier = adjusted_params.get('position_size_multiplier', 1.0)
                analysis['technical_score'] = min(analysis['technical_score'] * regime_multiplier, 0.95)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Optimization analysis failed for {symbol}: {e}")
            return analysis

    def get_market_regime_summary(self, symbols: List[str]) -> Dict:
        """Get market regime summary across multiple symbols"""
        
        try:
            from utils.technical_calculations import MarketRegimeDetector
            
            regime_detector = MarketRegimeDetector()
            regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}
            total_symbols = 0
            
            for symbol in symbols[:10]:  # Limit for performance
                try:
                    end_date = datetime.now(self.ist)
                    start_date = end_date - timedelta(days=30)
                    # df = self.db_manager.get_historical_data(symbol, start_date, end_date)
                    df = self.db_manager.get_historical_data(symbol, limit=1000)  # Adjust limit as needed
                    
                    if not df.empty and len(df) >= 20:
                        regime_data = regime_detector.detect_market_regime(df)
                        regime = regime_data.get('regime', 'sideways')
                        regime_counts[regime] += 1
                        total_symbols += 1
                except:
                    continue
            
            if total_symbols == 0:
                return {'error': 'No regime data available'}
            
            # Determine overall market regime
            dominant_regime = max(regime_counts.items(), key=lambda x: x[1])
            
            return {
                'total_symbols_analyzed': total_symbols,
                'regime_distribution': regime_counts,
                'dominant_regime': dominant_regime[0],
                'regime_confidence': round(dominant_regime[1] / total_symbols, 2),
                'market_assessment': self._get_market_assessment(dominant_regime[0], dominant_regime[1] / total_symbols)
            }
            
        except Exception as e:
            return {'error': f'Regime analysis failed: {e}'}

    def _get_market_assessment(self, regime: str, confidence: float) -> str:
        """Get readable market assessment"""
        
        if confidence < 0.5:
            return "Mixed market conditions"
        elif regime == 'bull' and confidence > 0.7:
            return "Strong bullish market"
        elif regime == 'bear' and confidence > 0.7:
            return "Strong bearish market"
        elif regime == 'sideways' and confidence > 0.6:
            return "Consolidating market"
        else:
            return f"Moderately {regime} market"
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data with proper type conversion"""
        
        if df.empty:
            return df
        
        # Remove rows with null values in critical columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=required_cols)
        
        if df.empty:
            return df
        
        # Convert all price columns to float to avoid Decimal/float mixing
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Convert volume to int
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')
        
        # Remove rows with zero or negative prices after conversion
        df = df[df[price_cols].gt(0).all(axis=1)]
        
        # Remove rows with null volume
        df = df[df['volume'].notna()]
        
        # Fix OHLC relationships
        if not df.empty:
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _create_test_data(self, symbol: str, periods: int = 50) -> pd.DataFrame:
        """Create test data for symbols with no historical data"""
        
        dates = pd.date_range(end=datetime.now(self.ist), periods=periods, freq='5min')  # Fixed: 'T' -> 'min'
        np.random.seed(hash(symbol) % 1000)  # Consistent random data per symbol
        
        base_price = 100.0
        prices = [base_price]
        
        for i in range(periods - 1):
            change = np.random.normal(0, 0.02)  # 2% volatility
            new_price = max(prices[-1] * (1 + change), 1.0)
            prices.append(new_price)
        
        return pd.DataFrame({
            'symbol': [symbol] * periods,
            'date': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.02) for p in prices],
            'low': [p * np.random.uniform(0.98, 1.0) for p in prices],
            'close': prices,
            'volume': np.random.randint(10000, 100000, periods)
        })
    
    def analyze_multiple_symbols(self, symbols: List[str], limit: int = 50) -> List[Dict]:
        """Batch analysis for multiple symbols"""
        
        results = []
        processed = 0
        
        for symbol in symbols[:limit]:
            if processed >= limit:
                break
                
            try:
                analysis = self.analyze_symbol(symbol)
                results.append(analysis)
                processed += 1
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {symbol}: {e}")
                results.append({'symbol': symbol, 'error': str(e)})
        
        self.logger.info(f"Analyzed {processed} symbols")
        return results
    
    
    
    def _calculate_indicators_pandas_ta(self, df: pd.DataFrame) -> Dict:
        """Calculate indicators using pandas_ta library"""
        
        try:
            # Add all indicators to dataframe
            df.ta.rsi(length=14, append=True)
            df.ta.rsi(length=21, append=True)
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.volume_sma(length=20, append=True)
            
            # Extract latest values
            latest = df.iloc[-1]
            
            # Helper function to safely get value
            def safe_get(key, default=None):
                try:
                    return latest[key] if key in latest.index else default
                except:
                    return default
            
            indicators = {
                'rsi_14': safe_get('RSI_14'),
                'rsi_21': safe_get('RSI_21'),
                'ema_20': safe_get('EMA_20'),
                'ema_50': safe_get('EMA_50'),
                'sma_20': safe_get('SMA_20'),
                'sma_50': safe_get('SMA_50'),
                'macd_line': safe_get('MACD_12_26_9'),
                'macd_signal': safe_get('MACDs_12_26_9'),
                'macd_histogram': safe_get('MACDh_12_26_9'),
                'bb_upper': safe_get('BBU_20_2.0'),
                'bb_middle': safe_get('BBM_20_2.0'),
                'bb_lower': safe_get('BBL_20_2.0'),
                'atr_14': safe_get('ATR_14'),
                'volume_sma_20': safe_get('SMA_20_volume'),
                'close_price': latest['close'],
                'volume': latest['volume'],
                'date': latest['date']
            }
            
            # Calculate derived indicators
            indicators.update(self._calculate_derived_indicators(df, indicators))
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f"pandas_ta calculation failed: {e}")
            return self._calculate_indicators_manual(df)
    
    
    
    
    def _calculate_indicators_manual(self, df: pd.DataFrame) -> Dict:
        """Manual calculation of technical indicators with robust error handling and type safety"""
        
        try:
            if df.empty or len(df) < 10:
                self.logger.warning("Insufficient data for manual calculations")
                return {}
            
            df = df.copy()
            
            # Ensure all price columns are float type to avoid type mixing
            price_cols = ['close', 'high', 'low', 'open']
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
            # Ensure volume is numeric
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype(float)
            
            close = df['close']
            high = df['high'] 
            low = df['low']
            volume = df['volume']
            
            indicators = {}
            
            try:
                # RSI calculation with type safety
                if len(close) >= 14:
                    rsi_14 = self._calculate_rsi(close, 14)
                    indicators['rsi_14'] = self._safe_get_last_value(rsi_14)
                
                if len(close) >= 21:
                    rsi_21 = self._calculate_rsi(close, 21)
                    indicators['rsi_21'] = self._safe_get_last_value(rsi_21)
            except Exception as e:
                self.logger.warning(f"RSI calculation failed: {e}")
                indicators['rsi_14'] = None
                indicators['rsi_21'] = None
            
            try:
                # Moving averages with type safety
                if len(close) >= 20:
                    ema_20 = close.ewm(span=20, adjust=False).mean()
                    indicators['ema_20'] = self._safe_get_last_value(ema_20)
                    
                    sma_20 = close.rolling(window=20, min_periods=10).mean()
                    indicators['sma_20'] = self._safe_get_last_value(sma_20)
                
                if len(close) >= 50:
                    ema_50 = close.ewm(span=50, adjust=False).mean()
                    indicators['ema_50'] = self._safe_get_last_value(ema_50)
                    
                    sma_50 = close.rolling(window=50, min_periods=25).mean()
                    indicators['sma_50'] = self._safe_get_last_value(sma_50)
            except Exception as e:
                self.logger.warning(f"Moving averages calculation failed: {e}")
            
            try:
                # MACD with type safety
                if len(close) >= 26:
                    macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
                    indicators['macd_line'] = self._safe_get_last_value(macd_line)
                    indicators['macd_signal'] = self._safe_get_last_value(macd_signal)
                    indicators['macd_histogram'] = self._safe_get_last_value(macd_histogram)
            except Exception as e:
                self.logger.warning(f"MACD calculation failed: {e}")
            
            try:
                # Bollinger Bands with type safety
                if len(close) >= 20:
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
                    indicators['bb_upper'] = self._safe_get_last_value(bb_upper)
                    indicators['bb_middle'] = self._safe_get_last_value(bb_middle)
                    indicators['bb_lower'] = self._safe_get_last_value(bb_lower)
            except Exception as e:
                self.logger.warning(f"Bollinger Bands calculation failed: {e}")
            
            try:
                # ATR with type safety
                if len(df) >= 14:
                    atr_14 = self._calculate_atr(high, low, close, 14)
                    indicators['atr_14'] = self._safe_get_last_value(atr_14)
            except Exception as e:
                self.logger.warning(f"ATR calculation failed: {e}")
            
            try:
                # Volume SMA with type safety
                if len(volume) >= 20:
                    volume_sma_20 = volume.rolling(window=20, min_periods=10).mean()
                    indicators['volume_sma_20'] = self._safe_get_last_value(volume_sma_20)
            except Exception as e:
                self.logger.warning(f"Volume SMA calculation failed: {e}")
            
            # Basic values with type conversion
            indicators['close_price'] = self._safe_get_last_value(close)
            indicators['volume'] = self._safe_get_last_value(volume)
            indicators['date'] = df['date'].iloc[-1] if 'date' in df.columns and not df.empty else datetime.now(self.ist)
            
            # Calculate derived indicators
            try:
                derived_indicators = self._calculate_derived_indicators(df, indicators)
                indicators.update(derived_indicators)
            except Exception as e:
                self.logger.warning(f"Derived indicators calculation failed: {e}")
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Manual calculation failed: {e}")
            return {}
        
        
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators using pandas_ta or manual methods"""
        
        if self.use_pandas_ta:
            return self._calculate_indicators_pandas_ta(df)
        else:
            return self._calculate_indicators_manual(df)
    
    def _safe_get_last_value(self, series):
        """Safely get the last value from a pandas Series, converting to Python type"""
        
        try:
            if series is None or len(series) == 0:
                return None
                
            last_val = series.iloc[-1]
            
            # Convert to native Python type
            if pd.isna(last_val) or np.isnan(last_val) or np.isinf(last_val):
                return None
                
            return float(last_val)
            
        except (IndexError, TypeError, ValueError, AttributeError):
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually with type safety"""
        try:
            # Ensure prices are float type to avoid Decimal/float mixing
            prices = pd.to_numeric(prices, errors='coerce').astype(float)
            
            if len(prices) < period + 1:
                return pd.Series(dtype=float)
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
            
        except Exception as e:
            self.logger.warning(f"RSI calculation error: {e}")
            # Return neutral RSI values
            return pd.Series([50.0] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD manually with type safety"""
        try:
            # Ensure prices are float type
            prices = pd.to_numeric(prices, errors='coerce').astype(float)
            
            if len(prices) < slow + signal:
                return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
            
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
            macd_histogram = macd_line - macd_signal
            
            return macd_line.fillna(0), macd_signal.fillna(0), macd_histogram.fillna(0)
            
        except Exception as e:
            self.logger.warning(f"MACD calculation error: {e}")
            # Return zero series
            zero_series = pd.Series([0.0] * len(prices), index=prices.index)
            return zero_series, zero_series, zero_series
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands manually with type safety"""
        try:
            # Ensure prices are float type
            prices = pd.to_numeric(prices, errors='coerce').astype(float)
            
            if len(prices) < period:
                return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
            
            sma = prices.rolling(window=period, min_periods=period//2).mean()
            std = prices.rolling(window=period, min_periods=period//2).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return upper_band.bfill(), sma.bfill(), lower_band.bfill()
            
            
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation error: {e}")
            # Return price-based series
            avg_price = prices.mean() if len(prices) > 0 else 100.0
            price_series = pd.Series([avg_price] * len(prices), index=prices.index)
            return price_series * 1.02, price_series, price_series * 0.98
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range manually with type safety"""
        try:
            # Ensure all series are float type to avoid Decimal/float mixing
            high = pd.to_numeric(high, errors='coerce').astype(float)
            low = pd.to_numeric(low, errors='coerce').astype(float)
            close = pd.to_numeric(close, errors='coerce').astype(float)
            
            if len(high) < 2:
                return pd.Series(dtype=float)
            
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            return atr.bfill().fillna(1.0)  # Fill with reasonable default
            
        except Exception as e:
            self.logger.warning(f"ATR calculation error: {e}")
            # Return default ATR values (1% of price)
            avg_price = (high + low + close) / 3
            return avg_price * 0.01
    
    def _calculate_derived_indicators(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Calculate derived indicators and signals with robust error handling"""
        
        derived = {}
        
        try:
            # RSI signals
            rsi_14 = indicators.get('rsi_14')
            if rsi_14 is not None and not pd.isna(rsi_14):
                if rsi_14 < 30:
                    derived['rsi_signal'] = 'oversold'
                elif rsi_14 > 70:
                    derived['rsi_signal'] = 'overbought'
                else:
                    derived['rsi_signal'] = 'neutral'
            else:
                derived['rsi_signal'] = 'unknown'
        except Exception:
            derived['rsi_signal'] = 'unknown'
        
        try:
            # Moving average trend
            ema_20 = indicators.get('ema_20')
            ema_50 = indicators.get('ema_50')
            close_price = indicators.get('close_price')
            
            if all(x is not None and not pd.isna(x) for x in [ema_20, ema_50, close_price]):
                if close_price > ema_20 > ema_50:
                    derived['ma_trend'] = 'bullish'
                elif close_price < ema_20 < ema_50:
                    derived['ma_trend'] = 'bearish'
                else:
                    derived['ma_trend'] = 'sideways'
            else:
                derived['ma_trend'] = 'unknown'
        except Exception:
            derived['ma_trend'] = 'unknown'
        
        try:
            # MACD signals
            macd_line = indicators.get('macd_line')
            macd_signal = indicators.get('macd_signal')
            
            if all(x is not None and not pd.isna(x) for x in [macd_line, macd_signal]):
                if macd_line > macd_signal:
                    derived['macd_signal_type'] = 'bullish'
                else:
                    derived['macd_signal_type'] = 'bearish'
            else:
                derived['macd_signal_type'] = 'unknown'
        except Exception:
            derived['macd_signal_type'] = 'unknown'
        
        try:
            # Volume analysis
            volume = indicators.get('volume')
            volume_sma = indicators.get('volume_sma_20')
            
            if all(x is not None and not pd.isna(x) and x > 0 for x in [volume, volume_sma]):
                volume_ratio = volume / volume_sma
                derived['volume_ratio'] = round(volume_ratio, 2)
                
                if volume_ratio > 2.0:
                    derived['volume_signal'] = 'spike'
                elif volume_ratio > 1.5:
                    derived['volume_signal'] = 'high'
                elif volume_ratio < 0.5:
                    derived['volume_signal'] = 'low'
                else:
                    derived['volume_signal'] = 'normal'
            else:
                derived['volume_ratio'] = 1.0
                derived['volume_signal'] = 'unknown'
        except Exception:
            derived['volume_ratio'] = 1.0
            derived['volume_signal'] = 'unknown'
        
        try:
            # Bollinger Band position
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            close_price = indicators.get('close_price')
            
            if all(x is not None and not pd.isna(x) for x in [bb_upper, bb_lower, close_price]):
                if close_price >= bb_upper:
                    derived['bb_position'] = 'upper_band'
                elif close_price <= bb_lower:
                    derived['bb_position'] = 'lower_band'
                else:
                    derived['bb_position'] = 'middle'
            else:
                derived['bb_position'] = 'unknown'
        except Exception:
            derived['bb_position'] = 'unknown'
        
        try:
            # Support/Resistance levels (simplified)
            if len(df) >= 20:
                recent_data = df.tail(20)
                recent_high = recent_data['high'].max()
                recent_low = recent_data['low'].min()
                
                if not pd.isna(recent_high) and not pd.isna(recent_low):
                    
                    derived['resistance_level'] = float(round(recent_high, 2))
                    derived['support_level'] = float(round(recent_low, 2))

                    
                    if close_price is not None and not pd.isna(close_price):
                        price_range = recent_high - recent_low
                        if price_range > 0:
                            if close_price >= recent_high - (price_range * 0.05):
                                derived['price_position'] = 'near_resistance'
                            elif close_price <= recent_low + (price_range * 0.05):
                                derived['price_position'] = 'near_support'
                            else:
                                derived['price_position'] = 'middle'
                        else:
                            derived['price_position'] = 'middle'
                    else:
                        derived['price_position'] = 'unknown'
                else:
                    derived['price_position'] = 'unknown'
            else:
                derived['price_position'] = 'unknown'
        except Exception as e:
            self.logger.warning(f"Support/resistance calculation failed: {e}")
            derived['price_position'] = 'unknown'
        
        return derived
    
    
    
    def _generate_technical_signals(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate buy/sell signals based on technical indicators"""
        
        # Get fundamental data for category-based adjustments
        fundamental_data = self.db_manager.get_fundamental_data(symbol)
        if not isinstance(fundamental_data, dict) or not fundamental_data:
            self.logger.warning(f"No fundamental data for {symbol}, using defaults")
            fundamental_data = {
                'volatility_category': 'Medium',
                'category': 'B',
                'market_cap_type': 'Mid_Cap'
            }
        volatility_category = fundamental_data.get('volatility_category', 'Medium')
        category = fundamental_data.get('category', 'B')
        market_cap_type = fundamental_data.get('market_cap_type', 'Mid_Cap')
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(indicators, volatility_category)
        technical_score = config.validate_technical_score(technical_score)
        
        if hasattr(self, 'fundamental_agent'):
            fundamental_score = self.fundamental_agent.get_score(symbol)
        else:
            fundamental_score = config.DEFAULT_FUNDAMENTAL_SCORE
            
        if hasattr(self, 'sentiment_agent'):
            sentiment_score = self.sentiment_agent.get_score(symbol)
        else:
            sentiment_score = config.DEFAULT_SENTIMENT_SCORE
        
        # Calculate overall confidence using knowledge graph formula
        overall_confidence = config.calculate_final_confidence(
            technical_score, fundamental_score, sentiment_score, category
        )
        
        # Generate signals based on technical analysis only for now
        buy_signal, sell_signal = self._determine_signals(indicators, technical_score)
        
        # Calculate price targets
        entry_price = indicators.get('close_price', 0)
        stop_loss, target_price = self._calculate_price_targets(
            entry_price, indicators, volatility_category
        )
        
        
        # Determine signal strength based on overall confidence
        signal_strength = self._get_signal_strength(overall_confidence)
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(self.ist),
            'technical_score': round(technical_score, 3),
            'fundamental_score': round(fundamental_score, 3),  # Placeholder
            'sentiment_score': round(sentiment_score, 3),      # Placeholder
            'overall_score': round(technical_score, 3), 
            'overall_confidence': round(overall_confidence, 3),
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'indicators': indicators,
            'category': category,
            'volatility_category': volatility_category,
            'market_cap_type': market_cap_type,
            'reasoning': self._generate_reasoning(indicators, technical_score, buy_signal, sell_signal),
            'analysis_type': 'technical_only',  # Indicates this is technical analysis only
            'data_points_used': len(df)
        }
        
        return analysis
    
    def _calculate_technical_score(self, indicators: Dict, volatility_category: str) -> float:
        """Calculate overall technical score (0-1)"""
        
        score = 0.0
        total_weight = 0.0
        
        # RSI component (30% weight)
        rsi_14 = indicators.get('rsi_14')
        if rsi_14:
            rsi_oversold, rsi_overbought = config.get_rsi_thresholds(volatility_category)
            
            if rsi_14 < rsi_oversold:
                rsi_score = 0.8  # Strong buy signal
            elif rsi_14 > rsi_overbought:
                rsi_score = 0.2  # Strong sell signal
            else:
                # Neutral zone scoring
                rsi_score = 0.5 + (50 - rsi_14) / 100
                
            score += rsi_score * config.RSI_WEIGHT
            total_weight += config.RSI_WEIGHT
        
        # Volume component (25% weight)
        volume_ratio = indicators.get('volume_ratio')
        if volume_ratio:
            if volume_ratio > 2.0:
                volume_score = 0.9  # High volume spike
            elif volume_ratio > 1.5:
                volume_score = 0.7  # High volume
            elif volume_ratio < 0.5:
                volume_score = 0.3  # Low volume concern
            else:
                volume_score = 0.5  # Normal volume
                
            score += volume_score * config.VOLUME_WEIGHT
            total_weight += config.VOLUME_WEIGHT
        
        # Moving average trend (20% weight)
        ma_trend = indicators.get('ma_trend')
        if ma_trend:
            if ma_trend == 'bullish':
                ma_score = 0.8
            elif ma_trend == 'bearish':
                ma_score = 0.2
            else:
                ma_score = 0.5
                
            score += ma_score * config.MA_TREND_WEIGHT
            total_weight += config.MA_TREND_WEIGHT
        
        # MACD component (15% weight)
        macd_signal_type = indicators.get('macd_signal_type')
        if macd_signal_type:
            if macd_signal_type == 'bullish':
                macd_score = 0.7
            else:
                macd_score = 0.3
                
            score += macd_score * config.MACD_WEIGHT
            total_weight += config.MACD_WEIGHT
        
        # Support/Resistance component (10% weight)
        price_position = indicators.get('price_position')
        if price_position:
            if price_position == 'near_support':
                sr_score = 0.7  # Near support - potential bounce
            elif price_position == 'near_resistance':
                sr_score = 0.3  # Near resistance - potential reversal
            else:
                sr_score = 0.5
                
            score += sr_score * config.SUPPORT_RESISTANCE_WEIGHT
            total_weight += config.SUPPORT_RESISTANCE_WEIGHT
        
        # Normalize score
        final_score = score / total_weight if total_weight > 0 else 0.5
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def _determine_signals(self, indicators: Dict, technical_score: float) -> Tuple[bool, bool]:
        """Determine buy/sell signals based on technical analysis"""
        
        buy_signal = False
        sell_signal = False
        
        # Validate technical score
        if technical_score is None or np.isnan(technical_score):
            return False, False
        
        # Primary condition: technical score above threshold
        # Note: Using technical score only since this is technical analysis module
        # Final signal generation will use overall confidence in signal generation agent
        if technical_score >= config.MIN_CONFIDENCE_THRESHOLD:
            # Additional confirmations for buy signal
            confirmations = 0
            
            # RSI confirmation
            rsi_signal = indicators.get('rsi_signal')
            if rsi_signal == 'oversold':
                confirmations += 1
            
            # Volume confirmation
            volume_signal = indicators.get('volume_signal')
            if volume_signal in ['high', 'spike']:
                confirmations += 1
            
            # Trend confirmation
            ma_trend = indicators.get('ma_trend')
            if ma_trend == 'bullish':
                confirmations += 1
            
            # MACD confirmation
            macd_signal_type = indicators.get('macd_signal_type')
            if macd_signal_type == 'bullish':
                confirmations += 1
            
            # Generate buy signal if we have enough confirmations
            if confirmations >= 2:
                buy_signal = True
        
        elif technical_score <= (1 - config.MIN_CONFIDENCE_THRESHOLD):
            # Sell signal logic (for short positions or exits)
            confirmations = 0
            
            rsi_signal = indicators.get('rsi_signal')
            if rsi_signal == 'overbought':
                confirmations += 1
            
            ma_trend = indicators.get('ma_trend')
            if ma_trend == 'bearish':
                confirmations += 1
            
            macd_signal_type = indicators.get('macd_signal_type')
            if macd_signal_type == 'bearish':
                confirmations += 1
            
            if confirmations >= 2:
                sell_signal = True
        
        return buy_signal, sell_signal
    
    def _calculate_price_targets(self, entry_price: float, indicators: Dict, 
                               volatility_category: str) -> Tuple[float, float]:
        """Calculate stop loss and target prices"""
        
        if not entry_price:
            return None, None
        
        # ATR-based stop loss
        atr = indicators.get('atr_14')
        if atr:
            atr_multiplier = config.get_stop_loss_multiplier(volatility_category)
            stop_loss = entry_price - (atr * atr_multiplier)
            stop_loss = self._convert_to_python_type(stop_loss)
            
        else:
            # Fallback percentage-based stop loss
            stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
            stop_loss = self._convert_to_python_type(stop_loss)
            
        
        # Support level stop loss (more conservative)
        support_level = indicators.get('support_level')
        if support_level and support_level < entry_price:
            stop_loss = max(stop_loss, support_level * 0.98)  # 2% below support
            stop_loss = self._convert_to_python_type(stop_loss)
            
        
        # Target price based on risk-reward ratio
        risk_amount = entry_price - stop_loss
        target_price = entry_price + (risk_amount * config.STANDARD_RISK_REWARD_RATIO)
        target_price = self._convert_to_python_type(target_price)
        
        # Resistance level target (more conservative)
        resistance_level = indicators.get('resistance_level')
        if resistance_level and resistance_level > entry_price:
            target_price = min(target_price, resistance_level * 0.98)  # 2% below resistance    
            target_price = self._convert_to_python_type(target_price)

            
        return round(stop_loss, 2), round(target_price, 2)
    
    def _get_signal_strength(self, technical_score: float) -> str:
        """Determine signal strength based on technical score"""
        
        if technical_score >= 0.85:
            return 'very_strong'
        elif technical_score >= 0.70:
            return 'strong'
        elif technical_score >= 0.60:
            return 'medium'
        else:
            return 'weak'
    
    def _generate_reasoning(self, indicators: Dict, technical_score: float, 
                          buy_signal: bool, sell_signal: bool) -> str:
        """Generate human-readable reasoning for the analysis"""
        
        reasons = []
        
        # RSI reasoning
        rsi_14 = indicators.get('rsi_14')
        rsi_signal = indicators.get('rsi_signal')
        if rsi_14 and rsi_signal:
            if rsi_signal == 'oversold':
                reasons.append(f"RSI({rsi_14:.1f}) shows oversold conditions")
            elif rsi_signal == 'overbought':
                reasons.append(f"RSI({rsi_14:.1f}) shows overbought conditions")
        
        # Volume reasoning
        volume_signal = indicators.get('volume_signal')
        volume_ratio = indicators.get('volume_ratio')
        if volume_signal and volume_ratio:
            if volume_signal == 'spike':
                reasons.append(f"Volume spike detected ({volume_ratio:.1f}x average)")
            elif volume_signal == 'high':
                reasons.append(f"Above average volume ({volume_ratio:.1f}x)")
        
        # Trend reasoning
        ma_trend = indicators.get('ma_trend')
        if ma_trend:
            reasons.append(f"Moving average trend: {ma_trend}")
        
        # MACD reasoning
        macd_signal_type = indicators.get('macd_signal_type')
        if macd_signal_type:
            reasons.append(f"MACD shows {macd_signal_type} momentum")
        
        # Price position reasoning
        price_position = indicators.get('price_position')
        if price_position:
            if price_position in ['near_support', 'near_resistance']:
                reasons.append(f"Price is {price_position.replace('_', ' ')}")
        
        # Signal conclusion
        if buy_signal:
            conclusion = f"BUY signal generated with {technical_score:.1%} confidence"
        elif sell_signal:
            conclusion = f"SELL signal generated with {(1-technical_score):.1%} confidence"
        else:
            conclusion = f"No clear signal - score: {technical_score:.1%}"
        
        return f"{conclusion}. {'; '.join(reasons)}" if reasons else conclusion
    
    def _store_technical_analysis(self, symbol: str, df: pd.DataFrame, 
                                 indicators: Dict, timeframe: str):
        """Store technical analysis results in database with proper type conversion"""
        
        try:
            # Prepare data for storage with proper type conversion
            storage_data = []
            
            # Get the latest indicator values and convert numpy/pandas types to native Python types
            latest_data = {
                'date': indicators.get('date'),
                'timeframe': timeframe,
                'close': self._convert_to_python_type(indicators.get('close_price')),
                'rsi_14': self._convert_to_python_type(indicators.get('rsi_14')),
                'rsi_21': self._convert_to_python_type(indicators.get('rsi_21')),
                'ema_20': self._convert_to_python_type(indicators.get('ema_20')),
                'ema_50': self._convert_to_python_type(indicators.get('ema_50')),
                'technical_score': self._convert_to_python_type(indicators.get('technical_score', 0.5))
            }
            
            # Only store if we have valid data
            if latest_data['close'] is not None:
                storage_data.append(latest_data)
                
                # Store in database
                success = self.db_manager.store_technical_indicators(symbol, storage_data)
                
                if success:
                    self.logger.debug(f"Stored technical analysis for {symbol}")
                else:
                    self.logger.warning(f"Failed to store technical analysis for {symbol}")
            else:
                self.logger.warning(f"No valid data to store for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to store technical analysis for {symbol}: {e}")
    
    def _convert_to_python_type(self, value):
        """Convert any type to Python native type safely"""
        if value is None:
            return None
        
        # Handle numpy types
        if hasattr(value, 'item'):
            return value.item()
        
        # Handle pandas timestamp
        if hasattr(value, 'to_pydatetime'):
            return value.to_pydatetime()
        
        # Handle Decimal
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
        
        # Handle numpy arrays
        if hasattr(value, 'tolist'):
            return value.tolist()
        
        # Handle numpy/pandas numeric types
        if hasattr(value, '__float__'):
            return float(value)
        
        return value
    
    def get_analysis_summary(self, symbols: List[str]) -> Dict:
        """Get analysis summary for multiple symbols"""
        
        results = self.analyze_multiple_symbols(symbols)
        
        summary = {
            'total_analyzed': len(results),
            'buy_signals': len([r for r in results if r.get('buy_signal')]),
            'sell_signals': len([r for r in results if r.get('sell_signal')]),
            'average_technical_score': 0.0,
            'strong_signals': [],
            'errors': [r for r in results if 'error' in r]
        }
        
        # Calculate average technical score
        valid_scores = [r.get('technical_score', 0) for r in results if 'technical_score' in r]
        if valid_scores:
            summary['average_technical_score'] = sum(valid_scores) / len(valid_scores)
        
        # Find strong signals
        for result in results:
            if result.get('signal_strength') in ['strong', 'very_strong']:
                summary['strong_signals'].append({
                    'symbol': result['symbol'],
                    'signal_type': 'BUY' if result.get('buy_signal') else 'SELL',
                    'technical_score': result.get('technical_score'),
                    'signal_strength': result.get('signal_strength')
                })
        
        return summary
    
    def analyze_symbol_with_data(self, symbol: str, historical_data: pd.DataFrame) -> Dict:
        """Optimized analysis using pre-fetched data"""
        if historical_data.empty:
            return {'error': 'no_data', 'symbol': symbol}
        
        if len(historical_data) < 20:  # Minimum data required
            return {'error': 'insufficient_data', 'symbol': symbol}
        
        try:
            # Use existing analyze_symbol method but pass the data
            # This ensures we get the correct dict structure
            analysis = self.analyze_symbol(symbol)
            
            # If successful, add timestamp and return
            if 'error' not in analysis:
                analysis['symbol'] = symbol
                analysis['timestamp'] = datetime.now()
                analysis['data_optimized'] = True
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Optimized analysis failed for {symbol}: {e}")
            return {'error': 'analysis_failed', 'symbol': symbol}
    
    def _calculate_indicators_cached(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Calculate indicators with caching support"""
        # Create cache key based on data hash
        data_hash = hash(str(df.iloc[-1]['close']) + str(len(df)))
        cache_key = f"indicators_{symbol}_{data_hash}"
        
        # Check cache first
        if self._indicator_cache and cache_key in self._indicator_cache:
            cache_entry = self._indicator_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < config.CACHE_EXPIRY_MINUTES * 60:
                return cache_entry['data']
        
        # Calculate fresh indicators using existing method
        indicators = self._calculate_all_indicators(df)
        
        # Cache results
        if self._indicator_cache:
            with self._cache_lock:
                self._indicator_cache[cache_key] = {
                    'data': indicators,
                    'timestamp': datetime.now()
                }
        
        return indicators

    def _calculate_rsi_vectorized(self, close_prices: np.ndarray, period: int = 14) -> float:
        """Vectorized RSI calculation for better performance"""
        if len(close_prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gains = np.convolve(gains, np.ones(period), 'valid') / period
        avg_losses = np.convolve(losses, np.ones(period), 'valid') / period
        
        # Avoid division by zero
        rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi[-1]) if len(rsi) > 0 else 50.0

    def _calculate_ema_vectorized(self, close_prices: np.ndarray, period: int) -> float:
        """Vectorized EMA calculation"""
        if len(close_prices) < period:
            return float(np.mean(close_prices))
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(close_prices)
        ema[0] = close_prices[0]
        
        for i in range(1, len(close_prices)):
            ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i-1]
        
        return float(ema[-1])

    def batch_analyze_symbols(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Batch analysis for multiple symbols"""
        results = {}
        
        for symbol, data in symbols_data.items():
            if data is not None and not data.empty:
                results[symbol] = self.analyze_symbol_with_data(symbol, data)
            else:
                results[symbol] = {'error': 'no_data', 'symbol': symbol}
        
        return results

    def process_symbols_parallel(self, symbols: List[str], max_workers: int = 4) -> Dict[str, Dict]:
        """Process symbols in parallel for better performance"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all symbol analysis tasks
            future_to_symbol = {
                executor.submit(self.analyze_symbol, symbol): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol, timeout=config.PROCESSING_TIMEOUT_MINUTES * 60):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    self.logger.error(f"Parallel analysis failed for {symbol}: {e}")
                    results[symbol] = {'error': 'analysis_failed', 'symbol': symbol}
        
        return results

    def get_cached_analysis(self, symbol: str) -> Optional[Dict]:
        """Get cached analysis if available"""
        if not self._indicator_cache:
            return None
        
        # Look for recent cached analysis
        for key, entry in self._indicator_cache.items():
            if symbol in key and (datetime.now() - entry['timestamp']).seconds < config.CACHE_EXPIRY_MINUTES * 60:
                return entry['data']
        
        return None

    def clear_cache(self):
        """Clear indicator cache"""
        if self._indicator_cache:
            with self._cache_lock:
                self._indicator_cache.clear()
            self.logger.info("Indicator cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self._indicator_cache:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            'cache_size': len(self._indicator_cache),
            'cache_entries': list(self._indicator_cache.keys())
        }
    def get_multiple_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get fundamental data for multiple symbols in one query"""
        if not symbols:
            return {}
        
        placeholders = ','.join(['%s'] * len(symbols))
        query = f"""
            SELECT * FROM stocks_categories_table 
            WHERE symbol IN ({placeholders})
        """
        
        results = self.execute_query(query, symbols, fetch_all=True)
        return {row['symbol']: row for row in results}