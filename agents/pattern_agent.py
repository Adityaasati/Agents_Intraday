"""
FILE: agents/pattern_agent.py
PURPOSE: Advanced Pattern Recognition Agent

DESCRIPTION:
- Detects advanced chart patterns (Head & Shoulders, Double Tops/Bottoms, Triangles, Flags)
- Identifies breakout and breakdown patterns
- Integrates with existing technical analysis framework
- Stores pattern data for strategy validation

USAGE:
- Called by main.py for pattern analysis
- Integrates with technical_agent.py for enhanced signals
- Results stored in agent_pattern_signals table
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

import config
from database.enhanced_database_manager import EnhancedDatabaseManager

class PatternAgent:
    """Advanced chart pattern recognition and analysis"""
    
    def __init__(self, db_manager: EnhancedDatabaseManager = None):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager or EnhancedDatabaseManager()
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Pattern configuration
        self.pattern_config = {
            'head_shoulders': {'min_periods': 30, 'tolerance': 0.02},
            'double_top': {'min_periods': 20, 'tolerance': 0.015},
            'double_bottom': {'min_periods': 20, 'tolerance': 0.015},
            'triangle': {'min_periods': 15, 'tolerance': 0.01},
            'flag': {'min_periods': 10, 'tolerance': 0.005},
            'channel': {'min_periods': 25, 'tolerance': 0.02}
        }
    
    def analyze_patterns(self, symbol: str, lookback_days: int = 60) -> Dict:
        """Complete pattern analysis for a symbol"""
        
        try:
            # Get historical data
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=lookback_days)
            
            # df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            df = self.db_manager.get_historical_data(symbol, limit=1000)  # Adjust limit as needed
            
            if df.empty or len(df) < 30:
                return {'symbol': symbol, 'patterns': [], 'status': 'insufficient_data'}
            
            # Detect all patterns
            patterns = []
            patterns.extend(self._detect_head_shoulders(df))
            patterns.extend(self._detect_double_patterns(df))
            patterns.extend(self._detect_triangle_patterns(df))
            patterns.extend(self._detect_flag_patterns(df))
            patterns.extend(self._detect_channel_patterns(df))
            
            # Score and rank patterns
            scored_patterns = self._score_patterns(patterns, df)
            
            # Store results
            self._store_pattern_analysis(symbol, scored_patterns)
            
            return {
                'symbol': symbol,
                'patterns': scored_patterns,
                'total_patterns': len(scored_patterns),
                'high_confidence_patterns': len([p for p in scored_patterns if p['confidence'] > 0.7]),
                'timestamp': datetime.now(self.ist)
            }
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Head and Shoulders patterns"""
        
        patterns = []
        highs = df['high'].values
        lows = df['low'].values
        
        if len(df) < 30:
            return patterns
        
        # Find potential head and shoulders
        for i in range(10, len(df) - 10):
            # Look for three peaks
            left_peak = np.argmax(highs[i-10:i])
            head_peak = np.argmax(highs[i-5:i+5])
            right_peak = np.argmax(highs[i:i+10])
            
            if head_peak == 5 and left_peak < 10 and right_peak > 0:
                left_high = highs[i-10+left_peak]
                head_high = highs[i-5+head_peak]
                right_high = highs[i+right_peak]
                
                # Check pattern validity
                tolerance = self.pattern_config['head_shoulders']['tolerance']
                if (head_high > left_high * (1 + tolerance) and 
                    head_high > right_high * (1 + tolerance) and
                    abs(left_high - right_high) / left_high < tolerance):
                    
                    patterns.append({
                        'type': 'head_shoulders',
                        'subtype': 'bearish',
                        'start_idx': i-10,
                        'end_idx': i+10,
                        'key_levels': [left_high, head_high, right_high],
                        'neckline': min(lows[i-5:i+5])
                    })
        
        # Inverse head and shoulders
        for i in range(10, len(df) - 10):
            left_valley = np.argmin(lows[i-10:i])
            head_valley = np.argmin(lows[i-5:i+5])
            right_valley = np.argmin(lows[i:i+10])
            
            if head_valley == 5 and left_valley < 10 and right_valley > 0:
                left_low = lows[i-10+left_valley]
                head_low = lows[i-5+head_valley]
                right_low = lows[i+right_valley]
                
                tolerance = self.pattern_config['head_shoulders']['tolerance']
                if (head_low < left_low * (1 - tolerance) and 
                    head_low < right_low * (1 - tolerance) and
                    abs(left_low - right_low) / left_low < tolerance):
                    
                    patterns.append({
                        'type': 'inverse_head_shoulders',
                        'subtype': 'bullish',
                        'start_idx': i-10,
                        'end_idx': i+10,
                        'key_levels': [left_low, head_low, right_low],
                        'neckline': max(highs[i-5:i+5])
                    })
        
        return patterns
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Double Top and Double Bottom patterns"""
        
        patterns = []
        highs = df['high'].values
        lows = df['low'].values
        
        # Double tops
        for i in range(15, len(df) - 5):
            tolerance = self.pattern_config['double_top']['tolerance']
            
            # Find recent high
            recent_high_idx = np.argmax(highs[i-5:i+5])
            recent_high = highs[i-5+recent_high_idx]
            
            # Look for similar high in past
            for j in range(i-15, i-5):
                past_high = highs[j]
                if abs(past_high - recent_high) / recent_high < tolerance:
                    # Check valley between peaks
                    valley_start = j + 1
                    valley_end = i - 5 + recent_high_idx
                    valley_low = np.min(lows[valley_start:valley_end])
                    
                    if valley_low < recent_high * 0.95:  # Significant valley
                        patterns.append({
                            'type': 'double_top',
                            'subtype': 'bearish',
                            'start_idx': j,
                            'end_idx': i,
                            'key_levels': [past_high, recent_high],
                            'valley_level': valley_low
                        })
                        break
        
        # Double bottoms
        for i in range(15, len(df) - 5):
            tolerance = self.pattern_config['double_bottom']['tolerance']
            
            recent_low_idx = np.argmin(lows[i-5:i+5])
            recent_low = lows[i-5+recent_low_idx]
            
            for j in range(i-15, i-5):
                past_low = lows[j]
                if abs(past_low - recent_low) / recent_low < tolerance:
                    peak_start = j + 1
                    peak_end = i - 5 + recent_low_idx
                    peak_high = np.max(highs[peak_start:peak_end])
                    
                    if peak_high > recent_low * 1.05:
                        patterns.append({
                            'type': 'double_bottom',
                            'subtype': 'bullish',
                            'start_idx': j,
                            'end_idx': i,
                            'key_levels': [past_low, recent_low],
                            'peak_level': peak_high
                        })
                        break
        
        return patterns
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Triangle patterns (Ascending, Descending, Symmetrical)"""
        
        patterns = []
        if len(df) < 20:
            return patterns
        
        # Look for converging trend lines
        for i in range(15, len(df) - 5):
            window = df.iloc[i-15:i].copy()
            
            # Find higher lows (ascending triangle)
            lows = window['low'].values
            low_trend = np.polyfit(range(len(lows)), lows, 1)
            
            # Find lower highs (descending triangle)
            highs = window['high'].values
            high_trend = np.polyfit(range(len(highs)), highs, 1)
            
            # Ascending triangle (rising lows, flat highs)
            if low_trend[0] > 0 and abs(high_trend[0]) < 0.1:
                resistance_level = np.max(highs[-5:])
                patterns.append({
                    'type': 'ascending_triangle',
                    'subtype': 'bullish',
                    'start_idx': i-15,
                    'end_idx': i,
                    'resistance_level': resistance_level,
                    'support_slope': low_trend[0]
                })
            
            # Descending triangle (falling highs, flat lows)
            elif high_trend[0] < 0 and abs(low_trend[0]) < 0.1:
                support_level = np.min(lows[-5:])
                patterns.append({
                    'type': 'descending_triangle',
                    'subtype': 'bearish',
                    'start_idx': i-15,
                    'end_idx': i,
                    'support_level': support_level,
                    'resistance_slope': high_trend[0]
                })
            
            # Symmetrical triangle (converging lines)
            elif low_trend[0] > 0 and high_trend[0] < 0:
                patterns.append({
                    'type': 'symmetrical_triangle',
                    'subtype': 'neutral',
                    'start_idx': i-15,
                    'end_idx': i,
                    'support_slope': low_trend[0],
                    'resistance_slope': high_trend[0]
                })
        
        return patterns
    
    def _detect_flag_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Flag and Pennant patterns"""
        
        patterns = []
        if len(df) < 15:
            return patterns
        
        for i in range(10, len(df) - 5):
            # Look for strong move followed by consolidation
            recent_window = df.iloc[i-10:i]
            
            # Check for strong upward move
            price_change = (recent_window['close'].iloc[-1] - recent_window['close'].iloc[0]) / recent_window['close'].iloc[0]
            
            if price_change > 0.05:  # 5% move
                # Check for sideways consolidation
                consolidation = df.iloc[i:i+5]
                if len(consolidation) >= 5:
                    volatility = consolidation['close'].std() / consolidation['close'].mean()
                    
                    if volatility < 0.02:  # Low volatility consolidation
                        patterns.append({
                            'type': 'bull_flag',
                            'subtype': 'bullish',
                            'start_idx': i-10,
                            'end_idx': i+5,
                            'flagpole_gain': price_change,
                            'consolidation_range': [consolidation['low'].min(), consolidation['high'].max()]
                        })
            
            elif price_change < -0.05:  # 5% decline
                consolidation = df.iloc[i:i+5]
                if len(consolidation) >= 5:
                    volatility = consolidation['close'].std() / consolidation['close'].mean()
                    
                    if volatility < 0.02:
                        patterns.append({
                            'type': 'bear_flag',
                            'subtype': 'bearish',
                            'start_idx': i-10,
                            'end_idx': i+5,
                            'flagpole_decline': price_change,
                            'consolidation_range': [consolidation['low'].min(), consolidation['high'].max()]
                        })
        
        return patterns
    
    def _detect_channel_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Channel patterns (Parallel support and resistance)"""
        
        patterns = []
        if len(df) < 25:
            return patterns
        
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            
            # Fit trend lines to highs and lows
            highs = window['high'].values
            lows = window['low'].values
            x = np.arange(len(window))
            
            high_trend = np.polyfit(x, highs, 1)
            low_trend = np.polyfit(x, lows, 1)
            
            # Check if lines are roughly parallel
            slope_diff = abs(high_trend[0] - low_trend[0])
            
            if slope_diff < 0.1:  # Parallel lines
                # Calculate channel width
                high_line = high_trend[0] * x + high_trend[1]
                low_line = low_trend[0] * x + low_trend[1]
                avg_width = np.mean(high_line - low_line)
                
                if avg_width > 0:  # Valid channel
                    if high_trend[0] > 0.05:  # Rising channel
                        channel_type = 'rising_channel'
                        bias = 'bullish'
                    elif high_trend[0] < -0.05:  # Falling channel
                        channel_type = 'falling_channel'
                        bias = 'bearish'
                    else:  # Horizontal channel
                        channel_type = 'horizontal_channel'
                        bias = 'neutral'
                    
                    patterns.append({
                        'type': channel_type,
                        'subtype': bias,
                        'start_idx': i-20,
                        'end_idx': i,
                        'upper_slope': high_trend[0],
                        'lower_slope': low_trend[0],
                        'channel_width': avg_width
                    })
        
        return patterns
    
    def _score_patterns(self, patterns: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Score and rank detected patterns"""
        
        scored_patterns = []
        
        for pattern in patterns:
            score = 0.5  # Base score
            
            # Pattern-specific scoring
            if pattern['type'] in ['head_shoulders', 'inverse_head_shoulders']:
                score += 0.2  # Reliable reversal patterns
            elif pattern['type'] in ['double_top', 'double_bottom']:
                score += 0.15
            elif 'triangle' in pattern['type']:
                score += 0.1
            elif 'flag' in pattern['type']:
                score += 0.25  # Strong continuation patterns
            
            # Volume confirmation
            start_idx = pattern.get('start_idx', 0)
            end_idx = pattern.get('end_idx', len(df))
            pattern_volume = df.iloc[start_idx:end_idx]['volume'].mean()
            avg_volume = df['volume'].mean()
            
            if pattern_volume > avg_volume * 1.2:
                score += 0.1
            
            # Recency bonus
            days_old = (len(df) - end_idx) / 5  # Assuming 5-min data
            if days_old < 5:
                score += 0.1
            
            pattern['confidence'] = min(score, 0.95)
            scored_patterns.append(pattern)
        
        return sorted(scored_patterns, key=lambda x: x['confidence'], reverse=True)
    
    def _store_pattern_analysis(self, symbol: str, patterns: List[Dict]):
        """Store pattern analysis results"""
        
        try:
            # Clear old patterns for this symbol
            self.db_manager.execute_query(
                "DELETE FROM agent_pattern_signals WHERE symbol = %s AND detected_date < %s",
                (symbol, datetime.now(self.ist) - timedelta(days=30))
            )
            
            # Store new patterns
            for pattern in patterns:
                self.db_manager.execute_query("""
                    INSERT INTO agent_pattern_signals 
                    (symbol, pattern_type, pattern_subtype, confidence_score, 
                     key_levels, detected_date, pattern_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    symbol,
                    pattern['type'],
                    pattern['subtype'],
                    pattern['confidence'],
                    str(pattern.get('key_levels', [])),
                    datetime.now(self.ist),
                    str(pattern)
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to store patterns for {symbol}: {e}")
    
    def get_active_patterns(self, symbol: str = None) -> List[Dict]:
        """Get active patterns for symbol or all symbols"""
        
        query = """
            SELECT symbol, pattern_type, pattern_subtype, confidence_score, 
                   key_levels, detected_date, pattern_data
            FROM agent_pattern_signals 
            WHERE detected_date > %s
        """
        params = [datetime.now(self.ist) - timedelta(days=7)]
        
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        
        query += " ORDER BY confidence_score DESC"
        
        results = self.db_manager.fetch_query(query, params)
        return results if results else []