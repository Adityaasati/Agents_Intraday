"""
FILE: utils/technical_calculations.py
LOCATION: /utils/ directory  
PURPOSE: Technical Analysis Utility Functions - Basic Day 1/2 Functions Only

DESCRIPTION:
- Basic utility functions for technical analysis
- Data validation and cleaning functions for OHLCV data
- Simple support/resistance level detection
- Basic validation functions for technical indicators

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