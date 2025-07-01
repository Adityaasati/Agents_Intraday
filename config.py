# config.py - Simplified for Day 1/2 Implementation Only
import os
import numpy as np
from datetime import datetime

# ==========================================
# TRADING CONFIGURATION - DAY 1/2 ONLY
# ==========================================

# Capital Management
TOTAL_CAPITAL = 100000  # Total trading capital in INR
RISK_PER_TRADE = 2.0    # Risk percentage per trade (2% = 2000 INR per trade)
MAX_POSITIONS_LIVE = 5   # Maximum concurrent positions for Day 1/2

# Position Sizing
MIN_POSITION_SIZE = 5000    # Minimum position size in INR
MAX_POSITION_SIZE_PERCENT = 15  # Maximum 15% of capital per position

# ==========================================
# TECHNICAL ANALYSIS PARAMETERS
# ==========================================

# RSI Configuration
RSI_PERIODS = [14, 21]
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Volatility-based RSI thresholds
RSI_OVERSOLD_LOW_VOL = 25   # Tighter for low volatility stocks
RSI_OVERBOUGHT_LOW_VOL = 75
RSI_OVERSOLD_HIGH_VOL = 20  # Wider for high volatility stocks  
RSI_OVERBOUGHT_HIGH_VOL = 80

# MACD Configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands Configuration
BB_PERIOD = 20
BB_STD_DEV = 2

# Moving Averages Configuration
EMA_PERIODS = [20, 50]
SMA_PERIODS = [20, 50]

# ATR Configuration
ATR_PERIOD = 14

# Volume Analysis
VOLUME_SMA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0  # Volume considered spike if >2x average

# ==========================================
# RISK MANAGEMENT PARAMETERS
# ==========================================

# Stop Loss Configuration
STOP_LOSS_PERCENT = 1.5        # Base stop loss percentage
ATR_STOP_MULTIPLIER = {
    'Low': 1.5,     # Low volatility: 1.5x ATR
    'Medium': 2.0,  # Medium volatility: 2.0x ATR
    'High': 2.5     # High volatility: 2.5x ATR
}

# Risk-Reward Configuration
MIN_RISK_REWARD_RATIO = 2.0    # Minimum 2:1 risk-reward
STANDARD_RISK_REWARD_RATIO = 2.5

# Portfolio Risk Limits
MAX_SECTOR_ALLOCATION = 30.0   # Maximum 30% allocation per sector
MAX_CORRELATION = 0.7          # Maximum correlation between positions
MIN_DIVERSIFICATION_SECTORS = 3  # Minimum sectors for diversification

# ==========================================
# SIGNAL GENERATION PARAMETERS
# ==========================================

# Confidence Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.60    # Minimum 60% confidence for signal generation
HIGH_CONFIDENCE_THRESHOLD = 0.70   # High confidence threshold

# Signal Weights (Day 2 implementation)
TECHNICAL_WEIGHT = 0.50      # 50% weight to technical analysis
FUNDAMENTAL_WEIGHT = 0.30    # 30% weight to fundamental analysis
SENTIMENT_WEIGHT = 0.20      # 20% weight to news sentiment (Day 3)

# Technical Sub-weights
RSI_WEIGHT = 0.30           # 30% of technical score
VOLUME_WEIGHT = 0.25        # 25% of technical score
MA_TREND_WEIGHT = 0.20      # 20% of technical score
MACD_WEIGHT = 0.15          # 15% of technical score
SUPPORT_RESISTANCE_WEIGHT = 0.10  # 10% of technical score

# Category Adjustments
CATEGORY_MULTIPLIER = {
    'A': 1.1,   # A-category stocks get 10% boost
    'B': 1.0,   # B-category stocks no adjustment
    'C': 0.9    # C-category stocks get 10% penalty
}

# Volatility Adjustments
VOLATILITY_POSITION_MULTIPLIER = {
    'Low': 1.2,     # Low volatility: increase position size
    'Medium': 1.0,  # Medium volatility: standard position size
    'High': 0.8     # High volatility: reduce position size
}

# Market Cap Adjustments
MARKET_CAP_MULTIPLIER = {
    'Large_Cap': 1.1,   # Large cap: slight increase
    'Mid_Cap': 1.0,     # Mid cap: standard
    'Small_Cap': 0.9    # Small cap: slight decrease
}

# ==========================================
# DATA CONFIGURATION
# ==========================================

# Your Existing Table Names
STOCKS_CATEGORIES_TABLE = 'stocks_categories_table'
HISTORICAL_DATA_PREFIX = 'historical_data_3m_'

# Market Hours (IST) - Simplified
MARKET_START_TIME = "09:15"
MARKET_END_TIME = "15:30"

# ==========================================
# SYMBOL SELECTION CRITERIA
# ==========================================

# News Sentiment Configuration
MAX_NEWS_ARTICLES_PER_SYMBOL = 8        # Articles to analyze per symbol
MAX_MARKET_NEWS_ARTICLES = 12           # Market news articles
NEWS_FETCH_TIMEOUT = 10                 # News fetching timeout (seconds)
CLAUDE_API_TIMEOUT = 15                 # Claude API timeout (seconds)

# Sentiment Impact Weights
HIGH_IMPACT_WEIGHT = 1.5      # High impact news multiplier
MEDIUM_IMPACT_WEIGHT = 1.0    # Medium impact news multiplier
LOW_IMPACT_WEIGHT = 0.7       # Low impact news multiplier

# Default Filtering Criteria for Day 1/2
DEFAULT_CATEGORIES = ['A', 'B']  # Exclude C-category initially
DEFAULT_MARKET_CAP_TYPES = ['Large_Cap', 'Mid_Cap']  # Exclude Small_Cap initially
DEFAULT_VOLATILITY_TYPES = ['Low', 'Medium']  # Exclude High volatility initially

# Minimum Criteria
MIN_MARKET_CAP_CRORES = 1000  # Minimum 1000 crores market cap
MAX_PE_RATIO = 50             # Maximum PE ratio

# Testing Symbols (High Quality Large Caps)
TESTING_SYMBOLS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK',
    'HDFCBANK', 'HINDUNILVR', 'ITC', 'BAJFINANCE', 'MARUTI'
]

# ==========================================
# PERFORMANCE THRESHOLDS - DAY 1/2
# ==========================================

# Basic Performance Requirements
MIN_WIN_RATE = 50.0           # Minimum 50% win rate
MAX_DRAWDOWN = 10.0           # Maximum 10% drawdown

# System Performance
MAX_SYMBOLS_PER_BATCH = 10    # Process maximum 10 symbols per batch for Day 1/2
MAX_PROCESSING_TIME_MINUTES = 5  # Maximum processing time per batch

# ==========================================
# LOGGING CONFIGURATION - SIMPLIFIED
# ==========================================

# Log Levels
LOG_LEVELS = {
    'database': 'INFO',
    'agents': 'INFO',
    'system': 'INFO'
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_current_quarter():
    """Get current quarter string"""
    now = datetime.now()
    quarter = ((now.month - 1) // 3) + 1
    return f"{now.year}_q{quarter}"

def get_rsi_thresholds(volatility_category: str) -> tuple:
    """Get RSI thresholds based on volatility"""
    
    if volatility_category == 'Low':
        return (RSI_OVERSOLD_LOW_VOL, RSI_OVERBOUGHT_LOW_VOL)
    elif volatility_category == 'High':
        return (RSI_OVERSOLD_HIGH_VOL, RSI_OVERBOUGHT_HIGH_VOL)
    else:
        return (RSI_OVERSOLD, RSI_OVERBOUGHT)

def get_stop_loss_multiplier(volatility_category: str) -> float:
    """Get ATR stop loss multiplier based on volatility"""
    return ATR_STOP_MULTIPLIER.get(volatility_category, 2.0)

def calculate_final_confidence(technical_score: float, fundamental_score: float = 0.5, 
                             sentiment_score: float = 0.5, category: str = 'B') -> float:
    """Calculate final confidence score combining all components"""
    
    # Weighted combination as per Day 2 specification
    combined_score = (
        technical_score * TECHNICAL_WEIGHT +
        fundamental_score * FUNDAMENTAL_WEIGHT +
        sentiment_score * SENTIMENT_WEIGHT
    )
    
    # Apply category adjustment
    category_adj = CATEGORY_MULTIPLIER.get(category, 1.0)
    final_score = combined_score * category_adj
    
    # Ensure within bounds
    return max(0.0, min(1.0, final_score))

def validate_technical_score(score: float) -> float:
    """Validate and normalize technical score"""
    if score is None or np.isnan(score) or np.isinf(score):
        return 0.5  # Default neutral score
    return max(0.0, min(1.0, float(score)))

def get_position_size_for_category(base_size: float, category: str, volatility: str, market_cap_type: str) -> float:
    """Calculate adjusted position size based on stock characteristics"""
    
    adjusted_size = base_size
    adjusted_size *= CATEGORY_MULTIPLIER.get(category, 1.0)
    adjusted_size *= VOLATILITY_POSITION_MULTIPLIER.get(volatility, 1.0)
    adjusted_size *= MARKET_CAP_MULTIPLIER.get(market_cap_type, 1.0)
    
    # Ensure within limits
    adjusted_size = max(MIN_POSITION_SIZE, adjusted_size)
    adjusted_size = min(TOTAL_CAPITAL * MAX_POSITION_SIZE_PERCENT / 100, adjusted_size)
    
    return round(adjusted_size, 0)

# Update current quarter on import
CURRENT_QUARTER = get_current_quarter()

# ==========================================
# VALIDATION - Basic checks for Day 1/2
# ==========================================

def validate_config():
    """Validate critical parameters - called only when needed"""
    try:
        assert TOTAL_CAPITAL > 0, "Total capital must be positive"
        assert 0 < RISK_PER_TRADE <= 20, "Risk per trade must be between 0 and 20%"
        assert MIN_CONFIDENCE_THRESHOLD <= 1.0, "Confidence threshold must be <= 1.0"
        assert MIN_RISK_REWARD_RATIO >= 1.0, "Risk-reward ratio must be >= 1.0"
        print("✓ Configuration validation passed")
        return True
    except AssertionError as e:
        print(f"✗ Configuration validation error: {e}")
        return False