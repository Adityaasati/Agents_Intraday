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
# Add these variables to the existing config.py file (replace the previous live trading section)

# ==========================================
# LIVE TRADING CONFIGURATION
# ==========================================

# Master Trading Control
TRADE_MODE = os.getenv('TRADE_MODE', 'no').lower() == 'yes'  # Simple yes/no control
LIVE_TRADING_MODE = bool(os.getenv('LIVE_TRADING_MODE', 'False').lower() == 'true')
LIVE_TRADING_CAPITAL = float(os.getenv('LIVE_TRADING_CAPITAL', '50000'))

# Kite API Configuration (simplified)
KITE_API_KEY = os.getenv('KITE_API_KEY', '')
KITE_API_SECRET = os.getenv('KITE_API_SECRET', '')

# Trading Logic Control
# If TRADE_MODE = no: Generate signals only, no orders
# If TRADE_MODE = yes: Place actual orders (requires LIVE_TRADING_MODE = true)
GENERATE_SIGNALS_ONLY = not TRADE_MODE

# Order Management
LIVE_ORDER_TIMEOUT_SECONDS = 30
LIVE_ORDER_RETRY_ATTEMPTS = 3
LIVE_ORDER_RETRY_DELAY_SECONDS = 2

# Live Trading Risk Controls (Conservative)
LIVE_MAX_LOSS_PER_DAY = 5000
LIVE_MAX_POSITIONS = 5
LIVE_POSITION_SIZE_LIMIT = 0.10  # 10% max per position
LIVE_MIN_CONFIDENCE_THRESHOLD = 0.75

# Market Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Live Execution Settings
LIVE_EXECUTION_DELAY_SECONDS = 5
LIVE_PRICE_TOLERANCE_PERCENT = 0.5
LIVE_QUANTITY_ROUNDING = True

# Monitoring and Alerts
LIVE_MONITORING_INTERVAL_SECONDS = 30
LIVE_POSITION_CHECK_INTERVAL_SECONDS = 60
LIVE_HEARTBEAT_INTERVAL_SECONDS = 300

# Safety Mechanisms
LIVE_CIRCUIT_BREAKER_LOSS_PERCENT = 3.0
LIVE_EMERGENCY_EXIT_ENABLED = True
LIVE_MAX_ORDERS_PER_MINUTE = 5

# Token Management
KITE_TOKEN_FILE = "kite_token.txt"
AUTO_REFRESH_TOKEN = True

# Live Trading Logging
LIVE_TRADING_LOG_LEVEL = 'INFO'
LIVE_ORDER_LOG_RETENTION_DAYS = 30

# Approved Symbols for Live Trading (Conservative list)
LIVE_TRADING_APPROVED_SYMBOLS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC', 
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'LT', 'WIPRO'
]
# ==========================================
# SIGNAL GENERATION PARAMETERS
# ==========================================
# ==========================================
# ADD THESE LINES TO END OF config.py
# ==========================================

# Performance Optimization Settings (Day 7A)
DB_CONNECTION_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
DB_CONNECTION_POOL_MAX = int(os.getenv('DB_POOL_MAX', '20'))
DB_CONNECTION_TIMEOUT = int(os.getenv('DB_TIMEOUT', '30'))
DB_QUERY_TIMEOUT = int(os.getenv('DB_QUERY_TIMEOUT', '60'))

MAX_CONCURRENT_SYMBOLS = int(os.getenv('MAX_CONCURRENT_SYMBOLS', '10'))
BATCH_SIZE_SYMBOLS = int(os.getenv('BATCH_SIZE', '25'))
MAX_PARALLEL_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
PROCESSING_TIMEOUT_MINUTES = int(os.getenv('PROCESSING_TIMEOUT', '15'))

MAX_MEMORY_USAGE_MB = int(os.getenv('MAX_MEMORY_MB', '1024'))
CACHE_SIZE_INDICATORS = int(os.getenv('CACHE_SIZE', '1000'))
CLEANUP_INTERVAL_MINUTES = int(os.getenv('CLEANUP_INTERVAL', '30'))

TARGET_SYMBOLS_PER_MINUTE = int(os.getenv('TARGET_SPEED', '25'))
MAX_PROCESSING_TIME_PER_SYMBOL = float(os.getenv('MAX_TIME_PER_SYMBOL', '2.0'))
PERFORMANCE_ALERT_THRESHOLD = float(os.getenv('PERF_ALERT_THRESHOLD', '1.5'))

ENABLE_INDICATOR_CACHE = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
CACHE_EXPIRY_MINUTES = int(os.getenv('CACHE_EXPIRY', '15'))
ENABLE_QUERY_CACHE = os.getenv('ENABLE_QUERY_CACHE', 'true').lower() == 'true'

ENABLE_PERFORMANCE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
MONITOR_INTERVAL_SECONDS = int(os.getenv('MONITOR_INTERVAL', '30'))
LOG_PERFORMANCE_DETAILS = os.getenv('LOG_PERFORMANCE', 'false').lower() == 'true'

ENABLE_QUERY_OPTIMIZATION = os.getenv('ENABLE_QUERY_OPT', 'true').lower() == 'true'
USE_PREPARED_STATEMENTS = os.getenv('USE_PREPARED_STMT', 'true').lower() == 'true'
ENABLE_BULK_OPERATIONS = os.getenv('ENABLE_BULK_OPS', 'true').lower() == 'true'

# Performance Helper Functions
def get_optimal_batch_size(total_symbols: int) -> int:
    """Calculate optimal batch size based on total symbols"""
    if total_symbols <= 50:
        return min(BATCH_SIZE_SYMBOLS, total_symbols)
    elif total_symbols <= 200:
        return min(50, total_symbols // 4)
    else:
        return min(100, total_symbols // 8)

def get_worker_count(batch_size: int) -> int:
    """Calculate optimal worker count for batch size"""
    return min(MAX_PARALLEL_WORKERS, max(1, batch_size // 10))

def validate_performance_config() -> dict:
    """Validate performance configuration"""
    return {
        'pool_size': DB_CONNECTION_POOL_SIZE,
        'batch_size': BATCH_SIZE_SYMBOLS,
        'workers': MAX_PARALLEL_WORKERS,
        'cache_enabled': ENABLE_INDICATOR_CACHE,
        'monitoring_enabled': ENABLE_PERFORMANCE_MONITORING
    }
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

# Add these configuration parameters to existing config.py file
# Insert at the end of the file with the portfolio risk management settings

# =============================================================================
# PORTFOLIO RISK MANAGEMENT SETTINGS
# =============================================================================

# Portfolio Concentration Limits
MAX_SECTOR_ALLOCATION = 30.0  # Maximum 30% allocation per sector
MAX_POSITION_SIZE_PERCENT = 15.0  # Maximum 15% per individual position
MIN_DIVERSIFICATION_SECTORS = 3  # Minimum number of sectors

# Correlation Risk Management
MAX_CORRELATION_THRESHOLD = 0.7  # Maximum correlation between positions
CORRELATION_ANALYSIS_DAYS = 30  # Days of data for correlation calculation
HIGH_CORRELATION_PENALTY = 0.4  # Position size reduction for high correlation

# Portfolio Risk Limits
MAX_PORTFOLIO_RISK_PERCENT = 20.0  # Maximum total portfolio risk
RISK_BUDGET_WARNING_THRESHOLD = 90.0  # Warning when 90% of risk budget used
PORTFOLIO_BETA_TARGET = 1.0  # Target portfolio beta

# Cash Management
MIN_CASH_BUFFER_PERCENT = 10.0  # Minimum cash buffer (10% of capital)
OPTIMAL_CASH_BUFFER_PERCENT = 15.0  # Optimal cash buffer (15% of capital)
MAX_CASH_THRESHOLD_PERCENT = 30.0  # Alert if cash exceeds 30%

# Market Regime Adjustments
REGIME_POSITION_MULTIPLIERS = {
    'bull': 1.2,      # Increase position size in bull market
    'bear': 0.7,      # Reduce position size in bear market
    'sideways': 1.0   # Normal position size in sideways market
}

REGIME_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to apply regime adjustments

# Risk Monitoring Settings
RISK_MONITORING_FREQUENCY_HOURS = 4  # Monitor portfolio risk every 4 hours
MONITORING_HISTORY_DAYS = 30  # Keep 30 days of monitoring history
CRITICAL_RISK_ALERT_THRESHOLD = 3  # Number of alerts to trigger critical status

# Diversification Scoring
DIVERSIFICATION_SECTOR_POINTS = 15  # Points per sector (max 60 points)
DIVERSIFICATION_CONCENTRATION_POINTS = 40  # Points for low concentration (max 40 points)
EXCELLENT_DIVERSIFICATION_THRESHOLD = 80  # Score for excellent diversification
GOOD_DIVERSIFICATION_THRESHOLD = 60  # Score for good diversification

# Enhanced Position Sizing Parameters
VOLATILITY_POSITION_MULTIPLIERS = {
    'Low': 1.2,     # Increase size for low volatility stocks
    'Medium': 1.0,  # Normal size for medium volatility
    'High': 0.8     # Reduce size for high volatility stocks
}

MARKET_CAP_POSITION_MULTIPLIERS = {
    'Large_Cap': 1.1,   # Slightly larger positions in large caps
    'Mid_Cap': 1.0,     # Normal positions in mid caps
    'Small_Cap': 0.9    # Smaller positions in small caps
}

CATEGORY_POSITION_MULTIPLIERS = {
    'A': 1.2,  # Larger positions in category A stocks
    'B': 1.0,  # Normal positions in category B stocks
    'C': 0.8   # Smaller positions in category C stocks
}

# Correlation-Based Adjustments
SECTOR_CONCENTRATION_MULTIPLIERS = {
    'overweight': 0.5,  # Significantly reduce if sector overweight (>25%)
    'normal': 0.8,      # Moderate reduction if sector normal (15-25%)
    'underweight': 1.0  # No reduction if sector underweight (<15%)
}

# Risk Alert Definitions
RISK_ALERT_THRESHOLDS = {
    'HIGH_PORTFOLIO_RISK': 20.0,        # Total portfolio risk > 20%
    'HIGH_SECTOR_CONCENTRATION': 30.0,   # Single sector > 30%
    'HIGH_CORRELATION_RISK': 0.8,        # Max correlation > 0.8
    'RISK_BUDGET_EXHAUSTED': 95.0,       # Risk budget > 95% used
    'POSITION_LIMIT_EXCEEDED': None      # Based on MAX_POSITIONS_LIVE
}

# Performance Thresholds for Enhanced Features
ENHANCED_FEATURES_CONFIG = {
    'correlation_calculation_timeout': 10,    # Max seconds for correlation calc
    'risk_monitoring_batch_size': 50,         # Max positions to analyze at once
    'optimization_max_iterations': 100,       # Max optimization iterations
    'fallback_to_basic_on_error': True       # Fallback to basic methods on error
}

# =============================================================================
# VALIDATION FUNCTIONS FOR PORTFOLIO RISK SETTINGS
# =============================================================================

def validate_portfolio_risk_config() -> bool:
    """Validate portfolio risk configuration parameters"""
    
    try:
        # Check percentage values are within valid ranges
        percentage_configs = [
            ('MAX_SECTOR_ALLOCATION', MAX_SECTOR_ALLOCATION, 1, 100),
            ('MAX_POSITION_SIZE_PERCENT', MAX_POSITION_SIZE_PERCENT, 1, 50),
            ('MAX_PORTFOLIO_RISK_PERCENT', MAX_PORTFOLIO_RISK_PERCENT, 5, 50),
            ('MIN_CASH_BUFFER_PERCENT', MIN_CASH_BUFFER_PERCENT, 5, 30),
            ('OPTIMAL_CASH_BUFFER_PERCENT', OPTIMAL_CASH_BUFFER_PERCENT, 10, 40)
        ]
        
        for name, value, min_val, max_val in percentage_configs:
            if not (min_val <= value <= max_val):
                print(f"Invalid {name}: {value}% (should be {min_val}-{max_val}%)")
                return False
        
        # Check correlation threshold
        if not (0.1 <= MAX_CORRELATION_THRESHOLD <= 1.0):
            print(f"Invalid MAX_CORRELATION_THRESHOLD: {MAX_CORRELATION_THRESHOLD}")
            return False
        
        # Check regime multipliers
        for regime, multiplier in REGIME_POSITION_MULTIPLIERS.items():
            if not (0.1 <= multiplier <= 2.0):
                print(f"Invalid regime multiplier for {regime}: {multiplier}")
                return False
        
        # Check consistency
        if MIN_CASH_BUFFER_PERCENT >= OPTIMAL_CASH_BUFFER_PERCENT:
            print("MIN_CASH_BUFFER should be less than OPTIMAL_CASH_BUFFER")
            return False
        
        if MAX_POSITION_SIZE_PERCENT * MIN_DIVERSIFICATION_SECTORS > 80:
            print("Position size and diversification settings may be incompatible")
            return False
        
        return True
        
    except Exception as e:
        print(f"Portfolio risk config validation failed: {e}")
        return False

def get_portfolio_risk_summary() -> dict:
    """Get summary of portfolio risk configuration"""
    
    return {
        'concentration_limits': {
            'max_sector_percent': MAX_SECTOR_ALLOCATION,
            'max_position_percent': MAX_POSITION_SIZE_PERCENT,
            'min_sectors': MIN_DIVERSIFICATION_SECTORS
        },
        'risk_limits': {
            'max_portfolio_risk': MAX_PORTFOLIO_RISK_PERCENT,
            'max_correlation': MAX_CORRELATION_THRESHOLD,
            'target_beta': PORTFOLIO_BETA_TARGET
        },
        'cash_management': {
            'min_buffer': MIN_CASH_BUFFER_PERCENT,
            'optimal_buffer': OPTIMAL_CASH_BUFFER_PERCENT,
            'max_threshold': MAX_CASH_THRESHOLD_PERCENT
        },
        'monitoring': {
            'frequency_hours': RISK_MONITORING_FREQUENCY_HOURS,
            'history_days': MONITORING_HISTORY_DAYS,
            'critical_threshold': CRITICAL_RISK_ALERT_THRESHOLD
        }
    }

# =============================================================================
# ENHANCED VALIDATION FUNCTION UPDATE
# =============================================================================

def validate_enhanced_config() -> bool:
    """Enhanced validation including portfolio risk settings"""
    
    # Run existing validation
    if not validate_config():
        return False
    
    # Run portfolio risk validation
    if not validate_portfolio_risk_config():
        return False
    
    print("Enhanced configuration validation: PASSED")
    return True



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
    
# STOP LOSS OPTIMIZATION SETTINGS
# =============================================================================

# Trailing Stop Loss Configuration
TRAILING_STOP_INITIAL_PERCENT = float(os.getenv('TRAILING_STOP_INITIAL', '15.0'))  # Initial trailing stop
TRAILING_STOP_MIN_PERCENT = float(os.getenv('TRAILING_STOP_MIN', '8.0'))          # Minimum trailing stop
TRAILING_STOP_TIGHTENING_THRESHOLD = float(os.getenv('TRAILING_TIGHTEN_THRESHOLD', '10.0'))  # Profit % to start tightening

# Volatility-Based Stop Loss
ATR_STOP_MULTIPLIER = float(os.getenv('ATR_STOP_MULTIPLIER', '2.0'))              # ATR multiplier for stops
ATR_CALCULATION_PERIODS = int(os.getenv('ATR_PERIODS', '14'))                     # Periods for ATR calculation
VOLATILITY_STOP_MAX_PERCENT = float(os.getenv('VOLATILITY_STOP_MAX', '20.0'))     # Max volatility stop

# Technical Level Stops
SUPPORT_RESISTANCE_BUFFER = float(os.getenv('SUPPORT_BUFFER_PERCENT', '2.0'))     # Buffer below support
TECHNICAL_STOP_LOOKBACK_DAYS = int(os.getenv('TECHNICAL_LOOKBACK_DAYS', '20'))    # Days to look for levels

# Time-Based Stops
TIME_STOP_DAYS = int(os.getenv('TIME_STOP_DAYS', '10'))                           # Days before time stop
TIME_STOP_ENABLED = os.getenv('TIME_STOP_ENABLED', 'true').lower() == 'true'     # Enable time stops

# Profit Protection Settings
BREAKEVEN_PROFIT_THRESHOLD = float(os.getenv('BREAKEVEN_THRESHOLD', '10.0'))      # Move to breakeven at +10%
PROFIT_PROTECTION_LEVELS = {
    10.0: 0.0,    # At +10% profit, move stop to breakeven
    20.0: 5.0,    # At +20% profit, move stop to +5%
    30.0: 15.0    # At +30% profit, move stop to +15%
}

# Stop Loss Strategy Priority (1 = highest)
STOP_LOSS_STRATEGY_PRIORITY = {
    'trailing': 1,
    'volatility': 2,
    'technical': 3,
    'time': 4
}

# =============================================================================
# RISK PARITY SETTINGS
# =============================================================================

# Risk Parity Target
TARGET_RISK_CONTRIBUTION_PERCENT = float(os.getenv('TARGET_RISK_CONTRIBUTION', '20.0'))  # Equal risk per position
RISK_PARITY_TOLERANCE = float(os.getenv('RISK_PARITY_TOLERANCE', '5.0'))                 # Tolerance for rebalancing
RISK_PARITY_REBALANCE_THRESHOLD = float(os.getenv('REBALANCE_THRESHOLD', '25.0'))        # Trigger rebalancing

# Portfolio Optimization Settings
EFFICIENT_FRONTIER_POINTS = int(os.getenv('EFFICIENT_FRONTIER_POINTS', '20'))            # Points on efficient frontier
OPTIMIZATION_LOOKBACK_MONTHS = int(os.getenv('OPTIMIZATION_LOOKBACK_MONTHS', '6'))       # Months of data for optimization
MIN_PORTFOLIO_POSITIONS = int(os.getenv('MIN_PORTFOLIO_POSITIONS', '3'))                 # Minimum positions for optimization

# Modern Portfolio Theory Parameters
RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '6.5'))                              # Risk-free rate (%)
TARGET_RETURN_RANGE = (8.0, 25.0)                                                       # Target return range (%)
MAX_SINGLE_ASSET_WEIGHT = float(os.getenv('MAX_SINGLE_ASSET_WEIGHT', '25.0'))           # Max weight per asset (%)

# Rebalancing Settings
REBALANCING_FREQUENCY_DAYS = int(os.getenv('REBALANCING_FREQUENCY', '7'))                # Weekly rebalancing
REBALANCING_THRESHOLD_PERCENT = float(os.getenv('REBALANCING_THRESHOLD', '15.0'))        # Rebalance if position deviates >15%
AUTO_REBALANCING_ENABLED = os.getenv('AUTO_REBALANCING', 'false').lower() == 'true'     # Auto rebalancing

# =============================================================================
# ADVANCED PORTFOLIO METRICS
# =============================================================================

# Performance Attribution
ATTRIBUTION_PERIODS = ['1D', '1W', '1M', '3M']                                          # Attribution time periods
BENCHMARK_SYMBOL = os.getenv('BENCHMARK_SYMBOL', 'NIFTY50')                             # Benchmark for comparison

# Risk Metrics Calculation
VAR_CONFIDENCE_LEVEL = float(os.getenv('VAR_CONFIDENCE', '95.0'))                        # VaR confidence level
EXPECTED_SHORTFALL_LEVEL = float(os.getenv('ES_LEVEL', '95.0'))                          # Expected shortfall level
DRAWDOWN_CALCULATION_PERIOD = int(os.getenv('DRAWDOWN_PERIOD', '252'))                   # Trading days for drawdown

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_stop_loss_config() -> bool:
    """Validate stop loss configuration parameters"""
    
    try:
        # Validate percentage ranges
        if not (5.0 <= TRAILING_STOP_INITIAL_PERCENT <= 25.0):
            return False
        if not (3.0 <= TRAILING_STOP_MIN_PERCENT <= 15.0):
            return False
        if not (1.0 <= ATR_STOP_MULTIPLIER <= 5.0):
            return False
        if not (5 <= ATR_CALCULATION_PERIODS <= 30):
            return False
        if not (5 <= TIME_STOP_DAYS <= 30):
            return False
        
        # Validate profit protection levels
        for profit_level, stop_level in PROFIT_PROTECTION_LEVELS.items():
            if stop_level >= profit_level:
                return False
        
        return True
        
    except Exception:
        return False

def validate_risk_parity_config() -> bool:
    """Validate risk parity configuration parameters"""
    
    try:
        # Validate risk parity settings
        if not (10.0 <= TARGET_RISK_CONTRIBUTION_PERCENT <= 50.0):
            return False
        if not (1.0 <= RISK_PARITY_TOLERANCE <= 10.0):
            return False
        if not (10 <= EFFICIENT_FRONTIER_POINTS <= 50):
            return False
        if not (1 <= OPTIMIZATION_LOOKBACK_MONTHS <= 24):
            return False
        if not (2 <= MIN_PORTFOLIO_POSITIONS <= 10):
            return False
        if not (10.0 <= MAX_SINGLE_ASSET_WEIGHT <= 50.0):
            return False
        
        # Validate return range
        min_return, max_return = TARGET_RETURN_RANGE
        if not (min_return < max_return and 0 <= min_return <= 50 and min_return <= max_return <= 100):
            return False
        
        return True
        
    except Exception:
        return False

def validate_advanced_config() -> bool:
    """Complete validation including stop loss and risk parity"""
    
    validations = [
        validate_config(),
        validate_portfolio_risk_config() if 'validate_portfolio_risk_config' in globals() else True,
        validate_stop_loss_config(),
        validate_risk_parity_config()
    ]
    
    return all(validations)

def get_stop_loss_summary() -> dict:
    """Get stop loss configuration summary"""
    
    return {
        'trailing_stops': {
            'initial_percent': TRAILING_STOP_INITIAL_PERCENT,
            'minimum_percent': TRAILING_STOP_MIN_PERCENT,
            'tightening_threshold': TRAILING_STOP_TIGHTENING_THRESHOLD
        },
        'volatility_stops': {
            'atr_multiplier': ATR_STOP_MULTIPLIER,
            'atr_periods': ATR_CALCULATION_PERIODS,
            'max_percent': VOLATILITY_STOP_MAX_PERCENT
        },
        'time_stops': {
            'enabled': TIME_STOP_ENABLED,
            'days': TIME_STOP_DAYS
        },
        'profit_protection': PROFIT_PROTECTION_LEVELS
    }

def get_risk_parity_summary() -> dict:
    """Get risk parity configuration summary"""
    
    return {
        'target_risk_contribution': TARGET_RISK_CONTRIBUTION_PERCENT,
        'rebalancing': {
            'frequency_days': REBALANCING_FREQUENCY_DAYS,
            'threshold_percent': REBALANCING_THRESHOLD_PERCENT,
            'auto_enabled': AUTO_REBALANCING_ENABLED
        },
        'optimization': {
            'lookback_months': OPTIMIZATION_LOOKBACK_MONTHS,
            'efficient_frontier_points': EFFICIENT_FRONTIER_POINTS,
            'max_single_weight': MAX_SINGLE_ASSET_WEIGHT
        },
        'risk_metrics': {
            'var_confidence': VAR_CONFIDENCE_LEVEL,
            'risk_free_rate': RISK_FREE_RATE
        }
    }
# ==========================================
# PAPER TRADING CONFIGURATION  
# ==========================================

# Paper Trading Settings
PAPER_TRADING_MODE = True
PAPER_TRADING_INITIAL_CAPITAL = float(os.getenv('PAPER_TRADING_CAPITAL', '100000'))
PAPER_TRADING_COMMISSION = 0.03  # 0.03% commission per trade
PAPER_TRADING_SLIPPAGE = 0.05    # 0.05% slippage simulation

# Order Management
MAX_ORDERS_PER_DAY = 20
ORDER_VALIDITY_HOURS = 24
AUTO_EXECUTE_SIGNALS = True

# Performance Tracking
TRACK_TRADE_PERFORMANCE = True
PERFORMANCE_REPORTING_FREQUENCY = 'daily'
BENCHMARK_SYMBOL = 'NIFTY50'

# Trade Execution Simulation
EXECUTION_DELAY_SECONDS = 2
MARKET_IMPACT_THRESHOLD = 50000
MARKET_IMPACT_PERCENT = 0.02

# Risk Controls for Paper Trading
PAPER_MAX_LOSS_PER_DAY = 5000
PAPER_MAX_POSITIONS = 10
PAPER_POSITION_SIZE_LIMIT = 0.15

# Reporting Configuration
GENERATE_DAILY_REPORTS = True
SAVE_TRADE_HISTORY = True
TRADE_HISTORY_RETENTION_DAYS = 90

# Add these variables to the end of existing config.py file

# ==========================================
# PAPER TRADING CONFIGURATION
# ==========================================

# Paper Trading Settings
PAPER_TRADING_MODE = True  # Enable paper trading by default
PAPER_TRADING_INITIAL_CAPITAL = float(os.getenv('PAPER_TRADING_CAPITAL', '100000'))
PAPER_TRADING_COMMISSION = 0.03  # 0.03% commission per trade
PAPER_TRADING_SLIPPAGE = 0.05    # 0.05% slippage simulation

# Order Management
MAX_ORDERS_PER_DAY = 20
ORDER_VALIDITY_HOURS = 24
AUTO_EXECUTE_SIGNALS = True  # Auto-convert signals to paper trades

# Performance Tracking
TRACK_TRADE_PERFORMANCE = True
PERFORMANCE_REPORTING_FREQUENCY = 'daily'  # daily, weekly, monthly
BENCHMARK_SYMBOL = 'NIFTY50'

# Trade Execution Simulation
EXECUTION_DELAY_SECONDS = 2  # Simulate execution delay
MARKET_IMPACT_THRESHOLD = 50000  # Position size above which market impact applies
MARKET_IMPACT_PERCENT = 0.02  # Market impact for large positions

# Risk Controls for Paper Trading
PAPER_MAX_LOSS_PER_DAY = 5000  # Max loss per day in paper trading
PAPER_MAX_POSITIONS = 10
PAPER_POSITION_SIZE_LIMIT = 0.15  # 15% max per position

# Reporting Configuration
GENERATE_DAILY_REPORTS = True
SAVE_TRADE_HISTORY = True
TRADE_HISTORY_RETENTION_DAYS = 90


# ==========================================
# ADD THESE LINES TO END OF config.py (Day 7B)
# ==========================================

# Dashboard Configuration
ENABLE_DASHBOARD = os.getenv('ENABLE_DASHBOARD', 'true').lower() == 'true'
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8080'))
DASHBOARD_AUTO_REFRESH = int(os.getenv('DASHBOARD_REFRESH', '30'))
DASHBOARD_THEME = os.getenv('DASHBOARD_THEME', 'dark')

# System Monitoring
ENABLE_SYSTEM_ALERTS = os.getenv('ENABLE_ALERTS', 'true').lower() == 'true'
ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
ALERT_WEBHOOK = os.getenv('ALERT_WEBHOOK', '')
SYSTEM_HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '300'))

# Automated Reporting
ENABLE_AUTO_REPORTS = os.getenv('ENABLE_AUTO_REPORTS', 'true').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '18:00')
WEEKLY_REPORT_DAY = os.getenv('WEEKLY_REPORT_DAY', 'Sunday')
REPORT_EMAIL_LIST = os.getenv('REPORT_EMAIL_LIST', '').split(',') if os.getenv('REPORT_EMAIL_LIST') else []

# Production Deployment
PRODUCTION_MODE = os.getenv('PRODUCTION_MODE', 'false').lower() == 'true'
MAX_CONCURRENT_USERS = int(os.getenv('MAX_CONCURRENT_USERS', '10'))
SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT', '60'))
BACKUP_INTERVAL_HOURS = int(os.getenv('BACKUP_INTERVAL', '24'))

# Security Settings
API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true'
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT', '100'))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))

# System Maintenance
AUTO_CLEANUP_ENABLED = os.getenv('AUTO_CLEANUP', 'true').lower() == 'true'
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '30'))
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '90'))
MAINTENANCE_WINDOW_START = os.getenv('MAINTENANCE_START', '02:00')
MAINTENANCE_WINDOW_END = os.getenv('MAINTENANCE_END', '04:00')

def get_dashboard_config() -> dict:
    """Get dashboard configuration"""
    return {
        'enabled': ENABLE_DASHBOARD,
        'host': DASHBOARD_HOST,
        'port': DASHBOARD_PORT,
        'refresh': DASHBOARD_AUTO_REFRESH,
        'theme': DASHBOARD_THEME,
        'production': PRODUCTION_MODE
    }

def get_monitoring_config() -> dict:
    """Get monitoring configuration"""
    return {
        'alerts_enabled': ENABLE_SYSTEM_ALERTS,
        'health_check_interval': SYSTEM_HEALTH_CHECK_INTERVAL,
        'auto_reports': ENABLE_AUTO_REPORTS,
        'report_time': DAILY_REPORT_TIME,
        'auto_cleanup': AUTO_CLEANUP_ENABLED
    }