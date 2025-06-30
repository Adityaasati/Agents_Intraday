import logging
import os
from datetime import datetime
import sys
from pathlib import Path

def setup_logger(name: str = 'nexus_trading', log_level: str = 'INFO') -> logging.Logger:
    """Setup logger with Windows-compatible formatting"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter (Windows-compatible, no special characters)
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - daily rotating
    today = datetime.now().strftime('%Y_%m_%d')
    log_file = log_dir / f'nexus_trading_{today}.log'
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler - only important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def setup_module_loggers():
    """Setup loggers for all modules"""
    
    # Main modules
    modules = [
        'nexus_trading',
        'database.enhanced_database_manager',
        'agents.technical_agent',
        'agents.fundamental_agent', 
        'agents.signal_agent',
        'agents.risk_agent',
        'agents.portfolio_agent',
        'utils.data_updater',
        'utils.system_monitor'
    ]
    
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    loggers = {}
    for module in modules:
        loggers[module] = setup_logger(module, log_level)
    
    return loggers

def log_system_start():
    """Log system startup information"""
    logger = setup_logger('nexus_trading.startup')
    
    logger.info("=" * 50)
    logger.info("NEXUS TRADING SYSTEM STARTING")
    logger.info("=" * 50)
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info(f"Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info("=" * 50)

def log_system_stop():
    """Log system shutdown"""
    logger = setup_logger('nexus_trading.shutdown')
    
    logger.info("=" * 50)
    logger.info("NEXUS TRADING SYSTEM STOPPING")
    logger.info("=" * 50)

class PerformanceLogger:
    """Log performance metrics without cluttering console"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.logger = setup_logger(f'nexus_trading.performance')
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(f"{self.operation_name} FAILED in {duration:.2f}s: {exc_val}")
        else:
            # Only log to file for performance, not console
            file_logger = logging.getLogger(f'nexus_trading.performance.file_only')
            file_logger.handlers = [h for h in self.logger.handlers if isinstance(h, logging.FileHandler)]
            file_logger.debug(f"{self.operation_name} completed in {duration:.2f}s")

# Error handling for critical errors
class CriticalErrorHandler:
    """Handle critical system errors"""
    
    @staticmethod
    def handle_critical_error(error: Exception, context: str = ""):
        """Log critical error and determine if system should continue"""
        logger = setup_logger('nexus_trading.critical')
        
        logger.critical(f"CRITICAL ERROR in {context}: {error}")
        logger.critical(f"Error Type: {type(error).__name__}")
        
        # Determine if error is recoverable
        recoverable_errors = (
            ConnectionError,
            TimeoutError,
            ValueError
        )
        
        if isinstance(error, recoverable_errors):
            logger.warning("Error appears recoverable, continuing operation...")
            return True
        else:
            logger.critical("Error appears non-recoverable, stopping operation...")
            return False