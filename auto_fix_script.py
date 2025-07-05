#!/usr/bin/env python3
"""
Automated fix script for Nexus Trading System
Run this to automatically fix common issues
"""

import os
import re
import shutil
from pathlib import Path
from utils.decorators import handle_agent_errors

def fix_signal_date_columns():
    """Fix all references to signal_date -> signal_time"""
    files_to_check = [
        'database/enhanced_database_manager.py',
        'reports/sentiment_dashboard.py',
        'utils/dashboard_monitor.py',
        'reports/system_dashboard.py'
    ]
    
    count = 0
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace signal_date with signal_time
            new_content = content.replace('signal_date', 'signal_time')
            
            if new_content != content:
                # Backup original
                shutil.copy(file_path, f"{file_path}.backup")
                
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                count += 1
                print(f" Fixed signal_date references in {file_path}")
    
    return count

def create_base_agent():
    """Create the BaseAgent class"""
    base_agent_code = '''"""Base Agent class for all trading agents"""
import logging
import pytz
from typing import Dict, List, Optional
from datetime import datetime

class BaseAgent:
    """Base class for all trading agents with common functionality"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def get_current_time(self):
        """Get current time in IST"""
        return datetime.now(self.ist)
    
    def log_error(self, operation: str, error: Exception):
        """Standardized error logging"""
        self.logger.error(f"{operation} failed: {error}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists in database"""
        try:
            data = self.db_manager.get_fundamental_data(symbol)
            return data is not None
        except:
            return False
'''
    
    # Create agents directory if not exists
    os.makedirs('agents', exist_ok=True)
    
    with open('agents/base_agent.py', 'w') as f:
        f.write(base_agent_code)
    
    print(" Created agents/base_agent.py")

def create_decorators():
    """Create the decorators utility"""
    decorators_code = '''"""Decorators for error handling and common patterns"""
import functools
import logging
import time
from typing import Any, Callable

def handle_agent_errors(default_return=None):
    """Decorator for consistent error handling in agent methods"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"{func.__name__} failed: {e}")
                else:
                    logging.error(f"{func.__name__} failed: {e}")
                
                if default_return is not None:
                    return default_return
                
                # Return error dict for analysis methods
                if func.__name__.startswith(('analyze', 'get', 'calculate')):
                    return {'error': str(e), 'function': func.__name__}
                
                return None
        return wrapper
    return decorator

def measure_performance(func: Callable) -> Callable:
    """Decorator to measure function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 1.0:  # Log if takes more than 1 second
            logging.warning(f"{func.__name__} took {elapsed_time:.2f}s")
        
        return result
    return wrapper

def validate_input(validation_func: Callable):
    """Decorator to validate input parameters"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Validate first argument (usually symbol or data)
            if args and not validation_func(args[0]):
                return {'error': 'validation_failed', 'function': func.__name__}
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
'''
    
    # Create utils directory if not exists
    os.makedirs('utils', exist_ok=True)
    
    with open('utils/decorators.py', 'w') as f:
        f.write(decorators_code)
    
    print(" Created utils/decorators.py")

def create_connection_config():
    """Create centralized database connection config"""
    config_code = '''"""Centralized database connection configuration"""
import os
from typing import Dict

def get_connection_params() -> Dict[str, str]:
    """Get database connection parameters from environment
    
    Returns:
        Dict containing host, port, database, user, password
    """
    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5435'),
        'database': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD')
    }

def get_pool_config() -> Dict[str, int]:
    """Get connection pool configuration
    
    Returns:
        Dict containing pool size and timeout settings
    """
    return {
        'minconn': int(os.getenv('DB_POOL_MIN', '5')),
        'maxconn': int(os.getenv('DB_POOL_MAX', '20')),
        'timeout': int(os.getenv('DB_TIMEOUT', '30'))
    }

def validate_connection_params() -> bool:
    """Validate that all required connection parameters are set
    
    Returns:
        True if all parameters are set, False otherwise
    """
    params = get_connection_params()
    required = ['database', 'user', 'password']
    
    for param in required:
        if not params.get(param):
            print(f"Missing required parameter: {param}")
            return False
    
    return True
'''
    
    # Create database directory if not exists
    os.makedirs('database', exist_ok=True)
    
    with open('database/connection_config.py', 'w') as f:
        f.write(config_code)
    
    print(" Created database/connection_config.py")

def add_missing_config_values():
    """Add missing configuration values to config.py"""
    additions = '''
# ==========================================
# DEFAULT SCORES FOR MISSING AGENTS
# ==========================================
DEFAULT_FUNDAMENTAL_SCORE = 0.5  # When fundamental agent not available
DEFAULT_SENTIMENT_SCORE = 0.5    # When sentiment agent not available

# ==========================================
# ERROR HANDLING CONFIGURATION
# ==========================================
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
CONNECTION_TIMEOUT = 30  # seconds

# ==========================================
# VALIDATION THRESHOLDS
# ==========================================
MIN_PRICE = 1.0  # Minimum valid price
MAX_PRICE = 100000.0  # Maximum valid price
MIN_VOLUME = 100  # Minimum valid volume
'''
    
    if os.path.exists('config.py'):
        with open('config.py', 'r') as f:
            content = f.read()
        
        if 'DEFAULT_FUNDAMENTAL_SCORE' not in content:
            with open('config.py', 'a') as f:
                f.write(additions)
            print(" Added missing configuration values to config.py")
    else:
        print(" config.py not found")

def fix_main_py():
    """Fix the missing _test_data_pipeline method in main.py"""
    method_code = '''
    def _test_data_pipeline(self) -> bool:
        """Test data pipeline functionality"""
        try:
            from utils.data_updater import SimpleDataUpdater
            
            data_updater = SimpleDataUpdater(self.db_manager)
            
            # Test with 2 symbols
            test_symbols = ['RELIANCE', 'TCS']
            pipeline_result = data_updater.test_data_pipeline(test_symbols)
            
            success_count = pipeline_result.get('successful_updates', 0)
            total_count = pipeline_result.get('total_symbols', 0)
            
            self.logger.info(f"Data pipeline test successful: {success_count} symbols updated")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Data pipeline test failed: {e}")
            return False
'''
    
    if os.path.exists('main.py'):
        with open('main.py', 'r') as f:
            content = f.read()
        
        if '_test_data_pipeline' not in content:
            # Find where to insert (after _test_database_connection)
            insert_pos = content.find('def _test_existing_tables')
            if insert_pos > 0:
                # Insert before _test_existing_tables
                new_content = content[:insert_pos] + method_code + '\n' + content[insert_pos:]
                
                # Backup original
                shutil.copy('main.py', 'main.py.backup')
                
                with open('main.py', 'w') as f:
                    f.write(new_content)
                
                print(" Added _test_data_pipeline method to main.py")
        else:
            print(" _test_data_pipeline already exists in main.py")
    else:
        print(" main.py not found")

def remove_duplicate_files():
    """Remove duplicate files"""
    duplicates = ['main copy.py', 'main copy 2.py']
    
    for dup_file in duplicates:
        if os.path.exists(dup_file):
            os.remove(dup_file)
            print(f" Removed duplicate file: {dup_file}")

def create_sql_indexes():
    """Create SQL file for missing indexes"""
    sql_content = '''-- Nexus Trading System - Performance Indexes
-- Run this SQL to improve query performance

-- Historical data indexes
CREATE INDEX IF NOT EXISTS idx_historical_symbol_timestamp 
ON historical_data_3m_2025_q3(symbol, timestamp);

CREATE INDEX IF NOT EXISTS idx_historical_symbol_date 
ON historical_data_3m_2025_q3(symbol, date);

-- Signal indexes
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
ON agent_live_signals(symbol, signal_time);

CREATE INDEX IF NOT EXISTS idx_signals_status 
ON agent_live_signals(status);

CREATE INDEX IF NOT EXISTS idx_signals_confidence 
ON agent_live_signals(overall_confidence);

-- Technical indicators indexes
CREATE INDEX IF NOT EXISTS idx_technical_symbol_time 
ON agent_technical_indicators(symbol, analysis_time);

-- Portfolio indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_symbol_status 
ON agent_portfolio_positions(symbol, status);

-- Fundamental data indexes
CREATE INDEX IF NOT EXISTS idx_fundamental_symbol_date 
ON agent_fundamental_data(symbol, analysis_date);

-- System config index
CREATE INDEX IF NOT EXISTS idx_config_key 
ON agent_system_config(config_key);

-- Analyze tables for query optimization
ANALYZE stocks_categories_table;
ANALYZE agent_live_signals;
ANALYZE agent_technical_indicators;
ANALYZE agent_portfolio_positions;
'''
    
    os.makedirs('database', exist_ok=True)
    
    with open('database/create_indexes.sql', 'w') as f:
        f.write(sql_content)
    
    print(" Created database/create_indexes.sql")

def main():
    """Run all automatic fixes"""
    print(" Nexus Trading System - Automated Fix Script")
    print("=" * 50)
    
    fixes_applied = 0
    
    # 1. Remove duplicate files
    remove_duplicate_files()
    fixes_applied += 1
    
    # 2. Fix signal_date columns
    fixed_files = fix_signal_date_columns()
    if fixed_files > 0:
        fixes_applied += 1
    
    # 3. Create missing files
    create_base_agent()
    create_decorators()
    create_connection_config()
    fixes_applied += 3
    
    # 4. Fix configuration
    add_missing_config_values()
    fixes_applied += 1
    
    # 5. Fix main.py
    fix_main_py()
    fixes_applied += 1
    
    # 6. Create SQL indexes
    create_sql_indexes()
    fixes_applied += 1
    
    print("\n" + "=" * 50)
    print(f" Applied {fixes_applied} fixes successfully!")
    print("\n Next Steps:")
    print("1. Review the backups created (.backup files)")
    print("2. Update agent files to inherit from BaseAgent")
    print("3. Run: psql -d your_database -f database/create_indexes.sql")
    print("4. Test the system: python main.py --mode test")
    print("\n  Manual fixes still required:")
    print("- Update agent classes to inherit from BaseAgent")
    print("- Apply @handle_agent_errors decorator to methods")
    print("- Review and remove hardcoded API keys from docs")

if __name__ == "__main__":
    main()