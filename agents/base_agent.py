"""Base Agent class for all trading agents"""
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
