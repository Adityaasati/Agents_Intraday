"""Centralized database connection configuration"""
import os
from typing import Dict

def get_connection_params() -> Dict[str, str]:
    """Get database connection parameters from environment
    
    Returns:
        Dict containing host, port, database, user, password
    """
    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5432'),
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
