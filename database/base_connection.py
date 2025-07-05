# database/base_connection.py
import os

class BaseConnection:
    """Shared database connection logic"""
    
    @property
    def connection_params(self):
        return {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': os.getenv('DATABASE_PORT', '5435'),
            'database': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD')
        }