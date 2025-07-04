"""Decorators for error handling and common patterns"""
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
