#!/usr/bin/env python3
"""
Test Execution Manager - Prevents connection pool exhaustion during tests
Integrates with existing test methods without changing their logic
"""

import time
import os
import logging
from contextlib import contextmanager
from typing import Dict, Callable, Any

class TestExecutionManager:
    """Manages test execution to prevent connection pool exhaustion"""
    
    def __init__(self, db_manager, logger=None):
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger('nexus_trading.test_manager')
        
        # Get settings from environment
        self.test_delay = float(os.getenv('TEST_DELAY_SECONDS', '2.0'))
        self.enable_sequential = os.getenv('ENABLE_SEQUENTIAL_TESTING', 'true').lower() == 'true'
        self.max_retries = int(os.getenv('MAX_TEST_RETRIES', '3'))
        
    @contextmanager
    def test_context(self, test_name: str):
        """Context manager for individual tests with cleanup and monitoring"""
        self.logger.info(f"Starting test: {test_name}")
        start_time = time.time()
        
        try:
            # Check pool status before test if available
            if hasattr(self.db_manager, 'get_pool_status'):
                pool_status = self.db_manager.get_pool_status()
                exhaustion_count = pool_status.get('pool_exhaustion_count', 0)
                
                if exhaustion_count > 10:
                    self.logger.warning(f"High pool exhaustion detected ({exhaustion_count}), waiting for recovery...")
                    time.sleep(5)  # Allow pool to recover
            
            yield
            
        except Exception as e:
            self.logger.error(f"Test {test_name} failed: {e}")
            raise
        finally:
            execution_time = time.time() - start_time
            self.logger.info(f"Test {test_name} completed in {execution_time:.2f}s")
            
            # Brief pause between tests to allow connection cleanup
            if self.enable_sequential and self.test_delay > 0:
                time.sleep(self.test_delay)
    
    def execute_test_with_retry(self, test_name: str, test_func: Callable) -> bool:
        """Execute a test with retry logic for pool exhaustion"""
        
        for attempt in range(self.max_retries):
            try:
                with self.test_context(f"{test_name} (attempt {attempt + 1})"):
                    result = test_func()
                    if result:
                        return True
                    elif attempt == self.max_retries - 1:
                        self.logger.warning(f"Test {test_name} failed after {self.max_retries} attempts")
                        return False
                    else:
                        self.logger.warning(f"Test {test_name} failed, retrying...")
                        time.sleep(2)  # Wait before retry
                        
            except Exception as e:
                if "connection pool exhausted" in str(e).lower() and attempt < self.max_retries - 1:
                    self.logger.warning(f"Pool exhausted in {test_name}, retrying in 5 seconds...")
                    time.sleep(5)  # Longer wait for pool exhaustion
                    continue
                elif attempt == self.max_retries - 1:
                    self.logger.error(f"Test {test_name} failed permanently: {e}")
                    return False
                else:
                    raise
        
        return False
    
    def get_pool_health_status(self) -> Dict:
        """Get current pool health for monitoring"""
        if hasattr(self.db_manager, 'get_pool_status'):
            pool_status = self.db_manager.get_pool_status()
            
            # Determine health level
            exhaustion_count = pool_status.get('pool_exhaustion_count', 0)
            if exhaustion_count == 0:
                health = 'excellent'
            elif exhaustion_count < 5:
                health = 'good'
            elif exhaustion_count < 15:
                health = 'warning'
            else:
                health = 'critical'
            
            return {
                'health': health,
                'exhaustion_count': exhaustion_count,
                'queries_executed': pool_status.get('queries_executed', 0),
                'avg_query_time': pool_status.get('avg_query_time', 0)
            }
        else:
            return {'health': 'unknown', 'message': 'Pool monitoring not available'}
    
    def should_use_sequential_execution(self) -> bool:
        """Determine if sequential execution should be used"""
        if not self.enable_sequential:
            return False
        
        # Check pool health
        health_status = self.get_pool_health_status()
        if health_status['health'] in ['critical', 'warning']:
            self.logger.info("Using sequential execution due to pool health concerns")
            return True
        
        return True  # Default to sequential if enabled
    
    def log_test_summary(self, test_results: Dict):
        """Log test execution summary with pool statistics"""
        
        # Calculate success rate
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() if result)
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.logger.info(f"Test Execution Summary:")
        self.logger.info(f"  Total tests: {total_tests}")
        self.logger.info(f"  Successful: {successful_tests}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        # Log pool statistics
        health_status = self.get_pool_health_status()
        self.logger.info(f"  Pool health: {health_status['health']}")
        self.logger.info(f"  Pool exhaustion events: {health_status.get('exhaustion_count', 'unknown')}")
        
        if health_status.get('exhaustion_count', 0) > 0:
            self.logger.warning("⚠️  Connection pool exhaustion detected during tests")
            self.logger.warning("   Consider increasing DB_POOL_MAX in .env file")
        
        # Recommendations based on results
        if success_rate < 80:
            self.logger.warning("⚠️  Low test success rate - consider:")
            self.logger.warning("   - Increasing TEST_DELAY_SECONDS")
            self.logger.warning("   - Enabling ENABLE_SEQUENTIAL_TESTING=true")
            self.logger.warning("   - Increasing DB_POOL_MAX")