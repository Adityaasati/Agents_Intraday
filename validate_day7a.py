#!/usr/bin/env python3
"""
FILE: validate_day7a.py
PURPOSE: Validate Day 7A Performance Optimization Implementation

USAGE:
- python validate_day7a.py
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import config

class Day7AValidator:
    """Validate Day 7A performance optimizations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    def run_validation(self) -> bool:
        """Run complete Day 7A validation"""
        
        print("=" * 60)
        print("DAY 7A PERFORMANCE OPTIMIZATION VALIDATION")
        print("=" * 60)
        
        validations = [
            ("Configuration Check", self.validate_config),
            ("Database Pool", self.validate_database_pool),
            ("Caching System", self.validate_caching),
            ("Batch Processing", self.validate_batch_processing),
            ("Performance Monitoring", self.validate_monitoring),
            ("New Modes", self.validate_new_modes)
        ]
        
        for test_name, test_func in validations:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                self.results[test_name] = result
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  Status: {status}")
            except Exception as e:
                self.results[test_name] = False
                print(f"  Status: ✗ ERROR - {e}")
        
        self.print_summary()
        return all(self.results.values())
    
    def validate_config(self) -> bool:
        """Validate performance configuration additions"""
        try:
            # Check new performance settings exist in config
            required_settings = [
                'DB_CONNECTION_POOL_SIZE', 'BATCH_SIZE_SYMBOLS', 'MAX_PARALLEL_WORKERS',
                'ENABLE_INDICATOR_CACHE', 'ENABLE_PERFORMANCE_MONITORING'
            ]
            
            missing = []
            for setting in required_settings:
                if not hasattr(config, setting):
                    missing.append(setting)
            
            if missing:
                print(f"  Missing config settings: {missing}")
                return False
            
            # Test helper functions
            batch_size = config.get_optimal_batch_size(100)
            worker_count = config.get_worker_count(25)
            
            print(f"  Optimal batch size for 100 symbols: {batch_size}")
            print(f"  Worker count for batch 25: {worker_count}")
            
            return batch_size > 0 and worker_count > 0
            
        except Exception as e:
            print(f"  Config validation error: {e}")
            return False
    
    def validate_database_pool(self) -> bool:
        """Validate database connection pool additions"""
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            
            # Check if performance methods were added
            required_methods = ['get_performance_stats', 'cleanup_cache', 'optimize_database']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(db_manager, method):
                    missing_methods.append(method)
            
            if missing_methods:
                print(f"  Missing database methods: {missing_methods}")
                return False
            
            # Test performance stats
            stats = db_manager.get_performance_stats()
            print(f"  Database pool status: {stats.get('pool_status', 'unknown')}")
            
            return 'pool_status' in stats
            
        except Exception as e:
            print(f"  Database pool error: {e}")
            return False
    
    def validate_caching(self) -> bool:
        """Validate caching system additions"""
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            
            # Test if optimized methods exist
            if not hasattr(db_manager, 'get_historical_data_optimized'):
                print("  Missing optimized data retrieval method")
                return False
            
            # Test cache functionality
            start_time = time.time()
            data1 = db_manager.get_historical_data_optimized('RELIANCE')
            first_call_time = time.time() - start_time
            
            start_time = time.time()
            data2 = db_manager.get_historical_data_optimized('RELIANCE')
            cached_call_time = time.time() - start_time
            
            print(f"  First call: {first_call_time:.3f}s, Cached: {cached_call_time:.3f}s")
            
            # Test batch method
            if hasattr(db_manager, 'get_multiple_symbols_data'):
                batch_data = db_manager.get_multiple_symbols_data(['RELIANCE', 'TCS'])
                print(f"  Batch retrieval: {len(batch_data)} symbols")
                return len(batch_data) > 0
            
            return not data1.empty
            
        except Exception as e:
            print(f"  Caching error: {e}")
            return False
    
    def validate_batch_processing(self) -> bool:
        """Validate batch processing additions"""
        try:
            from agents.technical_agent import TechnicalAgent
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            tech_agent = TechnicalAgent(db_manager)
            
            # Check if optimized methods were added
            required_methods = ['analyze_symbol_with_data', 'batch_analyze_symbols']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(tech_agent, method):
                    missing_methods.append(method)
            
            if missing_methods:
                print(f"  Missing technical agent methods: {missing_methods}")
                return False
            
            # Test batch analysis
            test_data = db_manager.get_historical_data('RELIANCE')
            if not test_data.empty:
                symbols_data = {'RELIANCE': test_data, 'TCS': test_data}
                batch_results = tech_agent.batch_analyze_symbols(symbols_data)
                print(f"  Batch analysis: {len(batch_results)} symbols processed")
                return len(batch_results) == 2
            
            return True
            
        except Exception as e:
            print(f"  Batch processing error: {e}")
            return False
    
    def validate_monitoring(self) -> bool:
        """Validate performance monitoring"""
        try:
            from utils.performance_monitor import get_performance_monitor
            
            monitor = get_performance_monitor()
            
            # Test monitoring functions
            monitor.start_monitoring()
            time.sleep(2)  # Let it collect some data
            
            stats = monitor.get_current_stats()
            summary = monitor.get_performance_summary()
            
            monitor.stop_monitoring()
            
            required_stats = ['cpu_usage', 'memory_usage_mb']
            missing_stats = [stat for stat in required_stats if stat not in stats]
            
            if missing_stats:
                print(f"  Missing monitoring stats: {missing_stats}")
                return False
            
            print(f"  CPU: {stats.get('cpu_usage', 0)}%, Memory: {stats.get('memory_usage_mb', 0)}MB")
            print(f"  Status: {summary.get('status', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"  Monitoring error: {e}")
            return False
    
    def validate_new_modes(self) -> bool:
        """Validate new processing modes"""
        try:
            from main import NexusTradingSystem
            
            system = NexusTradingSystem()
            
            # Check if new methods were added
            required_methods = ['_performance_test_mode', '_batch_processing_mode', '_optimization_mode']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(system, method):
                    missing_methods.append(method)
            
            if missing_methods:
                print(f"  Missing main system methods: {missing_methods}")
                return False
            
            print(f"  New processing modes available: performance, batch, optimize")
            return True
            
        except Exception as e:
            print(f"  New modes error: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary"""
        
        print("\n" + "=" * 60)
        print("DAY 7A VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        for test_name, result in self.results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:25}: {status}")
        
        print("-" * 60)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 DAY 7A PERFORMANCE OPTIMIZATION COMPLETE!")
            print("✓ Configuration enhanced with performance settings")
            print("✓ Database connection pooling and caching added")
            print("✓ Batch processing and parallel execution enabled")
            print("✓ Performance monitoring system active")
            print("✓ New processing modes: performance, batch, optimize")
            print("\n🚀 READY FOR DAY 7B!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} validations failed")
            print("Please fix issues before proceeding")
        
        print("=" * 60)

def main():
    """Run Day 7A validation"""
    
    validator = Day7AValidator()
    success = validator.run_validation()
    
    if success:
        print("\n✅ Day 7A Performance Optimization Validated!")
        print("\nNew commands available:")
        print("- python main.py --mode performance")
        print("- python main.py --mode batch") 
        print("- python main.py --mode optimize")
    else:
        print("\n❌ Day 7A Validation Failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()