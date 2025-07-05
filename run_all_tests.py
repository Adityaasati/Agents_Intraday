"""
NEXUS TRADING SYSTEM - COMPREHENSIVE TEST RUNNER
===============================================

PURPOSE: Complete system validation and testing framework
DESCRIPTION: Tests all agents, database connectivity, and trading functionality
USAGE: python run_all_tests.py [--mode quick|full|specific] [--component agent_name]

Author: Trading System Team
"""

import sys
import os
import time
import argparse
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SystemTestRunner:
    """Comprehensive system test runner"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        self.start_time = time.time()
        self.db_manager = None
        
        # Test configuration
        self.test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        self.quick_test_symbols = ['RELIANCE', 'TCS']
        
        print("=" * 80)
        print("NEXUS TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Mode: {'Verbose' if verbose else 'Silent'}")
        print("=" * 80)
    
    def print_status(self, message: str, status: str = "INFO"):
        """Print formatted status message"""
        if not self.verbose:
            return
            
        symbols = {
            "PASS": "✅",
            "FAIL": "❌", 
            "WARN": "⚠️",
            "INFO": "ℹ️",
            "TEST": "🔬"
        }
        
        symbol = symbols.get(status, "•")
        print(f"{symbol} {message}")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> Dict:
        """Run individual test with error handling"""
        self.print_status(f"Running {test_name}...", "TEST")
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if isinstance(result, dict) and result.get('error'):
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'result': result,
                    'elapsed': elapsed,
                    'error': result.get('error')
                }
                self.print_status(f"{test_name}: FAILED - {result.get('error')}", "FAIL")
            else:
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'result': result,
                    'elapsed': elapsed
                }
                self.print_status(f"{test_name}: PASSED ({elapsed:.2f}s)", "PASS")
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            self.test_results[test_name] = {
                'status': 'ERROR',
                'result': None,
                'elapsed': elapsed,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
            self.print_status(f"{test_name}: ERROR - {error_msg}", "FAIL")
        
        return self.test_results[test_name]
    
    # =====================================
    # DATABASE TESTS
    # =====================================
    
    def test_database_connection(self) -> Dict:
        """Test database connectivity"""
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            self.db_manager = EnhancedDatabaseManager()
            
            # Test basic connection
            if not self.db_manager.test_connection():
                return {'error': 'Database connection test failed'}
            
            # Test pool status
            pool_status = self.db_manager.get_pool_status()
            if pool_status.get('status') != 'active':
                return {'error': f'Connection pool not active: {pool_status}'}
            
            # Test basic query
            health_data = self.db_manager.get_system_health()
            if 'total_symbols' not in health_data:
                return {'error': 'System health check failed'}
            
            return {
                'connection': 'OK',
                'pool_status': pool_status,
                'total_symbols': health_data.get('total_symbols', 0),
                'available_quarters': health_data.get('available_quarters', 0)
            }
            
        except Exception as e:
            return {'error': f'Database test failed: {str(e)}'}
    
    def test_data_retrieval(self) -> Dict:
        """Test data retrieval functionality"""
        try:
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            # Test symbol data retrieval
            symbols_data = self.db_manager.get_symbols_from_categories(limit=5)
            if not symbols_data:
                return {'error': 'No symbol data retrieved'}
            
            # Test fundamental data
            test_symbol = symbols_data[0]['symbol']
            fundamental_data = self.db_manager.get_fundamental_data(test_symbol)
            if not fundamental_data:
                return {'error': f'No fundamental data for {test_symbol}'}
            
            # Test historical data
            historical_data = self.db_manager.get_historical_data(test_symbol, 10)
            if historical_data.empty:
                return {'error': f'No historical data for {test_symbol}'}
            
            return {
                'symbols_retrieved': len(symbols_data),
                'test_symbol': test_symbol,
                'fundamental_columns': len(fundamental_data.keys()),
                'historical_records': len(historical_data),
                'date_range': f"{historical_data['date'].min()} to {historical_data['date'].max()}"
            }
            
        except Exception as e:
            return {'error': f'Data retrieval test failed: {str(e)}'}
    
    # =====================================
    # AGENT TESTS
    # =====================================
    
    def test_technical_agent(self) -> Dict:
        """Test Technical Analysis Agent"""
        try:
            from agents.technical_agent import TechnicalAgent
            
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            agent = TechnicalAgent(self.db_manager)
            test_symbol = self.quick_test_symbols[0]
            
            # Test single symbol analysis
            result = agent.analyze_symbol(test_symbol)
            if 'error' in result:
                return {'error': f'Technical analysis failed: {result["error"]}'}
            
            # Validate result structure
            required_fields = ['technical_score', 'indicators', 'buy_signal', 'sell_signal']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                return {'error': f'Missing fields in technical result: {missing_fields}'}
            
            # Test batch analysis
            batch_result = agent.analyze_multiple_symbols(self.quick_test_symbols)
            if len(batch_result) == 0:
                return {'error': 'Batch analysis returned no results'}
            
            return {
                'single_analysis': 'OK',
                'test_symbol': test_symbol,
                'technical_score': result.get('technical_score'),
                'buy_signal': result.get('buy_signal'),
                'batch_symbols_analyzed': len(batch_result),
                'indicators_calculated': len(result.get('indicators', {}))
            }
            
        except Exception as e:
            return {'error': f'Technical agent test failed: {str(e)}'}
    
    def test_fundamental_agent(self) -> Dict:
        """Test Fundamental Analysis Agent"""
        try:
            from agents.fundamental_agent import FundamentalAgent
            
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            agent = FundamentalAgent(self.db_manager)
            test_symbol = self.quick_test_symbols[0]
            
            # Test fundamental analysis
            result = agent.analyze_symbol_fundamentals(test_symbol)
            if 'error' in result:
                return {'error': f'Fundamental analysis failed: {result["error"]}'}
            
            # Validate result structure
            required_fields = ['fundamental_score', 'quality_score', 'valuation_score']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                return {'error': f'Missing fields in fundamental result: {missing_fields}'}
            
            return {
                'analysis': 'OK',
                'test_symbol': test_symbol,
                'fundamental_score': result.get('fundamental_score'),
                'quality_score': result.get('quality_score'),
                'valuation_score': result.get('valuation_score'),
                'category': result.get('category', 'Unknown')
            }
            
        except Exception as e:
            return {'error': f'Fundamental agent test failed: {str(e)}'}
    
    def test_signal_agent(self) -> Dict:
        """Test Signal Generation Agent"""
        try:
            from agents.signal_agent import SignalAgent
            
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            agent = SignalAgent(self.db_manager)
            
            # Test single signal generation
            test_symbol = self.quick_test_symbols[0]
            signal = agent.generate_signal(test_symbol)
            if 'error' in signal:
                return {'error': f'Signal generation failed: {signal["error"]}'}
            
            # Validate signal structure
            required_fields = ['symbol', 'signal_type', 'overall_confidence', 'entry_price']
            missing_fields = [field for field in required_fields if field not in signal]
            if missing_fields:
                return {'error': f'Missing fields in signal: {missing_fields}'}
            
            # Test batch signal generation
            signals = agent.generate_signals(self.quick_test_symbols, limit=2)
            
            return {
                'single_signal': 'OK',
                'test_symbol': test_symbol,
                'signal_type': signal.get('signal_type'),
                'confidence': signal.get('overall_confidence'),
                'batch_signals_generated': len(signals),
                'entry_price': signal.get('entry_price')
            }
            
        except Exception as e:
            return {'error': f'Signal agent test failed: {str(e)}'}
    
    def test_risk_agent(self) -> Dict:
        """Test Risk Management Agent"""
        try:
            from agents.risk_agent import RiskAgent
            
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            agent = RiskAgent(self.db_manager)
            
            # Create mock signal for testing
            mock_signal = {
                'symbol': self.quick_test_symbols[0],
                'signal_type': 'BUY',
                'entry_price': 2500.0,
                'overall_confidence': 0.75,
                'category': 'A',
                'stop_loss': 2375.0,
                'target_price': 2750.0
            }
            
            # Test position sizing
            position_result = agent.calculate_position_size(mock_signal)
            if 'error' in position_result:
                return {'error': f'Position sizing failed: {position_result["error"]}'}
            
            # Test stop loss calculation
            stop_loss_result = agent.determine_stop_loss(
                mock_signal['symbol'], 
                mock_signal['entry_price']
            )
            if 'error' in stop_loss_result:
                return {'error': f'Stop loss calculation failed: {stop_loss_result["error"]}'}
            
            return {
                'position_sizing': 'OK',
                'stop_loss_calculation': 'OK',
                'recommended_shares': position_result.get('recommended_shares'),
                'position_value': position_result.get('recommended_position_value'),
                'risk_percent': position_result.get('risk_percent_of_capital'),
                'stop_loss_price': stop_loss_result.get('stop_loss_price')
            }
            
        except Exception as e:
            return {'error': f'Risk agent test failed: {str(e)}'}
    
    # =====================================
    # INTEGRATION TESTS
    # =====================================
    
    def test_complete_signal_flow(self) -> Dict:
        """Test complete signal generation flow"""
        try:
            from agents.signal_agent import SignalAgent
            
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            agent = SignalAgent(self.db_manager)
            test_symbol = self.quick_test_symbols[0]
            
            # Generate complete signal
            signal = agent.generate_signal(test_symbol)
            if 'error' in signal:
                return {'error': f'Complete flow failed: {signal["error"]}'}
            
            # Test signal storage (if live signal storage is available)
            try:
                signal_uuid = f"test_{test_symbol}_{int(time.time())}"
                signal_data = {
                    'symbol': signal['symbol'],
                    'signal_uuid': signal_uuid,
                    'signal_type': signal['signal_type'],
                    'signal_time': datetime.now(),
                    'entry_price': signal.get('entry_price', 0.0),
                    'stop_loss': signal.get('stop_loss', 0.0),
                    'target_price': signal.get('target_price', 0.0),
                    'overall_confidence': signal.get('overall_confidence', 0.5),
                    'technical_score': signal.get('technical_score', 0.5),
                    'fundamental_score': signal.get('fundamental_score', 0.5),
                    'sentiment_score': signal.get('sentiment_score', 0.5),
                    'recommended_position_size': float(signal.get('recommended_position_size', 10000.0)),
                    'status': 'ACTIVE',
                    'primary_reasoning': signal.get('primary_reasoning', 'Test signal'),
                    'created_at': datetime.now()
                }
                
                storage_result = self.db_manager.store_live_signal(signal_data)
                signal_storage_status = 'OK' if storage_result else 'FAILED'
                
            except Exception as e:
                signal_storage_status = f'ERROR: {str(e)}'
            
            return {
                'complete_flow': 'OK',
                'test_symbol': test_symbol,
                'signal_generated': signal.get('signal_type'),
                'confidence': signal.get('overall_confidence'),
                'has_entry_price': 'entry_price' in signal,
                'has_stop_loss': 'stop_loss' in signal,
                'has_target_price': 'target_price' in signal,
                'signal_storage': signal_storage_status,
                'reasoning': signal.get('primary_reasoning', 'No reasoning')[:50]
            }
            
        except Exception as e:
            return {'error': f'Complete signal flow test failed: {str(e)}'}
    
    def test_paper_trading_setup(self) -> Dict:
        """Test paper trading functionality"""
        try:
            # Test if paper trading modules are available
            try:
                from agents.portfolio_agent import PortfolioAgent
                portfolio_available = True
            except ImportError:
                portfolio_available = False
            
            if not portfolio_available:
                return {
                    'paper_trading': 'NOT_AVAILABLE',
                    'message': 'Portfolio agent not implemented yet'
                }
            
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            agent = PortfolioAgent(self.db_manager)
            
            # Test portfolio summary
            summary = agent.get_paper_portfolio_summary()
            if 'error' in summary:
                return {'error': f'Portfolio summary failed: {summary["error"]}'}
            
            return {
                'paper_trading': 'AVAILABLE',
                'portfolio_summary': 'OK',
                'total_value': summary.get('total_portfolio_value', 0),
                'cash_available': summary.get('cash_available', 0),
                'open_positions': summary.get('open_positions', 0)
            }
            
        except Exception as e:
            return {'error': f'Paper trading test failed: {str(e)}'}
    
    # =====================================
    # PERFORMANCE TESTS
    # =====================================
    
    def test_performance_benchmarks(self) -> Dict:
        """Test system performance benchmarks"""
        try:
            if not self.db_manager:
                return {'error': 'Database not initialized'}
            
            # Test database query performance
            start_time = time.time()
            symbols_data = self.db_manager.get_symbols_from_categories(limit=100)
            db_query_time = time.time() - start_time
            
            # Test technical analysis performance
            from agents.technical_agent import TechnicalAgent
            tech_agent = TechnicalAgent(self.db_manager)
            
            start_time = time.time()
            tech_result = tech_agent.analyze_symbol(self.quick_test_symbols[0])
            tech_analysis_time = time.time() - start_time
            
            # Test signal generation performance
            from agents.signal_agent import SignalAgent
            signal_agent = SignalAgent(self.db_manager)
            
            start_time = time.time()
            signal = signal_agent.generate_signal(self.quick_test_symbols[0])
            signal_generation_time = time.time() - start_time
            
            # Performance thresholds
            thresholds = {
                'db_query_time': 2.0,  # seconds
                'tech_analysis_time': 5.0,  # seconds
                'signal_generation_time': 10.0  # seconds
            }
            
            performance_status = 'OK'
            warnings = []
            
            if db_query_time > thresholds['db_query_time']:
                warnings.append(f'DB query slow: {db_query_time:.2f}s')
                performance_status = 'SLOW'
            
            if tech_analysis_time > thresholds['tech_analysis_time']:
                warnings.append(f'Technical analysis slow: {tech_analysis_time:.2f}s')
                performance_status = 'SLOW'
            
            if signal_generation_time > thresholds['signal_generation_time']:
                warnings.append(f'Signal generation slow: {signal_generation_time:.2f}s')
                performance_status = 'SLOW'
            
            return {
                'performance_status': performance_status,
                'db_query_time': round(db_query_time, 3),
                'tech_analysis_time': round(tech_analysis_time, 3),
                'signal_generation_time': round(signal_generation_time, 3),
                'symbols_retrieved': len(symbols_data),
                'warnings': warnings,
                'thresholds': thresholds
            }
            
        except Exception as e:
            return {'error': f'Performance test failed: {str(e)}'}
    
    # =====================================
    # MAIN TEST RUNNERS
    # =====================================
    
    def run_quick_tests(self) -> Dict:
        """Run essential quick tests"""
        self.print_status("Running Quick Test Suite", "INFO")
        print("-" * 60)
        
        # Essential tests only
        tests = [
            ('Database Connection', self.test_database_connection),
            ('Data Retrieval', self.test_data_retrieval),
            ('Technical Agent', self.test_technical_agent),
            ('Signal Generation', self.test_signal_agent),
            ('Complete Signal Flow', self.test_complete_signal_flow)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        return self.get_test_summary()
    
    def run_full_tests(self) -> Dict:
        """Run comprehensive test suite"""
        self.print_status("Running Complete Test Suite", "INFO")
        print("-" * 60)
        
        # All available tests
        tests = [
            ('Database Connection', self.test_database_connection),
            ('Data Retrieval', self.test_data_retrieval),
            ('Technical Agent', self.test_technical_agent),
            ('Fundamental Agent', self.test_fundamental_agent),
            ('Signal Agent', self.test_signal_agent),
            ('Risk Agent', self.test_risk_agent),
            ('Complete Signal Flow', self.test_complete_signal_flow),
            ('Paper Trading Setup', self.test_paper_trading_setup),
            ('Performance Benchmarks', self.test_performance_benchmarks)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        return self.get_test_summary()
    
    def run_specific_test(self, component: str) -> Dict:
        """Run tests for specific component"""
        component_tests = {
            'database': [
                ('Database Connection', self.test_database_connection),
                ('Data Retrieval', self.test_data_retrieval)
            ],
            'technical': [
                ('Technical Agent', self.test_technical_agent)
            ],
            'fundamental': [
                ('Fundamental Agent', self.test_fundamental_agent)
            ],
            'signal': [
                ('Signal Agent', self.test_signal_agent),
                ('Complete Signal Flow', self.test_complete_signal_flow)
            ],
            'risk': [
                ('Risk Agent', self.test_risk_agent)
            ],
            'portfolio': [
                ('Paper Trading Setup', self.test_paper_trading_setup)
            ],
            'performance': [
                ('Performance Benchmarks', self.test_performance_benchmarks)
            ]
        }
        
        tests = component_tests.get(component.lower())
        if not tests:
            self.print_status(f"Unknown component: {component}", "FAIL")
            return {'error': f'Unknown component: {component}'}
        
        self.print_status(f"Running {component.title()} Tests", "INFO")
        print("-" * 60)
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        return self.get_test_summary()
    
    def get_test_summary(self) -> Dict:
        """Generate comprehensive test summary"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'PASSED'])
        failed_tests = len([r for r in self.test_results.values() if r['status'] in ['FAILED', 'ERROR']])
        
        total_time = time.time() - self.start_time
        
        summary = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': round((passed_tests / total_tests * 100), 1) if total_tests > 0 else 0,
            'total_execution_time': round(total_time, 2),
            'test_results': self.test_results
        }
        
        return summary
    
    def print_final_summary(self, summary: Dict):
        """Print formatted final summary"""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Execution Time: {summary['total_execution_time']}s")
        
        if summary['failed'] > 0:
            print("\n" + "-" * 40)
            print("FAILED TESTS:")
            print("-" * 40)
            for test_name, result in summary['test_results'].items():
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"❌ {test_name}: {result.get('error', 'Unknown error')}")
        
        print("\n" + "-" * 40)
        print("DETAILED RESULTS:")
        print("-" * 40)
        for test_name, result in summary['test_results'].items():
            status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_symbol} {test_name}: {result['status']} ({result['elapsed']:.2f}s)")
        
        print("\n" + "=" * 80)
        overall_status = "SUCCESS" if summary['failed'] == 0 else "FAILED"
        print(f"OVERALL TEST STATUS: {overall_status}")
        print("=" * 80)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Nexus Trading System - Test Runner')
    parser.add_argument('--mode', choices=['quick', 'full', 'specific'], 
                       default='quick', help='Test execution mode')
    parser.add_argument('--component', type=str, 
                       help='Specific component to test (for specific mode)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--silent', action='store_true',
                       help='Silent mode (minimal output)')
    
    args = parser.parse_args()
    
    # Initialize test runner
    verbose = args.verbose and not args.silent
    runner = SystemTestRunner(verbose=verbose)
    
    try:
        # Run tests based on mode
        if args.mode == 'quick':
            summary = runner.run_quick_tests()
        elif args.mode == 'full':
            summary = runner.run_full_tests()
        elif args.mode == 'specific':
            if not args.component:
                print("Error: --component required for specific mode")
                print("Available components: database, technical, fundamental, signal, risk, portfolio, performance")
                sys.exit(1)
            summary = runner.run_specific_test(args.component)
        
        # Print summary
        if verbose:
            runner.print_final_summary(summary)
        else:
            # Silent mode - just success/failure
            status = "SUCCESS" if summary['failed'] == 0 else "FAILED"
            print(f"Test Status: {status} ({summary['passed']}/{summary['total_tests']} passed)")
        
        # Exit with appropriate code
        sys.exit(0 if summary['failed'] == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest runner failed: {str(e)}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()