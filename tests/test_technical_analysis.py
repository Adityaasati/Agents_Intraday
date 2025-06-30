"""
FILE: tests/test_technical_analysis.py
LOCATION: /tests/ directory
PURPOSE: Technical Analysis Test Suite - Comprehensive testing for technical analysis capabilities

DESCRIPTION:
- Complete test suite for validating technical analysis functionality
- Tests database connections, indicator calculations, signal generation
- Validates error handling and edge cases
- Performance testing for speed requirements
- Integration testing with synthetic and real data
- Can be run independently to validate technical analysis system

DEPENDENCIES:
- database/enhanced_database_manager.py (for database testing)
- agents/technical_agent.py (for technical analysis testing)
- utils/technical_calculations.py (for utility testing)
- config.py (for system parameters)

USAGE:
- Run directly: python tests/test_technical_analysis.py
- Called by main.py in test mode
- Used for continuous integration validation
"""

#!/usr/bin/env python3
"""
Technical Analysis Test Suite
Comprehensive testing for technical analysis capabilities
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from database.enhanced_database_manager import EnhancedDatabaseManager
from agents.technical_agent import TechnicalAgent
from utils.technical_calculations import TechnicalCalculations
import config

class TechnicalAnalysisTestSuite:
    """Comprehensive test suite for technical analysis"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.db_manager = None
        self.tech_agent = None
        self.test_results = {}
        
    def run_all_tests(self) -> dict:
        """Run all technical analysis tests"""
        
        print("=" * 60)
        print("TECHNICAL ANALYSIS TEST SUITE")
        print("=" * 60)
        
        tests = [
            ('Database Connection', self.test_database_connection),
            ('Technical Calculations', self.test_technical_calculations),
            ('Technical Agent Initialization', self.test_agent_initialization),
            ('Indicator Calculations', self.test_indicator_calculations),
            ('Signal Generation', self.test_signal_generation),
            ('Multiple Symbol Analysis', self.test_multiple_symbols),
            ('Data Storage', self.test_data_storage),
            ('Performance Testing', self.test_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\nRunning: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "PASS" if result else "FAIL"
                print(f"Result: {status}")
            except Exception as e:
                self.test_results[test_name] = False
                print(f"Result: FAIL ({e})")
        
        self._print_summary()
        return self.test_results
    
    def test_database_connection(self) -> bool:
        """Test database connection"""
        
        try:
            self.db_manager = EnhancedDatabaseManager()
            return self.db_manager.test_connection()
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def test_technical_calculations(self) -> bool:
        """Test technical calculation utilities"""
        
        try:
            # Create sample OHLCV data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')  # Daily data for calculations test
            np.random.seed(42)  # For reproducible results
            
            # Generate realistic price data
            base_price = 100
            prices = [base_price]
            
            for i in range(99):
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1))  # Ensure positive prices
            
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * np.random.uniform(1.0, 1.02) for p in prices],
                'low': [p * np.random.uniform(0.98, 1.0) for p in prices],
                'close': prices,
                'volume': np.random.randint(100000, 1000000, 100)
            })
            
            # Test data validation
            if not TechnicalCalculations.validate_ohlcv_data(df):
                print("OHLCV data validation failed")
                return False
            
            # Test data cleaning
            cleaned_df = TechnicalCalculations.clean_ohlcv_data(df)
            if cleaned_df.empty:
                print("Data cleaning failed")
                return False
            
            # Test support/resistance calculation
            sr_levels = TechnicalCalculations.find_support_resistance_levels(df)
            if not isinstance(sr_levels, dict):
                print("Support/resistance calculation failed")
                return False
            
            print(f"Generated {len(df)} data points for testing")
            print(f"Found {len(sr_levels.get('support_levels', []))} support levels")
            print(f"Found {len(sr_levels.get('resistance_levels', []))} resistance levels")
            
            return True
            
        except Exception as e:
            print(f"Technical calculations test failed: {e}")
            return False
    
    def test_agent_initialization(self) -> bool:
        """Test technical agent initialization"""
        
        try:
            self.tech_agent = TechnicalAgent(self.db_manager)
            
            # Test pandas_ta availability
            if self.tech_agent.use_pandas_ta:
                print("pandas_ta is available and will be used")
            else:
                print("pandas_ta not available, using manual calculations")
            
            return True
            
        except Exception as e:
            print(f"Agent initialization failed: {e}")
            return False
    
    def test_indicator_calculations(self) -> bool:
        """Test technical indicator calculations"""
        
        try:
            # Get a test symbol
            test_symbols = self.db_manager.get_testing_symbols()
            
            if not test_symbols:
                print("No test symbols available, using synthetic data")
                symbol = "TEST_SYMBOL"
                df = self._create_synthetic_data(symbol, 100)
            else:
                symbol = test_symbols[0]
                print(f"Testing indicators for: {symbol}")
                
                # Get historical data
                end_date = datetime.now(self.ist)
                start_date = end_date - timedelta(days=60)
                
                df = self.db_manager.get_historical_data(symbol, start_date, end_date)
                
                if df.empty:
                    print("No historical data available, creating synthetic data")
                    df = self._create_synthetic_data(symbol, 100)
            
            print(f"Using {len(df)} data points")
            
            # Validate data
            if len(df) < 20:
                print("Insufficient data points for testing")
                return False
            
            # Test indicator calculations
            indicators = self.tech_agent._calculate_all_indicators(df)
            
            if not indicators:
                print("No indicators calculated")
                return False
            
            # Count valid indicators
            valid_indicators = {k: v for k, v in indicators.items() 
                              if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            
            print(f"Calculated {len(valid_indicators)} valid indicators out of {len(indicators)} total")
            
            # Check for key indicators
            key_indicators = ['close_price', 'date']
            missing_key = [ind for ind in key_indicators if ind not in valid_indicators]
            
            if missing_key:
                print(f"Missing critical indicators: {missing_key}")
                return False
            
            # Print sample indicators
            sample_indicators = ['rsi_14', 'ema_20', 'ma_trend', 'volume_signal']
            for key in sample_indicators:
                value = indicators.get(key)
                if value is not None:
                    print(f"  {key}: {value}")
            
            # Must have at least 5 valid indicators for a successful test
            return len(valid_indicators) >= 5
            
        except Exception as e:
            print(f"Indicator calculations test failed: {e}")
            return False
    
    def test_signal_generation(self) -> bool:
        """Test signal generation"""
        
        try:
            test_symbols = self.db_manager.get_testing_symbols()
            
            if not test_symbols:
                return False
            
            symbol = test_symbols[0]
            print(f"Testing signal generation for: {symbol}")
            
            # Analyze symbol
            analysis = self.tech_agent.analyze_symbol(symbol)
            
            if 'error' in analysis:
                print(f"Analysis error: {analysis['error']}")
                # Continue with synthetic data
                analysis = self._test_with_synthetic_data(symbol)
            
            if 'error' in analysis:
                return False
            
            # Validate analysis structure
            required_fields = ['technical_score', 'buy_signal', 'sell_signal', 'signal_strength']
            missing_fields = [field for field in required_fields if field not in analysis]
            
            if missing_fields:
                print(f"Missing analysis fields: {missing_fields}")
                return False
            
            # Display results
            print(f"Technical Score: {analysis['technical_score']:.3f}")
            print(f"Buy Signal: {analysis['buy_signal']}")
            print(f"Sell Signal: {analysis['sell_signal']}")
            print(f"Signal Strength: {analysis['signal_strength']}")
            
            if analysis.get('reasoning'):
                print(f"Reasoning: {analysis['reasoning'][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"Signal generation test failed: {e}")
            return False
    
    def test_multiple_symbols(self) -> bool:
        """Test analysis of multiple symbols"""
        
        try:
            test_symbols = self.db_manager.get_testing_symbols()[:3]  # Test 3 symbols
            
            if len(test_symbols) < 2:
                print("Need at least 2 test symbols")
                return False
            
            print(f"Testing multiple symbol analysis: {test_symbols}")
            
            # Analyze multiple symbols
            results = self.tech_agent.analyze_multiple_symbols(test_symbols)
            
            if not results:
                print("No results from multiple symbol analysis")
                return False
            
            # Check results
            successful_analyses = len([r for r in results if 'error' not in r])
            total_symbols = len(test_symbols)
            
            success_rate = successful_analyses / total_symbols
            print(f"Success rate: {success_rate:.1%} ({successful_analyses}/{total_symbols})")
            
            # Get summary
            summary = self.tech_agent.get_analysis_summary(test_symbols)
            print(f"Analysis summary: {summary}")
            
            return success_rate >= 0.5  # At least 50% success rate
            
        except Exception as e:
            print(f"Multiple symbols test failed: {e}")
            return False
    
    def test_data_storage(self) -> bool:
        """Test storing technical analysis results"""
        
        try:
            # Create sample indicators data
            indicators_data = [{
                'date': datetime.now(self.ist),
                'timeframe': '5m',
                'close': 100.50,
                'rsi_14': 65.5,
                'ema_20': 99.8,
                'ema_50': 98.5,
                'technical_score': 0.72
            }]
            
            # Test storage
            success = self.db_manager.store_technical_indicators('TEST_SYMBOL', indicators_data)
            
            if not success:
                print("Failed to store technical indicators")
                return False
            
            print("Technical indicators stored successfully")
            return True
            
        except Exception as e:
            print(f"Data storage test failed: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance with larger datasets"""
        
        try:
            start_time = datetime.now()
            
            # Test with synthetic large dataset
            symbol = "PERFORMANCE_TEST"
            df = self._create_synthetic_data(symbol, periods=500)  # 500 data points
            
            # Calculate indicators
            indicators = self.tech_agent._calculate_all_indicators(df)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"Performance test completed in {duration:.2f} seconds")
            print(f"Processed {len(df)} data points")
            
            if indicators:
                print(f"Calculated {len(indicators)} indicators")
                
            # Performance should be under 5 seconds for 500 data points
            return duration < 5.0 and bool(indicators)
            
        except Exception as e:
            print(f"Performance test failed: {e}")
            return False
    
    def _create_synthetic_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Create synthetic OHLCV data for testing"""
        
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='5min')  # Fixed: 'T' -> 'min'
        np.random.seed(42)
        
        base_price = 100
        prices = [base_price]
        
        for i in range(periods - 1):
            change = np.random.normal(0, 0.01)  # 1% volatility per 5-minute bar
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        
        df = pd.DataFrame({
            'symbol': [symbol] * periods,
            'date': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.005) for p in prices],
            'low': [p * np.random.uniform(0.995, 1.0) for p in prices],
            'close': prices,
            'volume': np.random.randint(10000, 100000, periods)
        })
        
        return df
    
    def _test_with_synthetic_data(self, symbol: str) -> dict:
        """Test analysis with synthetic data"""
        
        try:
            df = self._create_synthetic_data(symbol)
            indicators = self.tech_agent._calculate_all_indicators(df)
            
            if not indicators:
                return {'error': 'synthetic_data_failed'}
            
            analysis = self.tech_agent._generate_technical_signals(symbol, df, indicators)
            return analysis
            
        except Exception as e:
            return {'error': f'synthetic_test_failed: {e}'}
    
    def _print_summary(self):
        """Print test summary"""
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:30}: {status}")
        
        print("-" * 60)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("ALL TESTS PASSED - Technical Analysis System Ready!")
        else:
            print("Some tests failed - Please review before proceeding")
        
        print("=" * 60)

def main():
    """Run technical analysis test suite"""
    
    test_suite = TechnicalAnalysisTestSuite()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()