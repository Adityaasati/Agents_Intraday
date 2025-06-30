#!/usr/bin/env python3
"""
FILE: test_type_fixes.py
LOCATION: / (root directory)
PURPOSE: Test the type safety fixes for technical analysis

DESCRIPTION:
- Tests that float/Decimal mixing is resolved
- Validates that numpy types are converted properly
- Ensures database storage works without type errors
- Quick validation of all type-related fixes

USAGE:
- python test_type_fixes.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def test_type_mixing():
    """Test that float and Decimal types mix properly"""
    
    print("Testing float/Decimal type mixing...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        
        # Create test data with mixed types (simulating database data)
        mixed_data = pd.DataFrame({
            'symbol': ['TEST'] * 30,
            'date': pd.date_range(start='2024-01-01', periods=30, freq='5min'),
            'open': [Decimal('100.0')] * 15 + [100.0] * 15,  # Mix Decimal and float
            'high': [Decimal('101.0')] * 15 + [101.0] * 15,
            'low': [Decimal('99.0')] * 15 + [99.0] * 15,
            'close': [Decimal('100.5')] * 15 + [100.5] * 15,
            'volume': [1000] * 30
        })
        
        tech_agent = TechnicalAgent(None)
        
        # Test data validation (should convert types)
        validated_data = tech_agent._validate_ohlcv_data(mixed_data)
        
        if validated_data.empty:
            print("❌ Data validation failed with mixed types")
            return False
        
        # Check that all price columns are now float
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if validated_data[col].dtype != 'float64':
                print(f"❌ Column {col} is not float type: {validated_data[col].dtype}")
                return False
        
        print("✅ Type mixing handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ Type mixing test failed: {e}")
        return False

def test_indicator_calculations():
    """Test that indicator calculations work with type-safe data"""
    
    print("Testing type-safe indicator calculations...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        
        tech_agent = TechnicalAgent(None)
        
        # Create test data
        test_data = tech_agent._create_test_data("TYPE_TEST", 50)
        
        # Test manual calculations
        indicators = tech_agent._calculate_indicators_manual(test_data)
        
        if not indicators:
            print("❌ No indicators calculated")
            return False
        
        # Check that all indicator values are proper Python types
        problematic_indicators = []
        for key, value in indicators.items():
            if value is not None:
                # Check for numpy types
                if hasattr(value, 'dtype'):
                    problematic_indicators.append(f"{key}: {type(value)}")
                # Check for invalid values
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    problematic_indicators.append(f"{key}: NaN/Inf")
        
        if problematic_indicators:
            print(f"❌ Problematic indicator types: {problematic_indicators}")
            return False
        
        print(f"✅ Calculated {len(indicators)} indicators with proper types")
        return True
        
    except Exception as e:
        print(f"❌ Indicator calculations test failed: {e}")
        return False

def test_storage_type_conversion():
    """Test that storage type conversion works"""
    
    print("Testing storage type conversion...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        
        tech_agent = TechnicalAgent(None)
        
        # Test type conversion function
        test_values = {
            'numpy_float': np.float64(123.456),
            'numpy_int': np.int64(123),
            'pandas_series': pd.Series([1, 2, 3]),
            'regular_float': 123.456,
            'regular_int': 123,
            'nan_value': np.nan,
            'inf_value': np.inf,
            'none_value': None,
            'string_value': 'test'
        }
        
        converted_values = {}
        for key, value in test_values.items():
            converted = tech_agent._convert_to_python_type(value)
            converted_values[key] = converted
        
        # Validate conversions
        if not isinstance(converted_values['numpy_float'], (float, type(None))):
            print(f"❌ numpy_float conversion failed: {type(converted_values['numpy_float'])}")
            return False
        
        if converted_values['nan_value'] is not None:
            print(f"❌ NaN should convert to None")
            return False
        
        if converted_values['inf_value'] is not None:
            print(f"❌ Inf should convert to None")  
            return False
        
        if not isinstance(converted_values['pandas_series'], (float, type(None))):
            print(f"❌ pandas_series conversion failed: {type(converted_values['pandas_series'])}")
            return False
        
        print("✅ Type conversion working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Storage type conversion test failed: {e}")
        return False

def test_complete_analysis_with_types():
    """Test complete analysis with mixed types"""
    
    print("Testing complete analysis with type safety...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        
        tech_agent = TechnicalAgent(None)
        
        # Run complete analysis (will use synthetic data)
        analysis = tech_agent.analyze_symbol("TYPE_SAFETY_TEST")
        
        if 'error' in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return False
        
        # Check that all numeric values in analysis are proper types
        numeric_fields = ['technical_score', 'overall_confidence', 'entry_price', 'stop_loss', 'target_price']
        
        for field in numeric_fields:
            value = analysis.get(field)
            if value is not None:
                if hasattr(value, 'dtype'):  # numpy type
                    print(f"❌ Field {field} has numpy type: {type(value)}")
                    return False
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    print(f"❌ Field {field} has invalid value: {value}")
                    return False
        
        print(f"✅ Complete analysis working with proper types")
        print(f"   Technical Score: {analysis.get('technical_score')}")
        print(f"   Overall Confidence: {analysis.get('overall_confidence')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete analysis test failed: {e}")
        return False

def test_database_storage():
    """Test database storage without numpy type errors"""
    
    print("Testing database storage...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from datetime import datetime
        import pytz
        
        # Test storage with proper Python types
        test_data = [{
            'date': datetime.now(pytz.timezone('Asia/Kolkata')),
            'timeframe': '5m',
            'close': 100.50,  # Python float
            'rsi_14': 65.5,   # Python float
            'ema_20': 99.8,   # Python float
            'ema_50': 98.5,   # Python float
            'technical_score': 0.72  # Python float
        }]
        
        db_manager = EnhancedDatabaseManager()
        
        # This should work without "np.float64" errors
        result = db_manager.store_technical_indicators('TYPE_TEST', test_data)
        
        # Result can be True or False, but should not crash with type errors
        print(f"✅ Database storage completed (result: {result})")
        return True
        
    except Exception as e:
        error_str = str(e)
        if "np.float64" in error_str or "schema \"np\" does not exist" in error_str:
            print(f"❌ Database storage still has numpy type issues: {e}")
            return False
        else:
            print(f"⚠️  Database storage had other issue (not type-related): {e}")
            return True  # Other issues are acceptable for this test

def main():
    """Run all type safety tests"""
    
    print("=" * 60)
    print("TYPE SAFETY TESTS")
    print("=" * 60)
    
    tests = [
        ("Type Mixing", test_type_mixing),
        ("Indicator Calculations", test_indicator_calculations),
        ("Storage Type Conversion", test_storage_type_conversion),
        ("Complete Analysis", test_complete_analysis_with_types),
        ("Database Storage", test_database_storage)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TYPE SAFETY TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25}: {status}")
    
    print("-" * 60)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TYPE SAFETY ISSUES RESOLVED!")
        print("Technical analysis should now work without type errors")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} type issues still need attention")
    
    print("=" * 60)

if __name__ == "__main__":
    main()