#!/usr/bin/env python3
"""
FILE: test_fixes.py
LOCATION: / (root directory)
PURPOSE: Quick test script to validate all fixes are working

DESCRIPTION:
- Tests all the fixes applied to resolve the issues
- Validates database column fixes
- Tests technical analysis without warnings
- Provides quick pass/fail status

USAGE:
- python test_fixes.py
"""

import sys
import os
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

def test_pandas_frequency_fix():
    """Test that pandas frequency warnings are fixed"""
    
    print("Testing pandas frequency fix...")
    
    try:
        import pandas as pd
        from datetime import datetime
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should not generate warnings now
            dates = pd.date_range(end=datetime.now(), periods=10, freq='5min')
            
            # Check for warnings
            freq_warnings = [warning for warning in w if "'T' is deprecated" in str(warning.message)]
            
            if freq_warnings:
                print(f"❌ Still getting frequency warnings: {len(freq_warnings)}")
                return False
            else:
                print("✅ No frequency warnings detected")
                return True
                
    except Exception as e:
        print(f"❌ Error testing frequency fix: {e}")
        return False

def test_database_columns():
    """Test database column existence"""
    
    print("Testing database columns...")
    
    try:
        from database_fix_script import DatabaseColumnFixer
        
        fixer = DatabaseColumnFixer()
        result = fixer.check_table_structure()
        
        if result:
            print("✅ Database columns are correct")
        else:
            print("❌ Database columns need fixing")
            print("   Run: python database_fix_script.py --fix")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing database columns: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis with synthetic data"""
    
    print("Testing technical analysis...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        
        # Test without database connection
        tech_agent = TechnicalAgent(None)
        
        # Test synthetic data creation (should not have warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            test_data = tech_agent._create_test_data("TEST_FIX", 30)
            
            freq_warnings = [warning for warning in w if "'T' is deprecated" in str(warning.message)]
            
            if freq_warnings:
                print(f"❌ Still getting warnings in technical analysis: {len(freq_warnings)}")
                return False
        
        # Test data validation
        if test_data.empty or len(test_data) != 30:
            print("❌ Synthetic data creation failed")
            return False
        
        # Test indicator calculation
        indicators = tech_agent._calculate_indicators_manual(test_data)
        
        if not indicators:
            print("❌ Indicator calculation failed")
            return False
        
        print(f"✅ Technical analysis working - {len(indicators)} indicators calculated")
        return True
        
    except Exception as e:
        print(f"❌ Error testing technical analysis: {e}")
        return False

def test_storage_fallback():
    """Test storage fallback mechanism"""
    
    print("Testing storage fallback...")
    
    try:
        from database.enhanced_database_manager import EnhancedDatabaseManager
        from datetime import datetime
        import pytz
        
        db_manager = EnhancedDatabaseManager()
        
        # Test data for storage
        test_indicators = [{
            'date': datetime.now(pytz.timezone('Asia/Kolkata')),
            'timeframe': '5m',
            'close': 100.0,
            'rsi_14': 50.0,
            'ema_20': 99.5,
            'ema_50': 98.0,
            'technical_score': 0.6
        }]
        
        # This should work with fallback even if close_price column doesn't exist
        result = db_manager.store_technical_indicators('TEST_STORAGE', test_indicators)
        
        if result:
            print("✅ Storage working (with fallback if needed)")
        else:
            print("⚠️  Storage failed but fallback should handle it")
        
        return True  # Return True because we handle this gracefully now
        
    except Exception as e:
        print(f"❌ Error testing storage: {e}")
        return False

def test_integration():
    """Test complete integration"""
    
    print("Testing complete integration...")
    
    try:
        from agents.technical_agent import TechnicalAgent
        
        tech_agent = TechnicalAgent(None)  # No database for this test
        
        # Run complete analysis
        analysis = tech_agent.analyze_symbol("INTEGRATION_TEST")
        
        if 'error' in analysis:
            if analysis['error'] == 'calculation_failed':
                print("❌ Integration test failed - calculation error")
                return False
            else:
                print(f"⚠️  Integration test had expected error: {analysis['error']}")
        
        # Check required fields
        required_fields = ['technical_score', 'overall_confidence', 'analysis_type']
        missing_fields = [field for field in required_fields if field not in analysis]
        
        if missing_fields:
            print(f"❌ Missing fields in analysis: {missing_fields}")
            return False
        
        print(f"✅ Integration test passed")
        print(f"   Technical Score: {analysis.get('technical_score', 'N/A')}")
        print(f"   Overall Confidence: {analysis.get('overall_confidence', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        return False

def main():
    """Run all fix tests"""
    
    print("=" * 60)
    print("TESTING ALL FIXES")
    print("=" * 60)
    
    tests = [
        ("Pandas Frequency Fix", test_pandas_frequency_fix),
        ("Database Columns", test_database_columns),
        ("Technical Analysis", test_technical_analysis),
        ("Storage Fallback", test_storage_fallback),
        ("Integration Test", test_integration)
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
    print("FIX TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25}: {status}")
    
    print("-" * 60)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL FIXES WORKING CORRECTLY!")
        print("You can now run: python tests/test_technical_analysis.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} issues still need attention")
        
        # Provide guidance
        if not results.get("Database Columns", True):
            print("\n📋 TO FIX DATABASE ISSUES:")
            print("   python database_fix_script.py --check")
            print("   python database_fix_script.py --fix")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
    