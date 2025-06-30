"""
FILE: quick_fix_validation.py
LOCATION: / (root directory)
PURPOSE: Quick Fix Validation Script - Validates all system corrections and ensures integrity

DESCRIPTION:
- Comprehensive validation script to test all system components
- Validates environment setup, imports, and configuration functions
- Tests database manager, technical agent, and technical calculations
- Ensures all bug fixes are working correctly
- Provides detailed validation reports and error identification

DEPENDENCIES:
- All system modules (database, agents, utils, config)
- Used after applying fixes to ensure system integrity

USAGE:
- Run directly: python quick_fix_validation.py
- Should be run after any system updates or fixes
- Provides pass/fail status for each component
- Identifies critical errors that need attention
"""

#!/usr/bin/env python3
"""
Quick Fix Validation Script
Validates all corrections and ensures system integrity
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class QuickFixValidator:
    """Validate all system fixes and corrections"""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_errors = []
        
    def run_validation(self):
        """Run complete validation suite"""
        
        print("=" * 60)
        print("QUICK FIX VALIDATION SUITE")
        print("=" * 60)
        
        validations = [
            ("Environment Setup", self.validate_environment),
            ("Import Dependencies", self.validate_imports),
            ("Config Functions", self.validate_config_functions),
            ("Database Manager", self.validate_database_manager),
            ("Technical Agent", self.validate_technical_agent),
            ("Technical Calculations", self.validate_technical_calculations),
            ("System Integration", self.validate_system_integration)
        ]
        
        for validation_name, validation_func in validations:
            print(f"\nValidating: {validation_name}")
            try:
                result = validation_func()
                self.validation_results[validation_name] = result
                status = "PASS" if result else "FAIL"
                print(f"Status: {status}")
            except Exception as e:
                self.validation_results[validation_name] = False
                self.critical_errors.append(f"{validation_name}: {str(e)}")
                print(f"Status: CRITICAL ERROR - {e}")
        
        self.print_summary()
        return all(self.validation_results.values())
    
    def validate_environment(self) -> bool:
        """Validate environment setup"""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                print(f"Python version {sys.version} may not be compatible")
                return False
            
            # Check essential directories
            required_dirs = ['agents', 'database', 'utils']
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    print(f"Missing directory: {dir_name}")
                    return False
            
            print("Environment setup is valid")
            return True
            
        except Exception as e:
            print(f"Environment validation failed: {e}")
            return False
    
    def validate_imports(self) -> bool:
        """Validate all imports work correctly"""
        try:
            # Test core imports
            import pandas as pd
            import numpy as np
            print(f"pandas: {pd.__version__}, numpy: {np.__version__}")
            
            # Test optional imports
            try:
                import pandas_ta as ta
                print(f"pandas_ta: {ta.version} (available)")
            except ImportError:
                print("pandas_ta: Not available (will use fallback)")
            
            # Test project imports
            from dotenv import load_dotenv
            load_dotenv()
            
            import config
            print("Config module imported successfully")
            
            from database.enhanced_database_manager import EnhancedDatabaseManager
            print("Database manager imported successfully")
            
            return True
            
        except Exception as e:
            print(f"Import validation failed: {e}")
            return False
    
    def validate_config_functions(self) -> bool:
        """Validate config helper functions"""
        try:
            import config
            
            # Test RSI thresholds function
            thresholds = config.get_rsi_thresholds('Low')
            if not isinstance(thresholds, tuple) or len(thresholds) != 2:
                print("RSI thresholds function failed")
                return False
            print(f"RSI thresholds (Low): {thresholds}")
            
            # Test stop loss multiplier function
            multiplier = config.get_stop_loss_multiplier('Medium')
            if not isinstance(multiplier, (int, float)):
                print("Stop loss multiplier function failed")
                return False
            print(f"Stop loss multiplier (Medium): {multiplier}")
            
            # Test confidence calculation function
            confidence = config.calculate_final_confidence(0.7, 0.6, 0.5, 'A')
            if not isinstance(confidence, float) or not (0 <= confidence <= 1):
                print("Confidence calculation function failed")
                return False
            print(f"Final confidence example: {confidence:.3f}")
            
            # Test score validation function
            validated_score = config.validate_technical_score(0.75)
            if validated_score != 0.75:
                print("Score validation function failed")
                return False
            
            # Test with invalid score
            validated_invalid = config.validate_technical_score(float('nan'))
            if validated_invalid != 0.5:
                print("Score validation with NaN failed")
                return False
            
            print("All config functions working correctly")
            return True
            
        except Exception as e:
            print(f"Config function validation failed: {e}")
            return False
    
    def validate_database_manager(self) -> bool:
        """Validate database manager functionality"""
        try:
            # Skip actual database connection if credentials not available
            db_host = os.getenv('DATABASE_HOST')
            if not db_host:
                print("Database credentials not available, skipping connection test")
                return True
            
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            # Test initialization
            db_manager = EnhancedDatabaseManager()
            print("Database manager initialized successfully")
            
            # Test connection if credentials available
            if all(os.getenv(var) for var in ['DATABASE_NAME', 'DATABASE_USER', 'DATABASE_PASSWORD']):
                connection_test = db_manager.test_connection()
                print(f"Database connection test: {'PASS' if connection_test else 'FAIL'}")
                return connection_test
            else:
                print("Database credentials incomplete, skipping connection test")
                return True
            
        except Exception as e:
            print(f"Database manager validation failed: {e}")
            return False
    
    def validate_technical_agent(self) -> bool:
        """Validate technical agent functionality"""
        try:
            from agents.technical_agent import TechnicalAgent
            
            # Test initialization without database
            tech_agent = TechnicalAgent(None)
            print("Technical agent initialized successfully")
            
            # Test synthetic data creation
            test_data = tech_agent._create_test_data("TEST_SYMBOL", 50)
            if test_data.empty or len(test_data) != 50:
                print("Synthetic data creation failed")
                return False
            print("Synthetic data creation working")
            
            # Test data validation
            validated_data = tech_agent._validate_ohlcv_data(test_data)
            if validated_data.empty:
                print("Data validation failed")
                return False
            print("Data validation working")
            
            # Test manual indicator calculation
            indicators = tech_agent._calculate_indicators_manual(test_data)
            if not indicators:
                print("Manual indicator calculation failed")
                return False
            print(f"Manual calculations working: {len(indicators)} indicators")
            
            return True
            
        except Exception as e:
            print(f"Technical agent validation failed: {e}")
            return False
    
    def validate_technical_calculations(self) -> bool:
        """Validate technical calculations utility"""
        try:
            from utils.technical_calculations import TechnicalCalculations
            import pandas as pd
            import numpy as np
            
            # Create test data
            test_df = pd.DataFrame({
                'open': [100, 101, 102, 101, 100],
                'high': [101, 102, 103, 102, 101],
                'low': [99, 100, 101, 100, 99],
                'close': [100.5, 101.5, 102.5, 101.5, 100.5],
                'volume': [1000, 1100, 1200, 1100, 1000]
            })
            
            # Test data validation
            is_valid = TechnicalCalculations.validate_ohlcv_data(test_df)
            if not is_valid:
                print("OHLCV validation failed")
                return False
            print("OHLCV validation working")
            
            # Test data cleaning
            cleaned_df = TechnicalCalculations.clean_ohlcv_data(test_df)
            if cleaned_df.empty:
                print("Data cleaning failed")
                return False
            print("Data cleaning working")
            
            # Test support/resistance calculation
            sr_levels = TechnicalCalculations.find_support_resistance_levels(test_df)
            if not isinstance(sr_levels, dict):
                print("Support/resistance calculation failed")
                return False
            print("Support/resistance calculation working")
            
            return True
            
        except Exception as e:
            print(f"Technical calculations validation failed: {e}")
            return False
    
    def validate_system_integration(self) -> bool:
        """Validate overall system integration"""
        try:
            # Test main module import
            import main
            print("Main module imported successfully")
            
            # Test agents package
            from agents import TechnicalAgent
            print("Agents package working")
            
            # Test end-to-end workflow (without database)
            from agents.technical_agent import TechnicalAgent
            
            tech_agent = TechnicalAgent(None)
            
            # Test complete analysis with synthetic data
            analysis = tech_agent.analyze_symbol("INTEGRATION_TEST")
            
            if 'error' in analysis:
                print(f"Integration test analysis failed: {analysis['error']}")
                return False
            
            # Verify analysis structure
            required_fields = ['technical_score', 'overall_confidence', 'buy_signal', 'analysis_type']
            missing_fields = [field for field in required_fields if field not in analysis]
            
            if missing_fields:
                print(f"Missing analysis fields: {missing_fields}")
                return False
            
            print(f"Integration test successful:")
            print(f"  Technical Score: {analysis['technical_score']:.3f}")
            print(f"  Overall Confidence: {analysis['overall_confidence']:.3f}")
            print(f"  Analysis Type: {analysis['analysis_type']}")
            
            return True
            
        except Exception as e:
            print(f"System integration validation failed: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary"""
        
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for result in self.validation_results.values() if result)
        
        for validation_name, result in self.validation_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{validation_name:25}: {status}")
        
        print("-" * 60)
        print(f"TOTAL: {passed_validations}/{total_validations} validations passed")
        
        if self.critical_errors:
            print(f"\nCRITICAL ERRORS:")
            for error in self.critical_errors:
                print(f"  - {error}")
        
        if passed_validations == total_validations:
            print("\n✅ ALL VALIDATIONS PASSED")
            print("System is ready for technical analysis testing!")
        else:
            print(f"\n❌ {total_validations - passed_validations} VALIDATIONS FAILED")
            print("Please fix issues before proceeding")
        
        print("=" * 60)

def main():
    """Run validation"""
    
    validator = QuickFixValidator()
    success = validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()