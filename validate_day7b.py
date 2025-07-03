#!/usr/bin/env python3
"""
FILE: validate_day7b.py
PURPOSE: Validate Day 7B Production Dashboard & Deployment Implementation

USAGE:
- python validate_day7b.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import config

class Day7BValidator:
    """Validate Day 7B dashboard and production features"""
    
    def __init__(self):
        self.results = {}
    
    def run_validation(self) -> bool:
        """Run complete Day 7B validation"""
        
        print("=" * 60)
        print("DAY 7B PRODUCTION DASHBOARD VALIDATION")
        print("=" * 60)
        
        validations = [
            ("Dashboard Configuration", self.validate_dashboard_config),
            ("Dashboard Monitor", self.validate_dashboard_monitor),
            ("System Reporter", self.validate_system_reporter),
            ("Dashboard Mode", self.validate_dashboard_mode),
            ("Report Generation", self.validate_report_generation),
            ("Production Settings", self.validate_production_settings)
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
    
    def validate_dashboard_config(self) -> bool:
        """Validate dashboard configuration"""
        try:
            # Check dashboard settings exist
            required_settings = [
                'ENABLE_DASHBOARD', 'DASHBOARD_HOST', 'DASHBOARD_PORT',
                'DASHBOARD_AUTO_REFRESH', 'ENABLE_AUTO_REPORTS'
            ]
            
            missing = []
            for setting in required_settings:
                if not hasattr(config, setting):
                    missing.append(setting)
            
            if missing:
                print(f"  Missing settings: {missing}")
                return False
            
            # Test configuration functions
            dashboard_config = config.get_dashboard_config()
            monitoring_config = config.get_monitoring_config()
            
            print(f"  Dashboard port: {dashboard_config['port']}")
            print(f"  Auto refresh: {dashboard_config['refresh']}s")
            print(f"  Monitoring enabled: {monitoring_config['alerts_enabled']}")
            
            return True
            
        except Exception as e:
            print(f"  Config error: {e}")
            return False
    
    def validate_dashboard_monitor(self) -> bool:
        """Validate dashboard monitor functionality"""
        try:
            from utils.dashboard_monitor import get_dashboard_monitor
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            # Initialize
            db_manager = EnhancedDatabaseManager()
            dashboard = get_dashboard_monitor(db_manager)
            
            # Test system status
            system_status = dashboard.get_system_status()
            if 'error' in system_status:
                print(f"  System status error: {system_status['error']}")
                return False
            
            # Test trading metrics
            trading_metrics = dashboard.get_trading_metrics()
            
            # Test alerts
            alerts = dashboard.get_system_alerts()
            
            print(f"  Health score: {system_status.get('health_score', 0)}/100")
            print(f"  System status: {system_status.get('status', 'unknown')}")
            print(f"  Active alerts: {len(alerts)}")
            
            return True
            
        except Exception as e:
            print(f"  Dashboard monitor error: {e}")
            return False
    
    def validate_system_reporter(self) -> bool:
        """Validate system reporting functionality"""
        try:
            from reports.system_dashboard import SystemReporter
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            reporter = SystemReporter(db_manager)
            
            # Test daily report
            daily_report = reporter.generate_daily_report()
            if 'error' in daily_report:
                print(f"  Daily report error: {daily_report['error']}")
                return False
            
            # Test health report
            health_report = reporter.generate_health_report()
            if 'error' in health_report:
                print(f"  Health report error: {health_report['error']}")
                return False
            
            # Test performance report
            performance_report = reporter.generate_performance_report()
            if 'error' in performance_report:
                print(f"  Performance report error: {performance_report['error']}")
                return False
            
            print(f"  Reports generated: daily, health, performance")
            print(f"  Daily summary: {daily_report.get('summary', {}).get('overall_status', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"  System reporter error: {e}")
            return False
    
    def validate_dashboard_mode(self) -> bool:
        """Validate dashboard mode in main system"""
        try:
            from main import NexusTradingSystem
            
            system = NexusTradingSystem()
            
            # Check dashboard mode method exists
            if not hasattr(system, '_dashboard_mode'):
                print("  Missing _dashboard_mode method")
                return False
            
            # Check report mode method exists
            if not hasattr(system, '_report_mode'):
                print("  Missing _report_mode method")
                return False
            
            # Check system health method exists
            if not hasattr(system, 'get_system_health'):
                print("  Missing get_system_health method")
                return False
            
            print(f"  Dashboard mode available")
            print(f"  Report mode available")
            print(f"  System health method available")
            
            return True
            
        except Exception as e:
            print(f"  Dashboard mode error: {e}")
            return False
    
    def validate_report_generation(self) -> bool:
        """Validate automated report generation"""
        try:
            from reports.system_dashboard import generate_all_reports
            from database.enhanced_database_manager import EnhancedDatabaseManager
            
            db_manager = EnhancedDatabaseManager()
            
            # Generate all reports
            reports = generate_all_reports(db_manager)
            
            # Check report types
            required_reports = ['daily', 'health', 'performance']
            missing_reports = []
            
            for report_type in required_reports:
                if report_type not in reports:
                    missing_reports.append(report_type)
                elif 'error' in reports[report_type]:
                    missing_reports.append(f"{report_type} (error)")
            
            if missing_reports:
                print(f"  Missing/failed reports: {missing_reports}")
                return False
            
            print(f"  All report types generated successfully")
            
            return True
            
        except Exception as e:
            print(f"  Report generation error: {e}")
            return False
    
    def validate_production_settings(self) -> bool:
        """Validate production deployment settings"""
        try:
            # Check production settings
            production_settings = [
                'PRODUCTION_MODE', 'MAX_CONCURRENT_USERS', 'SESSION_TIMEOUT_MINUTES',
                'AUTO_CLEANUP_ENABLED', 'LOG_RETENTION_DAYS', 'DATA_RETENTION_DAYS'
            ]
            
            for setting in production_settings:
                if not hasattr(config, setting):
                    print(f"  Missing production setting: {setting}")
                    return False
            
            # Check database backup functionality
            from database.enhanced_database_manager import EnhancedDatabaseManager
            db_manager = EnhancedDatabaseManager()
            
            # Test backup (if method exists)
            if hasattr(db_manager, 'backup_critical_data'):
                backup_result = db_manager.backup_critical_data()
                if backup_result.get('status') != 'completed':
                    print(f"  Backup test failed: {backup_result.get('error', 'unknown')}")
                    return False
                print(f"  Backup functionality working")
            
            # Test cleanup (if method exists)
            if hasattr(db_manager, 'get_health_check_data'):
                health_data = db_manager.get_health_check_data()
                if 'error' in health_data:
                    print(f"  Health check failed: {health_data['error']}")
                    return False
                print(f"  Health check functionality working")
            
            print(f"  Production mode: {config.PRODUCTION_MODE}")
            print(f"  Auto cleanup: {config.AUTO_CLEANUP_ENABLED}")
            
            return True
            
        except Exception as e:
            print(f"  Production settings error: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary"""
        
        print("\n" + "=" * 60)
        print("DAY 7B VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        for test_name, result in self.results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:30}: {status}")
        
        print("-" * 60)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 DAY 7B PRODUCTION DASHBOARD COMPLETE!")
            print("✓ Real-time monitoring dashboard operational")
            print("✓ Automated reporting system functional")
            print("✓ Production deployment features ready")
            print("✓ System administration tools available")
            print("\n🚀 PRODUCTION READY!")
            print("\nNew commands available:")
            print("- python main.py --mode dashboard")
            print("- python main.py --mode report")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} validations failed")
            print("Please fix issues before production deployment")
        
        print("=" * 60)

def main():
    """Run Day 7B validation"""
    
    validator = Day7BValidator()
    success = validator.run_validation()
    
    if success:
        print("\n✅ Day 7B Production Dashboard Validated!")
    else:
        print("\n❌ Day 7B Validation Failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()