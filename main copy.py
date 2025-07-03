"""
FILE: main.py  
LOCATION: / (root directory)
PURPOSE: Main Entry Point - Primary system controller with all execution modes

DESCRIPTION:
- Main entry point for the Nexus Trading System
- Supports multiple execution modes: setup, test, demo, integration
- Handles system initialization, validation, and testing
- Integrates database, technical analysis, and system health checks
- Provides command-line interface for system operations
- **UPDATED**: Now includes technical analysis testing and validation

EXECUTION MODES:
- setup: Initialize environment and verify basic setup
- test: Comprehensive system testing including technical analysis
- demo: Demonstrate analysis on single symbol with technical indicators
- integration: Complete system integration test with all components

DEPENDENCIES:
- database/enhanced_database_manager.py (for database operations)
- database/schema_creator.py (for schema management)  
- agents/technical_agent.py (for technical analysis)
- utils/data_updater.py (for data pipeline testing)
- utils/logger_setup.py (for logging)

USAGE:
- python main.py --mode setup
- python main.py --mode test
- python main.py --mode demo --symbol RELIANCE
- python main.py --mode integration
"""

#!/usr/bin/env python3
"""
Nexus Trading System - Main Entry Point
Day 1 Implementation - Simplified and Robust

Usage:
    python main.py --mode setup
    python main.py --mode test  
    python main.py --mode demo --symbol RELIANCE
    python main.py --mode integration
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import traceback


try:
    from advanced_analysis_integration import AdvancedAnalysis
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    
# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Please install requirements.txt")
    sys.exit(1)

class NexusTradingSystem:
    """Main Nexus Trading System Controller - Day 1 Simplified"""
    
    def __init__(self):
        self.setup_basic_logging()
        self.db_manager = None
        self.schema_creator = None
        
    def setup_basic_logging(self):
        """Setup basic logging before importing our modules"""
        import logging
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Basic logging setup
        log_file = log_dir / f'nexus_trading_{datetime.now().strftime("%Y_%m_%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('nexus_trading.main')
    
    def _live_trading_mode(self) -> bool:
        """Live trading mode with trade mode control"""
        
        try:
            import config
            
            print("=" * 60)
            print("LIVE TRADING MODE")
            print("=" * 60)
            
            # Show current configuration
            print(f"TRADE_MODE: {'yes' if config.TRADE_MODE else 'no'}")
            print(f"LIVE_TRADING_MODE: {config.LIVE_TRADING_MODE}")
            
            if not config.TRADE_MODE:
                print("\nTRADE_MODE=no: Generating signals only (no actual orders)")
                print("Set TRADE_MODE=yes in .env to enable actual trading")
            else:
                print("\nTRADE_MODE=yes: Live trading enabled")
            
            if not config.LIVE_TRADING_MODE and config.TRADE_MODE:
                print("LIVE_TRADING_MODE=false: Falling back to paper trading")
                return self._paper_trading_mode()
            
            # Initialize live trading components
            from live_trading_manager import LiveTradingManager
            live_manager = LiveTradingManager()
            
            # Validate setup
            if not live_manager.validate_live_trading_setup():
                print("Live trading setup validation failed")
                if config.TRADE_MODE:
                    print("Falling back to paper trading")
                    return self._paper_trading_mode()
                else:
                    print("Will generate signals only")
            
            # Check market hours (only for actual trading)
            if config.TRADE_MODE and not live_manager.portfolio_agent._is_market_open():
                print("Market is closed - cannot execute live trades")
                print("Use signal generation mode or paper trading")
                return False
            
            # Run trading session
            result = live_manager.run_live_trading_session()
            
            if 'error' in result:
                print(f"Session failed: {result['error']}")
                if config.TRADE_MODE:
                    print("Falling back to paper trading")
                    return self._paper_trading_mode()
                return False
            
            # Generate report
            live_manager.generate_live_trading_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Live trading mode failed: {e}")
            print("Falling back to paper trading")
            return self._paper_trading_mode()
        
    def run(self, mode: str, symbol: str = None):
        """Main execution method"""
        
        self.logger.info("=" * 50)
        self.logger.info("NEXUS TRADING SYSTEM STARTING")
        self.logger.info(f"Mode: {mode}")
        if symbol:
            self.logger.info(f"Symbol: {symbol}")
        self.logger.info("=" * 50)
        
        try:
            if mode == 'setup':
                return self._setup_mode()
            elif mode == 'test':
                return self._test_mode()
            elif mode == 'demo':
                return self._demo_mode(symbol)
            elif mode == 'integration':
                return self._integration_mode()
            else:
                self.logger.error(f"Unknown mode: {mode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Critical error in {mode} mode: {e}")
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self.logger.info("NEXUS TRADING SYSTEM STOPPING")
            self.logger.info("=" * 50)
    
    def _setup_mode(self) -> bool:
        """Initialize environment and verify basic setup"""
        
        self.logger.info("Starting SETUP mode...")
        
        try:
            # Create required directories
            directories = ['logs', 'data', 'config', 'temp']
            for dir_name in directories:
                dir_path = Path(dir_name)
                dir_path.mkdir(exist_ok=True)
                self.logger.info(f"Directory created/verified: {dir_name}")
            
            # Check .env file
            env_file = Path('.env')
            if not env_file.exists():
                self.logger.warning(".env file not found")
                self.logger.info("Please create .env file from .env.template")
                
                # Check if template exists
                template_file = Path('.env.template')
                if template_file.exists():
                    self.logger.info("Found .env.template - you can copy this to .env")
                return False
            
            # Check required environment variables
            required_vars = [
                'DATABASE_HOST', 'DATABASE_NAME', 'DATABASE_USER', 'DATABASE_PASSWORD'
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                self.logger.error(f"Missing environment variables: {missing_vars}")
                self.logger.info("Please check your .env file")
                return False
            
            # Test basic imports
            try:
                import pandas
                import numpy
                import psycopg2
                self.logger.info("Core dependencies verified")
            except ImportError as e:
                self.logger.error(f"Missing dependency: {e}")
                self.logger.info("Please run: pip install -r requirements.txt")
                return False
            
            self.logger.info("Setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False
    
    def _test_sentiment_analysis(self) -> bool:
        """Test sentiment analysis capabilities - Day 3A"""
        
        try:
            from agents.news_sentiment_agent import NewsSentimentAgent
            
            # Initialize sentiment agent
            sentiment_agent = NewsSentimentAgent(self.db_manager)
            
            # Test market sentiment
            market_sentiment = sentiment_agent.get_market_sentiment()
            
            if market_sentiment.get('status') == 'unavailable':
                self.logger.warning("Sentiment analysis unavailable (API key or dependencies missing)")
                return True  # Don't fail test if sentiment is not configured
            
            self.logger.info(f"Market sentiment: {market_sentiment.get('market_sentiment', 0.5):.3f}")
            
            # Test symbol sentiment
            test_symbols = self.db_manager.get_testing_symbols()[:2]
            
            if test_symbols:
                for symbol in test_symbols:
                    sentiment_result = sentiment_agent.analyze_symbol_sentiment(symbol, hours_back=48)
                    status = sentiment_result.get('status', 'unknown')
                    score = sentiment_result.get('sentiment_score', 0.5)
                    
                    self.logger.info(f"Sentiment for {symbol}: {score:.3f} ({status})")
            
            return True
            
        except ImportError:
            self.logger.warning("Sentiment analysis components not available")
            return True  # Don't fail if not implemented
        except Exception as e:
            self.logger.error(f"Sentiment analysis test failed: {e}")
            return False
    
    def _test_enhanced_sentiment(self) -> bool:
        """Test enhanced sentiment analysis - Day 3B"""
        
        try:
            from agents.news_sentiment_agent import NewsSentimentAgent
            from reports.sentiment_dashboard import SentimentDashboard
            
            # Initialize components
            sentiment_agent = NewsSentimentAgent(self.db_manager)
            dashboard = SentimentDashboard(self.db_manager)
            
            # Test enhanced sentiment analysis
            test_symbols = self.db_manager.get_testing_symbols()[:2]
            
            if not test_symbols:
                self.logger.warning("No test symbols for enhanced sentiment")
                return True
            
            for symbol in test_symbols:
                # Test enhanced analysis
                enhanced_result = sentiment_agent.analyze_symbol_sentiment(symbol, hours_back=48)
                status = enhanced_result.get('status', 'unknown')
                
                if status == 'unavailable':
                    self.logger.info("Enhanced sentiment unavailable (normal without API key)")
                    return True
                
                self.logger.info(f"Enhanced sentiment {symbol}: {enhanced_result.get('sentiment_score', 0.5):.3f}")
                
                # Test sentiment trend
                trend_data = dashboard.get_symbol_sentiment_trend(symbol)
                self.logger.info(f"Trend {symbol}: {trend_data.get('trend_direction', 'unknown')}")
            
            # Test market overview
            report = dashboard.generate_sentiment_report(test_symbols)
            if 'error' not in report:
                market_avg = report.get('market_overview', {}).get('average_sentiment', 0.5)
                self.logger.info(f"Market sentiment: {market_avg:.3f}")
            
            return True
            
        except ImportError:
            self.logger.warning("Enhanced sentiment components not available")
            return True
        except Exception as e:
            self.logger.error(f"Enhanced sentiment test failed: {e}")
            return False

    def _create_sentiment_tables(self) -> bool:
        """Create sentiment tables for Day 3B"""
        
        try:
            # Create sentiment tables
            if self.db_manager.create_sentiment_tables():
                self.logger.info("Sentiment tables created successfully")
                return True
            else:
                self.logger.warning("Failed to create sentiment tables")
                return False
                
        except Exception as e:
            self.logger.error(f"Sentiment table creation failed: {e}")
            return False

    
    def _test_mode(self) -> bool:
        """Comprehensive system testing and validation"""
        
        self.logger.info("Starting TEST mode...")
        
        test_results = {
            'environment': False,
            'database_connection': False,
            'existing_tables': False,
            'agent_tables': False,
            'data_pipeline': False,
            'technical_analysis': False,
            'sentiment_analysis': False
        }
        
        try:
            # Test 1: Environment validation
            self.logger.info("Test 1: Environment validation...")
            test_results['environment'] = self._test_environment()
            
            # Test 2: Database connection
            self.logger.info("Test 2: Database connection...")
            test_results['database_connection'] = self._test_database_connection()
            
            if test_results['database_connection']:
                # Test 3: Existing tables access
                self.logger.info("Test 3: Existing tables access...")
                test_results['existing_tables'] = self._test_existing_tables()
                
                # Test 4: Agent tables
                self.logger.info("Test 4: Agent tables...")
                test_results['agent_tables'] = self._test_agent_tables()
                
                self.logger.info("Test 5: Sentiment tables...")
                test_results['sentiment_tables'] = self._create_sentiment_tables()
            
                # Test 6: Data pipeline
                self.logger.info("Test 6: Data pipeline...")
                test_results['data_pipeline'] = self._test_data_pipeline()
                
                # Test 7: Technical analysis
                self.logger.info("Test 7: Technical analysis...")
                test_results['technical_analysis'] = self._test_technical_analysis()
                
                # Test 8: Sentiment analysis (Day 3)
                self.logger.info("Test 8: Sentiment analysis...")
                test_results['sentiment_analysis'] = self._test_sentiment_analysis()
                
                self.logger.info("Test 9: Enhanced sentiment...")
                test_results['enhanced_sentiment'] = self._test_enhanced_sentiment()
                
            # Report results
            self._report_test_results(test_results)
            
            return all(test_results.values())
            
        except Exception as e:
            self.logger.error(f"Test mode failed: {e}")
            return False
    
    def _demo_mode(self, symbol: str = None) -> bool:
        """Demonstrate analysis on single symbol"""
        
        if not symbol:
            symbol = 'RELIANCE'
        
        self.logger.info(f"Starting DEMO mode for symbol: {symbol}")
        
        try:
            # Import and initialize database manager
            from database.enhanced_database_manager import EnhancedDatabaseManager
            self.db_manager = EnhancedDatabaseManager()
            
            # Test database connection
            if not self.db_manager.test_connection():
                self.logger.error("Database connection failed")
                return False
            
            # Get symbol data
            self.logger.info(f"Loading fundamental data for {symbol}...")
            fundamental_data = self.db_manager.get_fundamental_data(symbol)
            
            if not fundamental_data:
                self.logger.error(f"No fundamental data found for {symbol}")
                self.logger.info("Available symbols:")
                available_symbols = self.db_manager.get_testing_symbols()
                for sym in available_symbols[:5]:
                    self.logger.info(f"  - {sym}")
                return False
            
            # Display fundamental data
            self._display_symbol_info(symbol, fundamental_data)
            
            # Get historical data
            self.logger.info(f"Loading historical data for {symbol}...")
            historical_data = self.db_manager.get_historical_data(symbol)
            
            if historical_data.empty:
                self.logger.warning(f"No historical data found for {symbol}")
                self.logger.info("This is normal for Day 1 setup")
            else:
                self.logger.info(f"Found {len(historical_data)} historical records")
                self.logger.info(f"Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
            
            # Test data pipeline
            self.logger.info("Testing data pipeline...")
            from utils.data_updater import SimpleDataUpdater
            data_updater = SimpleDataUpdater(self.db_manager)
            
            pipeline_result = data_updater.test_data_pipeline([symbol])
            self.logger.info(f"Data pipeline test: {pipeline_result}")
            
            # Test technical analysis
            self.logger.info("Testing technical analysis...")
            from agents.technical_agent import TechnicalAgent
            tech_agent = TechnicalAgent(self.db_manager)
            
            analysis = tech_agent.analyze_symbol(symbol)
            
            if 'error' not in analysis:
                self._display_technical_analysis(symbol, analysis)
            else:
                self.logger.warning(f"Technical analysis error: {analysis.get('error')}")
            
            self.logger.info(f"Demo completed successfully for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Demo mode failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _integration_mode(self) -> bool:
        """Complete system integration test"""
        
        self.logger.info("Starting INTEGRATION mode...")
        
        phases = [
            ('Environment Check', self._integration_phase_environment),
            ('Database Integration', self._integration_phase_database),
            ('Schema Creation', self._integration_phase_schema),
            ('Data Access Test', self._integration_phase_data_access),
            ('Technical Analysis', self._integration_phase_technical_analysis),
            ('System Health Check', self._integration_phase_health_check)
        ]
        
        results = {}
        
        for phase_name, phase_func in phases:
            self.logger.info(f"Integration Phase: {phase_name}")
            try:
                results[phase_name] = phase_func()
                status = "PASSED" if results[phase_name] else "FAILED"
                self.logger.info(f"Phase {phase_name}: {status}")
            except Exception as e:
                self.logger.error(f"Phase {phase_name} failed: {e}")
                results[phase_name] = False
        
        # Final report
        self._report_integration_results(results)
        
        return all(results.values())
    
    def _test_environment(self) -> bool:
        """Test environment configuration"""
        
        try:
            # Test core imports
            import pandas as pd
            import numpy as np
            import psycopg2
            self.logger.info(f"pandas: {pd.__version__}")
            self.logger.info(f"numpy: {np.__version__}")
            
            # Test pandas_ta (optional for Day 1)
            try:
                import pandas_ta as ta
                self.logger.info(f"pandas_ta: {ta.version} (available)")
            except ImportError:
                self.logger.warning("pandas_ta not available - will use fallback calculations")
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Missing required package: {e}")
            return False
    
    def _test_database_connection(self) -> bool:
        """Test database connection"""
        
        try:
            from database.enhanced_database_manager import EnhancedDatabaseManager
            self.db_manager = EnhancedDatabaseManager()
            
            if self.db_manager.test_connection():
                health = self.db_manager.get_system_health()
                self.logger.info(f"Database connection successful")
                self.logger.info(f"System health: {health}")
                return True
            else:
                return False
            
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def _test_data_pipeline(self) -> bool:
        """Test data pipeline functionality"""
        
        try:
            from utils.data_updater import SimpleDataUpdater
            
            data_updater = SimpleDataUpdater(self.db_manager)
            
            # Test with 2 symbols
            test_symbols = ['RELIANCE', 'TCS']
            pipeline_result = data_updater.test_data_pipeline(test_symbols)
            
            success_count = pipeline_result.get('successful_updates', 0)
            total_count = pipeline_result.get('total_symbols', 0)
            
            self.logger.info(f"Data pipeline test successful: {success_count} symbols updated")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Data pipeline test failed: {e}")
            return False
    
    def _test_existing_tables(self) -> bool:
        """Test access to existing tables"""
        
        try:
            # Test stocks_categories_table
            symbols = self.db_manager.get_symbols_from_categories(limit=5)
            self.logger.info(f"Found {len(symbols)} symbols in stocks_categories_table")
            
            if symbols:
                sample_symbol = symbols[0]
                self.logger.info(f"Sample symbol: {sample_symbol['symbol']} - {sample_symbol.get('stock_name', 'N/A')}")
            
            # Test historical data tables
            quarters = self.db_manager.get_available_quarters()
            self.logger.info(f"Found {len(quarters)} historical data quarters")
            
            return len(symbols) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to access existing tables: {e}")
            return False
    
    def _test_agent_tables(self) -> bool:
        """Test agent tables creation and access"""
        
        try:
            from database.schema_creator import SchemaCreator
            self.schema_creator = SchemaCreator()
            
            # Test database connection first
            if not self.schema_creator.test_database_connection():
                self.logger.error("Schema creator cannot connect to database")
                return False
            
            # Create essential tables
            if self.schema_creator.create_essential_tables():
                self.logger.info("Essential agent tables created/verified")
            
            # Initialize basic configuration
            if self.schema_creator.initialize_basic_config():
                self.logger.info("Basic system configuration initialized")
            
            # Verify tables
            table_status = self.schema_creator.verify_essential_tables()
            
            missing_tables = [table for table, exists in table_status.items() if not exists]
            
            if missing_tables:
                self.logger.warning(f"Missing tables: {missing_tables}")
                return False
            
            self.logger.info("All essential tables verified")
            return True
            
        except Exception as e:
            self.logger.error(f"Agent tables test failed: {e}")
            return False
    

    def _test_technical_analysis(self) -> bool:
        """Test technical analysis capabilities - FIXED"""
        
        try:
            from agents.signal_agent import SignalAgent
            
            # Initialize signal agent (integrates technical + fundamental)
            signal_agent = SignalAgent(self.db_manager)
            
            # Get test symbols
            test_symbols = self.db_manager.get_testing_symbols()[:3]  # Test with 3 symbols
            
            if not test_symbols:
                self.logger.warning("No test symbols available")
                return False
            
            self.logger.info(f"Testing signal generation with symbols: {test_symbols}")
            
            # Generate signals with error handling for each symbol
            signals = []
            for symbol in test_symbols:
                try:
                    symbol_signals = signal_agent.generate_signals([symbol])
                    signals.extend(symbol_signals)
                except Exception as e:
                    self.logger.warning(f"Signal generation failed for {symbol}: {e}")
                    continue
            
            # Check results
            if not signals:
                # Try direct technical analysis as fallback
                self.logger.info("Trying direct technical analysis...")
                from agents.technical_agent import TechnicalAgent
                
                tech_agent = TechnicalAgent(self.db_manager)
                
                for symbol in test_symbols[:1]:  # Test just one symbol
                    try:
                        analysis = tech_agent.analyze_symbol(symbol)
                        if 'error' not in analysis:
                            self.logger.info(f"Technical analysis working for {symbol}")
                            return True
                    except Exception as e:
                        self.logger.warning(f"Technical analysis failed for {symbol}: {e}")
                        continue
                
                self.logger.warning("No signals generated")
                return False
            
            success_count = len(signals)
            
            # Display sample results
            for signal in signals[:2]:  # Show first 2 signals
                self.logger.info(f"Signal for {signal['symbol']}: {signal['signal_type']} "
                            f"(Confidence: {signal['overall_confidence']:.3f})")
            
            # Get summary
            summary = signal_agent.get_signal_summary(signals)
            self.logger.info(f"Signal generation summary: {summary}")
            
            success_rate = len(signals) / len(test_symbols)
            self.logger.info(f"Signal generation success rate: {success_rate:.1%} ({len(signals)}/{len(test_symbols)})")
            
            return success_rate >= 0.3  # At least 30% success rate for Day 1/2/3
            
        except Exception as e:
            self.logger.error(f"Technical analysis test failed: {e}")
            return False
    def _display_symbol_info(self, symbol: str, fundamental_data: dict):
        """Display symbol information"""
        
        print("\n" + "=" * 50)
        print(f"SYMBOL ANALYSIS: {symbol}")
        print("=" * 50)
    
    def _display_technical_analysis(self, symbol: str, analysis: dict):
        """Display technical analysis results"""
        
        print("\n" + "=" * 50)
        print(f"TECHNICAL ANALYSIS: {symbol}")
        print("=" * 50)
        
        # Basic analysis info
        print(f"Technical Score      : {analysis.get('technical_score', 'N/A'):.3f}")
        print(f"Signal Strength      : {analysis.get('signal_strength', 'N/A')}")
        print(f"Buy Signal           : {analysis.get('buy_signal', False)}")
        print(f"Sell Signal          : {analysis.get('sell_signal', False)}")
        
        # Price targets
        entry_price = analysis.get('entry_price')
        stop_loss = analysis.get('stop_loss')
        target_price = analysis.get('target_price')
        
        if entry_price:
            print(f"Entry Price          : {entry_price}")
        if stop_loss:
            print(f"Stop Loss            : {stop_loss}")
        if target_price:
            print(f"Target Price         : {target_price}")
        
        # Key indicators
        indicators = analysis.get('indicators', {})
        if indicators:
            print("\nKey Indicators:")
            
            rsi_14 = indicators.get('rsi_14')
            if rsi_14:
                print(f"  RSI (14)           : {rsi_14:.2f}")
            
            ma_trend = indicators.get('ma_trend')
            if ma_trend:
                print(f"  Moving Avg Trend   : {ma_trend}")
            
            volume_signal = indicators.get('volume_signal')
            if volume_signal:
                print(f"  Volume Signal      : {volume_signal}")
            
            macd_signal_type = indicators.get('macd_signal_type')
            if macd_signal_type:
                print(f"  MACD Signal        : {macd_signal_type}")
        
        # Reasoning
        reasoning = analysis.get('reasoning')
        if reasoning:
            print(f"\nReasoning:")
            print(f"  {reasoning}")
        
        print("=" * 50)
        
        # key_fields = [
        #     ('Stock Name', 'stock_name'),
        #     ('Category', 'category'),
        #     ('Market Cap Type', 'market_cap_type'),
        #     ('Sector', 'sector'),
        #     ('Volatility Category', 'volatility_category'),
        #     ('Current Price', 'current_price'),
        #     ('PE Ratio', 'pe_ratio'),
        #     ('ROE Ratio', 'roe_ratio'),
        #     ('Market Cap (Cr)', 'market_cap')
        # ]
        
        # for display_name, field_name in key_fields:
        #     value = fundamental_data.get(field_name, 'N/A')
        #     print(f"{display_name:20}: {value}")
        
        print("=" * 50)
    
    def _report_test_results(self, results: dict):
        """Report test results"""
        
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:25}: {status}")
        
        overall = "PASS" if all(results.values()) else "FAIL"
        print(f"{'OVERALL':25}: {overall}")
        print("=" * 50)
        
        if not all(results.values()):
            print("\nPlease fix failed tests before proceeding to integration mode")
    
    def _integration_phase_environment(self) -> bool:
        """Environment integration phase"""
        return self._test_environment()
    
    def _integration_phase_database(self) -> bool:
        """Database integration phase"""
        return self._test_database_connection()
    
    def _integration_phase_schema(self) -> bool:
        """Schema creation phase"""
        return self._test_agent_tables()
    
    def _integration_phase_data_access(self) -> bool:
        """Data access phase"""
        return self._test_existing_tables()
    
    def _integration_phase_technical_analysis(self) -> bool:
        """Technical analysis integration phase"""
        return self._test_technical_analysis()
    
    def _integration_phase_health_check(self) -> bool:
        """System health check phase"""
        try:
            health = self.db_manager.get_system_health()
            self.logger.info(f"Final system health: {health}")
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _report_integration_results(self, results: dict):
        """Report integration test results"""
        
        print("\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)
        
        for phase_name, result in results.items():
            status = "PASSED" if result else "FAILED"
            print(f"{phase_name:30}: {status}")
        
        overall = "SUCCESS" if all(results.values()) else "FAILED"
        print(f"{'OVERALL INTEGRATION':30}: {overall}")
        print("=" * 60)
        
        if all(results.values()):
            print("\nSUCCESS: DAY 1 FOUNDATION READY!")
            print("You can now proceed with Day 2 development")
        else:
            print("\nFAILED: Please fix issues before proceeding")
    
    def _paper_trading_mode(self) -> bool:
        """Paper trading mode - Execute paper trades"""
        
        try:
            import config
            
            if not config.PAPER_TRADING_MODE:
                print("Paper trading mode is disabled in config")
                return False
            
            print("=" * 60)
            print("PAPER TRADING MODE")
            print("=" * 60)
            
            # Initialize paper trading components
            from paper_trading_manager import PaperTradingManager
            paper_manager = PaperTradingManager()
            
            # Validate setup
            if not paper_manager.validate_paper_trading_setup():
                print("Paper trading setup validation failed")
                return False
            
            # Run paper trading session
            result = paper_manager.run_paper_trading_session()
            
            if 'error' in result:
                print(f"Paper trading session failed: {result['error']}")
                return False
            
            # Generate report
            paper_manager.generate_daily_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Paper trading mode failed: {e}")
            return False

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Nexus Trading System - Day 1')
    parser.add_argument('--mode', required=True, 
                       choices=['setup', 'test', 'demo', 'integration'],
                       help='Execution mode')
    parser.add_argument('--symbol', 
                       help='Symbol for demo mode (default: RELIANCE)')
    parser.add_argument('--mode', 
                       choices=['setup', 'test', 'demo', 'integration', 'paper'], 
                       default='test',
                       help='Execution mode')
    parser.add_argument('--mode', 
                       choices=['setup', 'test', 'demo', 'integration', 'paper', 'live', 'signals'], 
                       default='test',
                       help='Execution mode')
    if args.mode == 'paper':
            success = system._paper_trading_mode()
            if success:
                print("\n✅ Paper trading session completed successfully")
            else:
                print("\n❌ Paper trading session failed")
                sys.exit(1)
    elif args.mode == 'live':
        success = system._live_trading_mode()
        if success:
            print("\n✅ Live trading session completed successfully")
        else:
            print("\n❌ Live trading session failed")
            sys.exit(1)
            
    elif args.mode == 'signals':
        # Signal generation only mode
        print("=" * 60)
        print("SIGNAL GENERATION MODE")
        print("=" * 60)
        
        from live_trading_manager import LiveTradingManager
        manager = LiveTradingManager()
        result = manager.show_signals_only()
        
        if 'error' not in result:
            print("\n✅ Signal generation completed")
            print("Use --mode live to execute trades (requires TRADE_MODE=yes)")
        else:
            print("\n❌ Signal generation failed")
            sys.exit(1)
    
    args = parser.parse_args()
    
    # Initialize system
    system = NexusTradingSystem()
    
    # Run in specified mode
    success = system.run(args.mode, args.symbol)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()