#!/usr/bin/env python3
"""
Historical Data Service - CLI and API compatible service for downloading market data
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from typing import Union, List, Dict, Optional
from datetime import datetime
from database.enhanced_database_manager import EnhancedDatabaseManager
from agents.historical_data_agent import HistoricalDataAgent
import logging

class HistoricalDataService:
    """Service for historical data operations - works for both CLI and API"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or EnhancedDatabaseManager()
        self.logger = logging.getLogger('historical_data_service')
    
    def download_data(
        self,
        interval: str = '5minute',
        start_date: str = '2020-01-01',
        end_date: str = 'today',
        symbols: Union[str, List[str]] = 'all',
        download_frequency: str = 'once'
    ) -> Dict:
        """
        Download historical data with progress tracking
        
        Returns:
            Dict with status, message, progress, errors
        """
        try:
            # Initialize agent
            agent = HistoricalDataAgent(self.db_manager)
            
            # Override settings
            agent.interval = interval
            agent.download_frequency = download_frequency
            
            # Parse dates
            if start_date != 'default':
                agent.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date != 'today':
                agent.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Handle symbols
            if isinstance(symbols, str) and symbols != 'all':
                symbols = symbols.split(',')
            
            if symbols != 'all':
                # Override get_symbols method
                agent.get_symbols = lambda x: symbols
            
            # Check if update needed
            if download_frequency != 'once':
                is_current = agent.is_data_current()
                if is_current:
                    return {
                        'status': 'success',
                        'message': 'Data is already up to date',
                        'updated': False
                    }
            
            # Run download
            success = agent.run_download()
            
            return {
                'status': 'success' if success else 'failed',
                'message': 'Download completed' if success else 'Download failed',
                'updated': True,
                'parameters': {
                    'interval': interval,
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': symbols,
                    'download_frequency': download_frequency
                }
            }
            
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'updated': False
            }

# For CLI usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download historical data')
    parser.add_argument('--interval', default='5minute', choices=['5minute', '15minute', '60minute'])
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='today', help='End date (YYYY-MM-DD or "today")')
    parser.add_argument('--symbols', default='all', help='all or comma-separated symbols')
    parser.add_argument('--frequency', default='once', choices=['once', '5min', '15min', '60min'])
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    service = HistoricalDataService()
    result = service.download_data(
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols,
        download_frequency=args.frequency
    )
    
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result.get('updated'):
        print(f"Parameters: {result.get('parameters', {})}")

if __name__ == "__main__":
    main()