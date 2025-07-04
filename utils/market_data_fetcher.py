#!/usr/bin/env python3
"""
Market Data Fetcher - Real historical data integration
Minimal integration with existing data_updater.py
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from dotenv import load_dotenv

load_dotenv()

class MarketDataFetcher:
    """Fetch real market data via Kite API"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logging.getLogger('MarketDataFetcher')
        self.kite = None
        
    def get_kite_client(self):
        """Get authenticated Kite client"""
        if self.kite:
            return self.kite
            
        try:
            from kite_token_generator import get_authenticated_kite_client
            self.kite = get_authenticated_kite_client()
            return self.kite
        except Exception as e:
            self.logger.error(f"Kite auth failed: {e}")
            return None
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for symbol"""
        kite = self.get_kite_client()
        if not kite:
            return None
            
        try:
            instruments = pd.DataFrame(kite.instruments("NSE"))
            match = instruments[
                (instruments['tradingsymbol'] == symbol) & 
                (instruments['instrument_type'] == 'EQ')
            ]
            return match.iloc[0]['instrument_token'] if not match.empty else None
        except Exception as e:
            self.logger.error(f"Token lookup failed for {symbol}: {e}")
            return None
    
    def fetch_latest_data(self, symbol: str) -> Optional[Dict]:
        """Fetch latest 5-minute data for symbol"""
        kite = self.get_kite_client()
        if not kite:
            return None
            
        try:
            token = self.get_instrument_token(symbol)
            if not token:
                return None
                
            # Get last 2 bars to ensure we have latest
            end_date = datetime.now()
            start_date = end_date - timedelta(minutes=15)
            
            data = kite.historical_data(token, start_date, end_date, "5minute")
            
            if data:
                latest = data[-1]  # Get most recent bar
                return {
                    'date': latest['date'],
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': int(latest['volume'])
                }
        except Exception as e:
            self.logger.error(f"Data fetch failed for {symbol}: {e}")
            
        return None
    
    def update_current_quarter_table(self, symbols: List[str] = None) -> Dict:
        """Update current quarter table with latest data"""
        if not self.db_manager:
            return {'status': 'error', 'message': 'No database manager'}
            
        if not symbols:
            symbols = self.db_manager.get_testing_symbols(10)  # Get 10 symbols for testing
            
        results = {'updated': 0, 'failed': 0, 'errors': []}
        
        for symbol in symbols:
            try:
                data = self.fetch_latest_data(symbol)
                if data:
                    # Use existing data updater method to store data
                    from utils.data_updater import DataUpdater
                    updater = DataUpdater(self.db_manager)
                    if updater.update_symbol_data(symbol, data):
                        results['updated'] += 1
                    else:
                        results['failed'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{symbol}: {str(e)}")
                
        return results
    
    def is_market_hours(self) -> bool:
        """Check if current time is market hours"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
            
        import config
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end