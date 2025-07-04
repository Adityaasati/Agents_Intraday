#!/usr/bin/env python3
"""
Market Hours Scheduler - Runs data updates during market hours only
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

class MarketScheduler:
    """Schedule market data updates during trading hours only"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logging.getLogger('MarketScheduler')
        self.running = False
        self.thread = None
        
    def is_market_time(self) -> bool:
        """Check if it's market hours (9:10 AM - 3:35 PM IST, weekdays)"""
        now = datetime.now()
        
        # No weekends
        if now.weekday() >= 5:
            return False
            
        # Market hours: 9:10 AM to 3:35 PM
        start_time = now.replace(hour=9, minute=10, second=0, microsecond=0)
        end_time = now.replace(hour=15, minute=35, second=0, microsecond=0)
        
        return start_time <= now <= end_time
    
    def update_market_data(self):
        """Update market data using MarketDataFetcher"""
        try:
            from utils.market_data_fetcher import MarketDataFetcher
            fetcher = MarketDataFetcher(self.db_manager)
            
            # Update data for current symbols
            result = fetcher.update_current_quarter_table()
            
            if result['updated'] > 0:
                self.logger.info(f"Updated {result['updated']} symbols")
            
            if result['failed'] > 0:
                self.logger.warning(f"Failed to update {result['failed']} symbols")
                
        except Exception as e:
            self.logger.error(f"Market data update failed: {e}")
    
    def run_scheduler(self):
        """Main scheduler loop - runs in background thread"""
        self.logger.info("Market scheduler started - will run 9:10 AM to 3:35 PM on weekdays")
        
        while self.running:
            try:
                if self.is_market_time():
                    self.update_market_data()
                    # Wait 5 minutes for next update
                    time.sleep(300)  # 5 minutes
                else:
                    # Outside market hours - check every hour
                    next_market = self.get_next_market_time()
                    if next_market:
                        self.logger.info(f"Market closed. Next market opens: {next_market}")
                    time.sleep(3600)  # 1 hour
                    
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def get_next_market_time(self) -> Optional[datetime]:
        """Get next market opening time"""
        now = datetime.now()
        
        # If it's weekend, next Monday at 9:10 AM
        if now.weekday() >= 5:
            days_ahead = 7 - now.weekday()  # Days until Monday
            next_monday = now + timedelta(days=days_ahead)
            return next_monday.replace(hour=9, minute=10, second=0, microsecond=0)
        
        # If it's before market hours today
        market_start = now.replace(hour=9, minute=10, second=0, microsecond=0)
        if now < market_start:
            return market_start
            
        # If it's after market hours, next day at 9:10 AM
        tomorrow = now + timedelta(days=1)
        if tomorrow.weekday() >= 5:  # If tomorrow is weekend
            days_ahead = 7 - tomorrow.weekday()
            next_monday = tomorrow + timedelta(days=days_ahead)
            return next_monday.replace(hour=9, minute=10, second=0, microsecond=0)
        else:
            return tomorrow.replace(hour=9, minute=10, second=0, microsecond=0)
    
    def start(self):
        """Start the scheduler in background thread"""
        if self.running:
            self.logger.warning("Scheduler already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.thread.start()
        self.logger.info("Market scheduler thread started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("Market scheduler stopped")