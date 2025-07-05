#!/usr/bin/env python3
"""
Optional Pool Monitoring - Advanced features that can be added separately
This file is completely optional and doesn't affect existing functionality
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import psutil
import os

class PoolHealthMonitor:
    """Optional advanced monitoring for database connection pool"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger('nexus_trading.pool_monitor')
        self.monitoring_active = False
        self.monitor_thread = None
        self.stats_history = []
        
        # Alert thresholds - can be customized
        self.alert_thresholds = {
            'exhaustion_count': int(os.getenv('POOL_EXHAUSTION_ALERT_THRESHOLD', '10')),
            'memory_usage': float(os.getenv('MEMORY_ALERT_THRESHOLD', '85.0')),
            'cpu_usage': float(os.getenv('CPU_ALERT_THRESHOLD', '90.0'))
        }
    
    def start_monitoring(self, interval_seconds=30):
        """Start optional monitoring (only if you want advanced features)"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Pool monitoring started (optional feature)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Pool monitoring stopped")
    
    def _monitor_loop(self, interval_seconds):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self._collect_stats()
                self._analyze_stats(stats)
                self._store_stats(stats)
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Pool monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_stats(self) -> Dict:
        """Collect current pool and system statistics"""
        stats = {
            'timestamp': datetime.now(),
            'pool_status': {},
            'system_memory': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'active_connections': 0
        }
        
        # Get pool status if available
        if hasattr(self.db_manager, 'get_pool_status'):
            stats['pool_status'] = self.db_manager.get_pool_status()
        
        return stats
    
    def _analyze_stats(self, stats: Dict):
        """Analyze stats for potential issues"""
        pool_status = stats.get('pool_status', {})
        
        # Check for pool exhaustion
        exhaustion_count = pool_status.get('pool_exhaustion_count', 0)
        if exhaustion_count > self.alert_thresholds['exhaustion_count']:
            self.logger.warning(f"HIGH POOL EXHAUSTION: {exhaustion_count} events")
            self._suggest_optimization()
        
        # Check system resources
        if stats['system_memory'] > self.alert_thresholds['memory_usage']:
            self.logger.warning(f"HIGH MEMORY USAGE: {stats['system_memory']:.1f}%")
        
        if stats['cpu_usage'] > self.alert_thresholds['cpu_usage']:
            self.logger.warning(f"HIGH CPU USAGE: {stats['cpu_usage']:.1f}%")
    
    def _suggest_optimization(self):
        """Suggest optimizations based on current issues"""
        suggestions = [
            "Consider increasing DB_POOL_MAX in .env file",
            "Enable sequential testing: ENABLE_SEQUENTIAL_TESTING=true",
            "Increase delay between tests: TEST_DELAY_SECONDS=3",
            "Check PostgreSQL max_connections setting"
        ]
        
        self.logger.info("OPTIMIZATION SUGGESTIONS:")
        for suggestion in suggestions:
            self.logger.info(f"  - {suggestion}")
    
    def _store_stats(self, stats: Dict):
        """Store stats history (keep last 100 entries)"""
        self.stats_history.append(stats)
        if len(self.stats_history) > 100:
            self.stats_history.pop(0)
    
    def get_monitoring_report(self) -> Dict:
        """Generate monitoring report"""
        if not self.stats_history:
            return {'status': 'no_data', 'message': 'No monitoring data available'}
        
        recent_stats = self.stats_history[-10:] if len(self.stats_history) >= 10 else self.stats_history
        
        # Calculate averages
        avg_exhaustion = sum(s.get('pool_status', {}).get('pool_exhaustion_count', 0) for s in recent_stats) / len(recent_stats)
        avg_memory = sum(s.get('system_memory', 0) for s in recent_stats) / len(recent_stats)
        avg_cpu = sum(s.get('cpu_usage', 0) for s in recent_stats) / len(recent_stats)
        
        # Determine health status
        health_score = 100
        if avg_exhaustion > 5:
            health_score -= 30
        if avg_memory > 80:
            health_score -= 25
        if avg_cpu > 80:
            health_score -= 20
        
        health_status = 'excellent' if health_score >= 90 else 'good' if health_score >= 70 else 'warning' if health_score >= 50 else 'critical'
        
        return {
            'status': 'active',
            'health_score': max(0, health_score),
            'health_status': health_status,
            'avg_pool_exhaustion': round(avg_exhaustion, 2),
            'avg_memory_usage': round(avg_memory, 2),
            'avg_cpu_usage': round(avg_cpu, 2),
            'monitoring_duration_minutes': (datetime.now() - self.stats_history[0]['timestamp']).total_seconds() / 60,
            'recommendations': self._get_recommendations(health_status, avg_exhaustion)
        }
    
    def _get_recommendations(self, health_status: str, avg_exhaustion: float) -> List[str]:
        """Get recommendations based on current health"""
        recommendations = []
        
        if health_status == 'critical':
            recommendations.append("URGENT: System needs immediate attention")
            
        if avg_exhaustion > 10:
            recommendations.append("Increase database connection pool size")
            recommendations.append("Enable sequential test execution")
            
        if health_status in ['warning', 'critical']:
            recommendations.append("Monitor system more frequently")
            recommendations.append("Consider reducing concurrent operations")
        
        return recommendations

class SimplePoolOptimizer:
    """Simple optimizer that suggests configuration improvements"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger('nexus_trading.pool_optimizer')
    
    def analyze_current_setup(self) -> Dict:
        """Analyze current setup and suggest improvements"""
        analysis = {
            'current_pool_config': self._get_current_pool_config(),
            'system_resources': self._get_system_resources(),
            'suggestions': []
        }
        
        # Get pool status if available
        if hasattr(self.db_manager, 'get_pool_status'):
            pool_status = self.db_manager.get_pool_status()
            exhaustion_count = pool_status.get('pool_exhaustion_count', 0)
            
            if exhaustion_count > 0:
                analysis['suggestions'].append({
                    'type': 'pool_size',
                    'priority': 'high',
                    'suggestion': f"Increase DB_POOL_MAX (current exhaustion count: {exhaustion_count})",
                    'specific_action': f"Set DB_POOL_MAX={min(50, analysis['current_pool_config']['max_connections'] + 10)}"
                })
            
            avg_query_time = pool_status.get('avg_query_time', 0)
            if avg_query_time > 2.0:
                analysis['suggestions'].append({
                    'type': 'performance',
                    'priority': 'medium',
                    'suggestion': f"Slow queries detected (avg: {avg_query_time:.3f}s)",
                    'specific_action': "Enable query caching and connection pooling optimizations"
                })
        
        # System resource analysis
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            analysis['suggestions'].append({
                'type': 'memory',
                'priority': 'medium',
                'suggestion': f"High memory usage: {memory_percent:.1f}%",
                'specific_action': "Consider reducing cache sizes or adding more RAM"
            })
        
        return analysis
    
    def _get_current_pool_config(self) -> Dict:
        """Get current pool configuration"""
        try:
            from database.connection_config import get_pool_config
            return get_pool_config()
        except:
            return {'min_connections': 'unknown', 'max_connections': 'unknown'}
    
    def _get_system_resources(self) -> Dict:
        """Get current system resource usage"""
        try:
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'available_memory_gb': psutil.virtual_memory().available / 1024**3
            }
        except:
            return {'memory_percent': 'unknown', 'cpu_percent': 'unknown'}

# USAGE EXAMPLE - Add to your main.py if you want advanced monitoring
def integrate_optional_monitoring(main_instance):
    """Optional integration example - only add if you want advanced features"""
    
    def start_optional_monitoring(self):
        """Start optional advanced monitoring"""
        if os.getenv('ENABLE_ADVANCED_MONITORING', 'false').lower() == 'true':
            try:
                from database.pool_monitoring import PoolHealthMonitor, SimplePoolOptimizer
                
                self.pool_monitor = PoolHealthMonitor(self.db_manager)
                self.pool_optimizer = SimplePoolOptimizer(self.db_manager)
                
                self.pool_monitor.start_monitoring()
                self.logger.info("Advanced pool monitoring enabled")
                
                # Show initial analysis
                analysis = self.pool_optimizer.analyze_current_setup()
                if analysis['suggestions']:
                    self.logger.info("Pool optimization suggestions:")
                    for suggestion in analysis['suggestions']:
                        self.logger.info(f"  {suggestion['priority'].upper()}: {suggestion['suggestion']}")
                
            except ImportError:
                self.logger.info("Advanced monitoring not available")
    
    def stop_optional_monitoring(self):
        """Stop optional monitoring"""
        if hasattr(self, 'pool_monitor'):
            self.pool_monitor.stop_monitoring()
    
    def get_advanced_health_report(self):
        """Get advanced health report"""
        if hasattr(self, 'pool_monitor'):
            return self.pool_monitor.get_monitoring_report()
        return {'status': 'not_available'}

# This file is completely optional and can be added later for advanced features
# Your system will work perfectly without itcls
