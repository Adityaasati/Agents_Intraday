# utils/dashboard_monitor.py - Dashboard Monitoring System

import psutil
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import config

class DashboardMonitor:
    """Real-time dashboard monitoring system"""
    
    def __init__(self, db_manager=None):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self._metrics_cache = {}
        self._last_update = None
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database status
            db_status = self._get_database_status()
            
            # Performance metrics
            perf_metrics = self._get_performance_metrics()
            
            # Overall health score
            health_score = self._calculate_health_score(cpu_percent, memory.percent, db_status['connected'])
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': round(cpu_percent, 1),
                    'memory_percent': round(memory.percent, 1),
                    'memory_used_gb': round(memory.used / (1024**3), 2),
                    'disk_percent': round(disk.percent, 1),
                    'disk_free_gb': round(disk.free / (1024**3), 2)
                },
                'database': db_status,
                'performance': perf_metrics,
                'health_score': health_score,
                'status': self._get_status_text(health_score)
            }
        except Exception as e:
            self.logger.error(f"System status error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_trading_metrics(self) -> Dict:
        """Get trading performance metrics"""
        if not self.db_manager:
            return {'error': 'database_unavailable'}
        
        try:
            # Recent signals
            recent_signals = self._get_recent_signals()
            
            # Performance stats
            performance_stats = self._get_trading_performance()
            
            # Active positions
            active_positions = self._get_active_positions()
            
            return {
                'signals': recent_signals,
                'performance': performance_stats,
                'positions': active_positions,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Trading metrics error: {e}")
            return {'error': str(e)}
    
    def get_system_alerts(self) -> List[Dict]:
        """Get current system alerts"""
        alerts = []
        
        try:
            # Performance alerts
            from utils.performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            summary = monitor.get_performance_summary()
            
            if summary['status'] == 'alert':
                for alert in summary.get('recent_alerts', []):
                    alerts.append({
                        'type': 'performance',
                        'severity': 'warning',
                        'message': alert['message'],
                        'timestamp': alert['timestamp'].isoformat()
                    })
            
            # System resource alerts
            system_status = self.get_system_status()
            if system_status.get('system', {}).get('memory_percent', 0) > 80:
                alerts.append({
                    'type': 'system',
                    'severity': 'warning',
                    'message': f"High memory usage: {system_status['system']['memory_percent']}%",
                    'timestamp': datetime.now().isoformat()
                })
            
            if system_status.get('system', {}).get('cpu_percent', 0) > 80:
                alerts.append({
                    'type': 'system',
                    'severity': 'warning',
                    'message': f"High CPU usage: {system_status['system']['cpu_percent']}%",
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            self.logger.error(f"Alerts error: {e}")
        
        return alerts[-10:]  # Return last 10 alerts
    
    def _get_database_status(self) -> Dict:
        """Get database connection status"""
        if not self.db_manager:
            return {'connected': False, 'error': 'no_manager'}
        
        try:
            # Test connection
            test_result = self.db_manager.test_connection()
            
            # Get performance stats if available (safe call)
            perf_stats = {}
            if hasattr(self.db_manager, 'get_performance_stats'):
                try:
                    perf_stats = self.db_manager.get_performance_stats()
                except Exception as e:
                    self.logger.warning(f"Performance stats unavailable: {e}")
                    perf_stats = {'error': str(e)}
            
            return {
                'connected': test_result,
                'pool_status': perf_stats.get('pool_status', 'unknown'),
                'queries_executed': perf_stats.get('queries_executed', 0),
                'cache_hit_rate': perf_stats.get('cache_hit_rate', 0),
                'avg_query_time': perf_stats.get('avg_query_time', 0)
            }
        except Exception as e:
            return {'connected': False, 'error': str(e)}
    
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance monitoring metrics"""
        try:
            from utils.performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            stats = monitor.get_current_stats()
            
            return {
                'symbols_processed': stats.get('symbols_processed', 0),
                'symbols_per_minute': stats.get('symbols_per_minute', 0),
                'avg_processing_time': stats.get('avg_processing_time', 0),
                'errors_count': stats.get('errors_count', 0),
                'uptime_minutes': stats.get('uptime_minutes', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_recent_signals(self) -> Dict:
        """Get recent trading signals"""
        try:
            conn = self.db_manager.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT symbol, signal_type, confidence_score, signal_date 
                    FROM agent_live_signals 
                    WHERE signal_date >= NOW() - INTERVAL '24 hours'
                    ORDER BY signal_date DESC 
                    LIMIT 10
                """)
                
                signals = []
                for row in cursor.fetchall():
                    signals.append({
                        'symbol': row[0],
                        'type': row[1],
                        'confidence': float(row[2]),
                        'date': row[3].isoformat()
                    })
                
                return {
                    'count': len(signals),
                    'recent': signals
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_trading_performance(self) -> Dict:
        """Get trading performance statistics"""
        try:
            conn = self.db_manager.get_connection()
            with conn.cursor() as cursor:
                # Get signal counts by type
                cursor.execute("""
                    SELECT signal_type, COUNT(*), AVG(confidence_score)
                    FROM agent_live_signals 
                    WHERE signal_date >= NOW() - INTERVAL '7 days'
                    GROUP BY signal_type
                """)
                
                signal_stats = {}
                for row in cursor.fetchall():
                    signal_stats[row[0]] = {
                        'count': row[1],
                        'avg_confidence': float(row[2])
                    }
                
                return {
                    'signals_7d': signal_stats,
                    'total_signals': sum(s['count'] for s in signal_stats.values()),
                    'avg_confidence': sum(s['avg_confidence'] * s['count'] for s in signal_stats.values()) / max(1, sum(s['count'] for s in signal_stats.values()))
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_active_positions(self) -> Dict:
        """Get active positions information"""
        try:
            # This would integrate with portfolio manager
            return {
                'total_positions': 0,
                'total_value': 0,
                'positions': []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_health_score(self, cpu: float, memory: float, db_connected: bool) -> int:
        """Calculate overall system health score (0-100)"""
        score = 100
        
        # CPU penalty
        if cpu > 80:
            score -= 30
        elif cpu > 60:
            score -= 15
        
        # Memory penalty
        if memory > 80:
            score -= 30
        elif memory > 60:
            score -= 15
        
        # Database penalty
        if not db_connected:
            score -= 40
        
        return max(0, score)
    
    def _get_status_text(self, health_score: int) -> str:
        """Get status text based on health score"""
        if health_score >= 80:
            return 'excellent'
        elif health_score >= 60:
            return 'good'
        elif health_score >= 40:
            return 'warning'
        else:
            return 'critical'
    
    def generate_dashboard_data(self) -> Dict:
        """Generate complete dashboard data"""
        return {
            'system_status': self.get_system_status(),
            'trading_metrics': self.get_trading_metrics(),
            'alerts': self.get_system_alerts(),
            'config': {
                'refresh_interval': config.DASHBOARD_AUTO_REFRESH,
                'theme': config.DASHBOARD_THEME,
                'production_mode': config.PRODUCTION_MODE
            }
        }

# Global dashboard monitor instance
_dashboard_monitor = None

def get_dashboard_monitor(db_manager=None):
    """Get global dashboard monitor instance"""
    global _dashboard_monitor
    if _dashboard_monitor is None:
        _dashboard_monitor = DashboardMonitor(db_manager)
    return _dashboard_monitor