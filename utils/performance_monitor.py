# utils/performance_monitor.py - NEW FILE for Day 7A Performance Monitoring

import psutil
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config

class PerformanceMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self._metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'processing_times': [],
            'symbols_processed': 0,
            'errors_count': 0,
            'start_time': None,
            'last_update': datetime.now()
        }
        self._alerts = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._metrics['start_time'] = datetime.now()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._collect_metrics()
                self._check_alerts()
                time.sleep(config.MONITOR_INTERVAL_SECONDS)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def _collect_metrics(self):
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Keep last 100 readings
        self._metrics['cpu_usage'].append({
            'timestamp': datetime.now(),
            'value': cpu_percent
        })
        if len(self._metrics['cpu_usage']) > 100:
            self._metrics['cpu_usage'].pop(0)
        
        self._metrics['memory_usage'].append({
            'timestamp': datetime.now(),
            'value': memory.percent,
            'used_mb': memory.used // (1024 * 1024)
        })
        if len(self._metrics['memory_usage']) > 100:
            self._metrics['memory_usage'].pop(0)
        
        self._metrics['last_update'] = datetime.now()
    
    def _check_alerts(self):
        """Check for performance alerts"""
        if not self._metrics['memory_usage'] or not self._metrics['cpu_usage']:
            return
        
        current_memory = self._metrics['memory_usage'][-1]['used_mb']
        current_cpu = self._metrics['cpu_usage'][-1]['value']
        
        # Memory alert
        if current_memory > config.MAX_MEMORY_USAGE_MB:
            self._add_alert('HIGH_MEMORY', f"Memory usage: {current_memory}MB")
        
        # CPU alert
        if current_cpu > 80:
            self._add_alert('HIGH_CPU', f"CPU usage: {current_cpu}%")
        
        # Processing time alert
        if self._metrics['processing_times']:
            avg_time = sum(self._metrics['processing_times'][-10:]) / min(10, len(self._metrics['processing_times']))
            if avg_time > config.PERFORMANCE_ALERT_THRESHOLD:
                self._add_alert('SLOW_PROCESSING', f"Avg time: {avg_time:.2f}s per symbol")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add performance alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now()
        }
        
        # Avoid duplicate alerts within 5 minutes
        recent_alerts = [a for a in self._alerts if 
                        (datetime.now() - a['timestamp']).seconds < 300 and 
                        a['type'] == alert_type]
        
        if not recent_alerts:
            self._alerts.append(alert)
            self.logger.warning(f"Performance Alert [{alert_type}]: {message}")
    
    def record_processing_time(self, symbol: str, processing_time: float):
        """Record symbol processing time"""
        self._metrics['processing_times'].append(processing_time)
        if len(self._metrics['processing_times']) > 1000:
            self._metrics['processing_times'].pop(0)
        
        self._metrics['symbols_processed'] += 1
        
        if config.LOG_PERFORMANCE_DETAILS:
            self.logger.debug(f"Processed {symbol} in {processing_time:.2f}s")
    
    def record_error(self, error_type: str):
        """Record processing error"""
        self._metrics['errors_count'] += 1
        self.logger.warning(f"Processing error recorded: {error_type}")
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self._metrics['memory_usage'] or not self._metrics['cpu_usage']:
            return {'status': 'no_data'}
        
        # Calculate averages
        recent_cpu = [m['value'] for m in self._metrics['cpu_usage'][-10:]]
        recent_memory = [m['used_mb'] for m in self._metrics['memory_usage'][-10:]]
        
        avg_cpu = sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0
        avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
        
        # Processing speed
        symbols_per_minute = 0
        if self._metrics['start_time']:
            elapsed_minutes = (datetime.now() - self._metrics['start_time']).seconds / 60
            if elapsed_minutes > 0:
                symbols_per_minute = self._metrics['symbols_processed'] / elapsed_minutes
        
        # Recent processing times
        avg_processing_time = 0
        if self._metrics['processing_times']:
            recent_times = self._metrics['processing_times'][-20:]
            avg_processing_time = sum(recent_times) / len(recent_times)
        
        return {
            'cpu_usage': round(avg_cpu, 1),
            'memory_usage_mb': round(avg_memory, 1),
            'memory_usage_percent': round(self._metrics['memory_usage'][-1]['value'], 1),
            'symbols_processed': self._metrics['symbols_processed'],
            'symbols_per_minute': round(symbols_per_minute, 1),
            'avg_processing_time': round(avg_processing_time, 2),
            'errors_count': self._metrics['errors_count'],
            'uptime_minutes': round((datetime.now() - self._metrics['start_time']).seconds / 60, 1) if self._metrics['start_time'] else 0,
            'active_alerts': len([a for a in self._alerts if (datetime.now() - a['timestamp']).seconds < 600])
        }
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        stats = self.get_current_stats()
        
        # Performance assessment
        performance_status = 'good'
        if stats.get('memory_usage_mb', 0) > config.MAX_MEMORY_USAGE_MB * 0.8:
            performance_status = 'warning'
        if stats.get('cpu_usage', 0) > 70 or stats.get('avg_processing_time', 0) > config.MAX_PROCESSING_TIME_PER_SYMBOL:
            performance_status = 'warning'
        if stats.get('active_alerts', 0) > 0:
            performance_status = 'alert'
        
        # Recent alerts
        recent_alerts = [a for a in self._alerts if (datetime.now() - a['timestamp']).seconds < 1800]
        
        return {
            'status': performance_status,
            'stats': stats,
            'recent_alerts': recent_alerts[-5:],  # Last 5 alerts
            'recommendations': self._get_recommendations(stats)
        }
    
    def _get_recommendations(self, stats: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if stats.get('memory_usage_mb', 0) > config.MAX_MEMORY_USAGE_MB * 0.7:
            recommendations.append("Consider reducing batch size or clearing cache")
        
        if stats.get('avg_processing_time', 0) > config.MAX_PROCESSING_TIME_PER_SYMBOL * 0.8:
            recommendations.append("Processing time approaching limit - optimize queries")
        
        if stats.get('symbols_per_minute', 0) < config.TARGET_SYMBOLS_PER_MINUTE * 0.7:
            recommendations.append("Processing speed below target - check database performance")
        
        if stats.get('errors_count', 0) > 0:
            recommendations.append("Errors detected - check logs for issues")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations
    
    def cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean old alerts
        self._alerts = [a for a in self._alerts if a['timestamp'] > cutoff_time]
        
        self.logger.info("Performance monitoring data cleaned")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def start_monitoring():
    """Start global performance monitoring"""
    if config.ENABLE_PERFORMANCE_MONITORING:
        get_performance_monitor().start_monitoring()

def stop_monitoring():
    """Stop global performance monitoring"""
    monitor = get_performance_monitor()
    if monitor:
        monitor.stop_monitoring()