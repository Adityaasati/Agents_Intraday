# reports/system_dashboard.py - System Reporting and Dashboard

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import config

class SystemReporter:
    """System reporting and dashboard data generation"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
    
    def generate_daily_report(self) -> Dict:
        """Generate daily system report"""
        try:
            today = datetime.now().date()
            
            # System performance
            system_metrics = self._get_daily_system_metrics()
            
            # Trading activity
            trading_metrics = self._get_daily_trading_metrics(today)
            
            # Alerts and issues
            alerts = self._get_daily_alerts()
            
            # Summary
            summary = self._generate_daily_summary(system_metrics, trading_metrics, alerts)
            
            report = {
                'date': today.isoformat(),
                'type': 'daily_report',
                'summary': summary,
                'system_metrics': system_metrics,
                'trading_metrics': trading_metrics,
                'alerts': alerts,
                'generated_at': datetime.now().isoformat()
            }
            
            # Save report
            self._save_report(report, f"daily_report_{today.strftime('%Y%m%d')}.json")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Daily report generation failed: {e}")
            return {'error': str(e)}
    
    def generate_health_report(self) -> Dict:
        """Generate system health report"""
        try:
            # Get system status
            from utils.dashboard_monitor import get_dashboard_monitor
            dashboard = get_dashboard_monitor(self.db_manager)
            system_status = dashboard.get_system_status()
            
            # Performance trends
            performance_trends = self._get_performance_trends()
            
            # Health recommendations
            recommendations = self._generate_health_recommendations(system_status)
            
            report = {
                'type': 'health_report',
                'current_status': system_status,
                'performance_trends': performance_trends,
                'recommendations': recommendations,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Health report generation failed: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self) -> Dict:
        """Generate performance analysis report"""
        try:
            # Get performance data
            from utils.performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            current_stats = monitor.get_current_stats()
            
            # Database performance
            db_performance = {}
            if hasattr(self.db_manager, 'get_performance_stats'):
                db_performance = self.db_manager.get_performance_stats()
            
            # Performance analysis
            analysis = self._analyze_performance(current_stats, db_performance)
            
            report = {
                'type': 'performance_report',
                'current_performance': current_stats,
                'database_performance': db_performance,
                'analysis': analysis,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}
    
    def _get_daily_system_metrics(self) -> Dict:
        """Get daily system performance metrics"""
        try:
            import psutil
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': {
                    'current': round(cpu_percent, 1),
                    'status': 'high' if cpu_percent > 80 else 'normal'
                },
                'memory_usage': {
                    'percent': round(memory.percent, 1),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'status': 'high' if memory.percent > 80 else 'normal'
                },
                'disk_usage': {
                    'percent': round(disk.percent, 1),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'status': 'high' if disk.percent > 80 else 'normal'
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    
    def _get_daily_trading_metrics(self, date) -> Dict:
        """Get daily trading activity metrics"""
        try:
            conn = self.db_manager.get_connection()
            with conn.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_live_signals'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    return {
                        'total_signals': 0,
                        'signals_by_type': {},
                        'average_confidence': 0.5,
                        'status': 'no_table'
                    }
                
                # Signals generated today
                cursor.execute("""
                    SELECT COUNT(*), signal_type 
                    FROM agent_live_signals 
                    WHERE DATE(signal_time) = %s
                    GROUP BY signal_type
                """, (date,))
                
                signals_by_type = {}
                total_signals = 0
                for row in cursor.fetchall():
                    signals_by_type[row[1]] = row[0]
                    total_signals += row[0]
                
                # Average confidence - find correct column
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'agent_live_signals' 
                    AND column_name IN ('overall_confidence', 'technical_score', 'confidence_score')
                """)
                
                confidence_columns = [row[0] for row in cursor.fetchall()]
                if confidence_columns:
                    confidence_col = confidence_columns[0]
                    cursor.execute(f"""
                        SELECT AVG({confidence_col}) 
                        FROM agent_live_signals 
                        WHERE DATE(signal_time) = %s
                    """, (date,))
                    
                    avg_confidence = cursor.fetchone()[0] or 0.5
                else:
                    avg_confidence = 0.5  # Default if no confidence column
                
                return {
                    'total_signals': total_signals,
                    'signals_by_type': signals_by_type,
                    'average_confidence': float(avg_confidence),
                    'status': 'active' if total_signals > 0 else 'inactive'
                }
                
        except Exception as e:
            self.logger.error(f"Daily trading metrics error: {e}")
            return {'error': str(e)}

    

    def _get_recent_signals(self) -> Dict:
        """Get recent trading signals"""
        try:
            conn = self.db_manager.get_connection()
            with conn.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_live_signals'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    return {'count': 0, 'recent': []}
                
                # Find confidence column
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'agent_live_signals' 
                    AND column_name IN ('overall_confidence', 'technical_score', 'confidence_score')
                """)
                
                confidence_columns = [row[0] for row in cursor.fetchall()]
                confidence_col = confidence_columns[0] if confidence_columns else '0.5 as confidence_score'
                
                cursor.execute(f"""
                    SELECT symbol, signal_type, {confidence_col}, signal_time 
                    FROM agent_live_signals 
                    WHERE signal_time >= NOW() - INTERVAL '24 hours'
                    ORDER BY signal_time DESC 
                    LIMIT 10
                """)
                
                signals = []
                for row in cursor.fetchall():
                    signals.append({
                        'symbol': row[0],
                        'type': row[1],
                        'confidence': float(row[2]) if row[2] is not None else 0.5,
                        'date': row[3].isoformat() if row[3] else datetime.now().isoformat()
                    })
                
                return {
                    'count': len(signals),
                    'recent': signals
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_daily_alerts(self) -> List[Dict]:
        """Get alerts from today"""
        try:
            # Get recent alerts from dashboard monitor
            from utils.dashboard_monitor import get_dashboard_monitor
            dashboard = get_dashboard_monitor(self.db_manager)
            alerts = dashboard.get_system_alerts()
            
            # Filter today's alerts
            today = datetime.now().date()
            today_alerts = []
            
            for alert in alerts:
                alert_date = datetime.fromisoformat(alert['timestamp']).date()
                if alert_date == today:
                    today_alerts.append(alert)
            
            return today_alerts
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def _generate_daily_summary(self, system_metrics, trading_metrics, alerts) -> Dict:
        """Generate daily summary"""
        # System health
        system_issues = 0
        if system_metrics.get('cpu_usage', {}).get('status') == 'high':
            system_issues += 1
        if system_metrics.get('memory_usage', {}).get('status') == 'high':
            system_issues += 1
        if system_metrics.get('disk_usage', {}).get('status') == 'high':
            system_issues += 1
        
        # Trading activity
        trading_active = trading_metrics.get('total_signals', 0) > 0
        
        # Alert count
        alert_count = len([a for a in alerts if 'error' not in a])
        
        # Overall status
        if system_issues == 0 and alert_count == 0:
            overall_status = 'excellent'
        elif system_issues <= 1 and alert_count <= 2:
            overall_status = 'good'
        elif system_issues <= 2 and alert_count <= 5:
            overall_status = 'warning'
        else:
            overall_status = 'critical'
        
        return {
            'overall_status': overall_status,
            'system_issues': system_issues,
            'trading_active': trading_active,
            'alert_count': alert_count,
            'total_signals': trading_metrics.get('total_signals', 0)
        }
    
    def _get_performance_trends(self) -> Dict:
        """Get performance trends over time"""
        try:
            from utils.performance_monitor import get_performance_monitor
            monitor = get_performance_monitor()
            stats = monitor.get_current_stats()
            
            return {
                'symbols_per_minute': stats.get('symbols_per_minute', 0),
                'avg_processing_time': stats.get('avg_processing_time', 0),
                'error_rate': stats.get('errors_count', 0) / max(1, stats.get('symbols_processed', 1)) * 100,
                'uptime_hours': stats.get('uptime_minutes', 0) / 60
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_health_recommendations(self, system_status) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        system = system_status.get('system', {})
        database = system_status.get('database', {})
        
        # Memory recommendations
        if system.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected - consider increasing system memory or optimizing batch sizes")
        
        # CPU recommendations
        if system.get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected - consider reducing concurrent processing or upgrading hardware")
        
        # Database recommendations
        if not database.get('connected', True):
            recommendations.append("Database connection issues detected - check database server status")
        
        # Performance recommendations
        if database.get('avg_query_time', 0) > 1:
            recommendations.append("Slow database queries detected - consider optimizing database indexes")
        
        if not recommendations:
            recommendations.append("System is performing optimally - no immediate action required")
        
        return recommendations
    
    def _analyze_performance(self, current_stats, db_performance) -> Dict:
        """Analyze performance metrics"""
        analysis = {
            'performance_score': 100,
            'bottlenecks': [],
            'strengths': []
        }
        
        # Analyze processing speed
        symbols_per_minute = current_stats.get('symbols_per_minute', 0)
        if symbols_per_minute < config.TARGET_SYMBOLS_PER_MINUTE * 0.7:
            analysis['bottlenecks'].append('Processing speed below target')
            analysis['performance_score'] -= 20
        elif symbols_per_minute >= config.TARGET_SYMBOLS_PER_MINUTE:
            analysis['strengths'].append('Processing speed meets target')
        
        # Analyze database performance
        cache_hit_rate = db_performance.get('cache_hit_rate', 0)
        if cache_hit_rate > 70:
            analysis['strengths'].append('Good database cache performance')
        elif cache_hit_rate < 30:
            analysis['bottlenecks'].append('Low database cache hit rate')
            analysis['performance_score'] -= 15
        
        # Analyze error rate
        error_count = current_stats.get('errors_count', 0)
        symbols_processed = current_stats.get('symbols_processed', 1)
        error_rate = (error_count / symbols_processed) * 100
        
        if error_rate < 5:
            analysis['strengths'].append('Low error rate')
        elif error_rate > 15:
            analysis['bottlenecks'].append('High error rate detected')
            analysis['performance_score'] -= 25
        
        return analysis
    
    def _save_report(self, report: Dict, filename: str):
        """Save report to file"""
        try:
            from pathlib import Path
            
            reports_dir = Path('reports/generated')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / filename
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")

def generate_all_reports(db_manager):
    """Generate all system reports"""
    reporter = SystemReporter(db_manager)
    
    reports = {
        'daily': reporter.generate_daily_report(),
        'health': reporter.generate_health_report(),
        'performance': reporter.generate_performance_report()
    }
    
    return reports