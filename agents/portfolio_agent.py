"""
FILE: agents/portfolio_agent.py
PURPOSE: Portfolio Management Agent - Basic portfolio tracking and management

DESCRIPTION:
- Tracks portfolio positions and performance
- Manages capital allocation across signals
- Monitors portfolio risk and diversification
- Generates portfolio reports and statistics

DEPENDENCIES:
- database/enhanced_database_manager.py
- agents/risk_agent.py
- config.py

USAGE:
- Called by main system for portfolio management
- Used for position tracking and performance monitoring
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz
import config
from .base_agent import BaseAgent

class PortfolioAgent(BaseAgent):
    """Portfolio management and tracking agent"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
    
    def allocate_capital(self, signals: List[Dict], total_capital: float = None) -> Dict:
        """Allocate capital across multiple signals"""
        
        if total_capital is None:
            total_capital = config.TOTAL_CAPITAL
        
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Calculate available capital
            allocated_capital = sum(pos.get('position_value', 0) for pos in current_positions)
            available_capital = total_capital - allocated_capital
            
            # Filter signals that pass risk checks
            from agents.risk_agent import RiskAgent
            risk_agent = RiskAgent(self.db_manager)
            
            approved_signals = []
            
            for signal in signals:
                # Calculate position size
                position_info = risk_agent.calculate_position_size(signal, total_capital)
                
                if 'error' not in position_info:
                    signal['position_info'] = position_info
                    
                    # Check portfolio risk
                    risk_check = risk_agent.portfolio_risk_check(current_positions, position_info)
                    
                    if risk_check.get('recommendation') == 'APPROVE':
                        approved_signals.append(signal)
                        # Update current positions for next iteration
                        current_positions.append(position_info)
            
            allocation_result = {
                'total_capital': total_capital,
                'allocated_capital': allocated_capital,
                'available_capital': available_capital,
                'signals_evaluated': len(signals),
                'signals_approved': len(approved_signals),
                'approved_signals': approved_signals,
                'allocation_summary': self._generate_allocation_summary(approved_signals)
            }
            
            return allocation_result
            
        except Exception as e:
            self.logger.error(f"Capital allocation failed: {e}")
            return {'error': str(e)}
    
    def get_current_positions(self) -> List[Dict]:
        """Get current portfolio positions"""
        
        try:
            # Get active signals from database
            active_signals = self.db_manager.get_active_signals(limit=50)
            
            positions = []
            for signal in active_signals:
                position = {
                    'signal_id': signal.get('id'),
                    'symbol': signal.get('symbol'),
                    'signal_type': signal.get('signal_type'),
                    'entry_price': signal.get('entry_price', 0),
                    'stop_loss': signal.get('stop_loss', 0),
                    'target_price': signal.get('target_price', 0),
                    'position_value': signal.get('recommended_position_size', 0),
                    'sector': self._get_symbol_sector(signal.get('symbol')),
                    'signal_time': signal.get('signal_time'),
                    'confidence': signal.get('overall_confidence', 0)
                }
                positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get current positions: {e}")
            return []
    
    def calculate_portfolio_performance(self) -> Dict:
        """Calculate portfolio performance metrics"""
        
        try:
            positions = self.get_current_positions()
            
            if not positions:
                return {
                    'total_positions': 0,
                    'total_value': 0,
                    'portfolio_status': 'empty'
                }
            
            total_value = sum(pos.get('position_value', 0) for pos in positions)
            
            # Sector breakdown
            sector_breakdown = {}
            for pos in positions:
                sector = pos.get('sector', 'Unknown')
                sector_breakdown[sector] = sector_breakdown.get(sector, 0) + pos.get('position_value', 0)
            
            # Signal type breakdown
            signal_breakdown = {}
            for pos in positions:
                signal_type = pos.get('signal_type', 'Unknown')
                signal_breakdown[signal_type] = signal_breakdown.get(signal_type, 0) + 1
            
            # Risk metrics
            total_risk = sum(self._calculate_position_risk(pos) for pos in positions)
            portfolio_risk_percent = (total_risk / config.TOTAL_CAPITAL) * 100 if total_risk > 0 else 0
            
            performance = {
                'total_positions': len(positions),
                'total_value': round(total_value, 2),
                'capital_utilization': round((total_value / config.TOTAL_CAPITAL) * 100, 2),
                'portfolio_risk_percent': round(portfolio_risk_percent, 2),
                'sector_breakdown': {k: round(v, 2) for k, v in sector_breakdown.items()},
                'signal_breakdown': signal_breakdown,
                'avg_confidence': round(sum(pos.get('confidence', 0) for pos in positions) / len(positions), 3),
                'diversification_score': self._calculate_diversification_score(sector_breakdown, total_value)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Portfolio performance calculation failed: {e}")
            return {'error': str(e)}
    
    def check_diversification_rules(self, current_positions: List[Dict], new_signal: Dict) -> Dict:
        """Check portfolio diversification rules"""
        
        try:
            # Sector diversification
            sectors = set(pos.get('sector', 'Unknown') for pos in current_positions)
            new_sector = self._get_symbol_sector(new_signal.get('symbol', ''))
            sectors.add(new_sector)
            
            # Check minimum sector requirement
            min_sectors_ok = len(sectors) >= config.MIN_DIVERSIFICATION_SECTORS
            
            # Check sector concentration
            sector_values = {}
            total_value = 0
            
            for pos in current_positions:
                sector = pos.get('sector', 'Unknown')
                value = pos.get('position_value', 0)
                sector_values[sector] = sector_values.get(sector, 0) + value
                total_value += value
            
            # Add new signal
            new_value = new_signal.get('recommended_position_value', 0)
            sector_values[new_sector] = sector_values.get(new_sector, 0) + new_value
            total_value += new_value
            
            # Check sector limits
            max_sector_percent = 0
            if total_value > 0:
                max_sector_percent = max(v / total_value * 100 for v in sector_values.values())
            
            sector_concentration_ok = max_sector_percent <= config.MAX_SECTOR_ALLOCATION
            
            return {
                'diversification_ok': min_sectors_ok and sector_concentration_ok,
                'sectors_count': len(sectors),
                'min_sectors_required': config.MIN_DIVERSIFICATION_SECTORS,
                'max_sector_concentration': round(max_sector_percent, 2),
                'max_allowed_concentration': config.MAX_SECTOR_ALLOCATION,
                'sector_breakdown': {k: round(v / total_value * 100, 2) for k, v in sector_values.items()} if total_value > 0 else {}
            }
            
        except Exception as e:
            self.logger.error(f"Diversification check failed: {e}")
            return {'error': str(e)}
    
    def generate_portfolio_report(self) -> Dict:
        """Generate comprehensive portfolio report"""
        
        try:
            performance = self.calculate_portfolio_performance()
            positions = self.get_current_positions()
            
            report = {
                'report_time': datetime.now(self.ist),
                'portfolio_overview': performance,
                'position_details': positions[:10],  # Top 10 positions
                'risk_summary': self._generate_risk_summary(positions),
                'recommendations': self._generate_portfolio_recommendations(performance, positions)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Portfolio report generation failed: {e}")
            return {'error': str(e)}
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        try:
            fundamental_data = self.db_manager.get_fundamental_data(symbol)
            return fundamental_data.get('sector', 'Unknown')
        except:
            return 'Unknown'
    
    def _calculate_position_risk(self, position: Dict) -> float:
        """Calculate risk amount for a position"""
        try:
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss', 0)
            position_value = position.get('position_value', 0)
            
            if entry_price > 0 and stop_loss > 0:
                shares = position_value / entry_price
                risk_per_share = entry_price - stop_loss
                return shares * risk_per_share
            
            return 0
        except:
            return 0
    
    def _calculate_diversification_score(self, sector_breakdown: Dict, total_value: float) -> float:
        """Calculate diversification score (0-1, higher is better)"""
        
        if not sector_breakdown or total_value <= 0:
            return 0.0
        
        # Calculate Herfindahl index (concentration measure)
        sector_shares = [v / total_value for v in sector_breakdown.values()]
        herfindahl_index = sum(share ** 2 for share in sector_shares)
        
        # Convert to diversification score (1 - HHI)
        # Perfect diversification would be 1/n where n is number of sectors
        max_diversification = 1 / len(sector_breakdown) if len(sector_breakdown) > 0 else 1
        diversification_score = (1 - herfindahl_index) / (1 - max_diversification) if max_diversification < 1 else 1
        
        return round(max(0.0, min(1.0, diversification_score)), 3)
    
    def _generate_allocation_summary(self, approved_signals: List[Dict]) -> Dict:
        """Generate capital allocation summary"""
        
        if not approved_signals:
            return {'total_allocation': 0, 'signal_count': 0}
        
        total_allocation = sum(s.get('position_info', {}).get('recommended_position_value', 0) for s in approved_signals)
        
        return {
            'total_allocation': round(total_allocation, 2),
            'signal_count': len(approved_signals),
            'avg_allocation_per_signal': round(total_allocation / len(approved_signals), 2),
            'allocation_by_strength': self._group_by_signal_strength(approved_signals)
        }
    
    def _group_by_signal_strength(self, signals: List[Dict]) -> Dict:
        """Group signals by strength"""
        
        strength_groups = {}
        
        for signal in signals:
            strength = signal.get('signal_strength', 'medium')
            allocation = signal.get('position_info', {}).get('recommended_position_value', 0)
            
            if strength not in strength_groups:
                strength_groups[strength] = {'count': 0, 'total_allocation': 0}
            
            strength_groups[strength]['count'] += 1
            strength_groups[strength]['total_allocation'] += allocation
        
        # Round allocation values
        for strength_data in strength_groups.values():
            strength_data['total_allocation'] = round(strength_data['total_allocation'], 2)
        
        return strength_groups
    
    def _generate_risk_summary(self, positions: List[Dict]) -> Dict:
        """Generate risk summary for portfolio"""
        
        total_risk = sum(self._calculate_position_risk(pos) for pos in positions)
        
        return {
            'total_risk_amount': round(total_risk, 2),
            'risk_as_percent_of_capital': round((total_risk / config.TOTAL_CAPITAL) * 100, 2),
            'risk_limit': config.MAX_DRAWDOWN,
            'risk_utilization': round((total_risk / config.TOTAL_CAPITAL) / (config.MAX_DRAWDOWN / 100) * 100, 2),
            'positions_count': len(positions),
            'avg_risk_per_position': round(total_risk / len(positions), 2) if positions else 0
        }
    
    def _generate_portfolio_recommendations(self, performance: Dict, positions: List[Dict]) -> List[str]:
        """Generate portfolio recommendations"""
        
        recommendations = []
        
        # Utilization recommendations
        utilization = performance.get('capital_utilization', 0)
        if utilization < 50:
            recommendations.append("Consider increasing position sizes or adding more signals")
        elif utilization > 90:
            recommendations.append("Portfolio near full capacity, consider reducing position sizes")
        
        # Diversification recommendations
        diversification = performance.get('diversification_score', 0)
        if diversification < 0.5:
            recommendations.append("Improve diversification across sectors")
        
        # Risk recommendations
        risk_percent = performance.get('portfolio_risk_percent', 0)
        if risk_percent > config.MAX_DRAWDOWN * 0.8:
            recommendations.append("Portfolio risk approaching limits, consider reducing exposure")
        
        # Position count recommendations
        position_count = len(positions)
        if position_count < 3:
            recommendations.append("Consider adding more positions for better diversification")
        elif position_count > config.MAX_POSITIONS_LIVE * 0.8:
            recommendations.append("Approaching maximum position limit")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    # Add these methods to existing agents/portfolio_agent.py file
    # Insert at the end of the PortfolioAgent class

    def allocate_capital_optimized(self, signals: List[Dict], total_capital: float = None) -> Dict:
        """Optimized capital allocation with correlation and risk analysis"""
        
        if total_capital is None:
            total_capital = config.TOTAL_CAPITAL
        
        try:
            from agents.risk_agent import RiskAgent
            risk_agent = RiskAgent(self.db_manager)
            
            current_positions = self.get_current_positions()
            
            # Calculate available capital
            allocated_capital = sum(pos.get('position_value', 0) for pos in current_positions)
            available_capital = total_capital - allocated_capital
            
            # Get current portfolio risk metrics
            portfolio_risk = risk_agent.get_portfolio_risk_metrics(current_positions)
            
            # Sort signals by opportunity score
            ranked_signals = self._rank_signals_by_opportunity(signals, current_positions)
            
            approved_signals = []
            cumulative_positions = current_positions.copy()
            
            for signal in ranked_signals:
                # Enhanced position sizing
                enhanced_position = risk_agent.calculate_enhanced_position_size(signal, cumulative_positions)
                
                if 'error' in enhanced_position:
                    continue
                
                signal['enhanced_position_info'] = enhanced_position
                
                # Portfolio risk check with enhanced metrics
                risk_check = risk_agent.portfolio_risk_check(cumulative_positions, enhanced_position)
                correlation_check = risk_agent.calculate_portfolio_correlation(cumulative_positions, signal.get('symbol'))
                
                # Additional checks for optimized allocation
                if (risk_check.get('recommendation') == 'APPROVE' and 
                    correlation_check.get('correlation_risk') != 'high' and
                    enhanced_position.get('recommended_position_value', 0) <= available_capital):
                    
                    approved_signals.append(signal)
                    cumulative_positions.append(enhanced_position)
                    available_capital -= enhanced_position.get('recommended_position_value', 0)
            
            # Generate allocation report
            allocation_report = self._generate_optimized_allocation_report(
                approved_signals, portfolio_risk, total_capital, allocated_capital
            )
            
            return allocation_report
            
        except Exception as e:
            self.logger.error(f"Optimized capital allocation failed: {e}")
            return self.allocate_capital(signals, total_capital)  # Fallback

    def monitor_portfolio_risk(self) -> Dict:
        """Real-time portfolio risk monitoring"""
        
        try:
            from agents.risk_agent import RiskAgent
            risk_agent = RiskAgent(self.db_manager)
            
            current_positions = self.get_current_positions()
            risk_metrics = risk_agent.get_portfolio_risk_metrics(current_positions)
            
            # Risk alerts
            alerts = []
            
            # Total risk check
            if risk_metrics.get('total_risk_percent', 0) > 20:
                alerts.append('HIGH_PORTFOLIO_RISK')
            
            # Concentration risk check
            if risk_metrics.get('concentration_risk', 0) > 30:
                alerts.append('HIGH_SECTOR_CONCENTRATION')
            
            # Correlation risk check
            if risk_metrics.get('correlation_risk') == 'high':
                alerts.append('HIGH_CORRELATION_RISK')
            
            # Position count check
            if len(current_positions) > config.MAX_POSITIONS_LIVE:
                alerts.append('POSITION_LIMIT_EXCEEDED')
            
            # Risk budget utilization
            risk_budget_used = risk_metrics.get('risk_budget_used', 0)
            if risk_budget_used > 90:
                alerts.append('RISK_BUDGET_EXHAUSTED')
            
            monitoring_result = {
                'timestamp': datetime.now(self.ist),
                'portfolio_health': 'GOOD' if not alerts else 'CAUTION' if len(alerts) < 3 else 'CRITICAL',
                'risk_metrics': risk_metrics,
                'alerts': alerts,
                'recommendations': self._generate_risk_recommendations(alerts, risk_metrics),
                'next_review': self._calculate_next_review_time(alerts)
            }
            
            # Store monitoring result
            self._store_risk_monitoring_result(monitoring_result)
            
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Portfolio risk monitoring failed: {e}")
            return {'error': str(e), 'portfolio_health': 'UNKNOWN'}

    def optimize_cash_allocation(self) -> Dict:
        """Optimize cash allocation and maintain buffers"""
        
        try:
            current_positions = self.get_current_positions()
            total_invested = sum(pos.get('position_value', 0) for pos in current_positions)
            
            cash_available = config.TOTAL_CAPITAL - total_invested
            
            # Calculate optimal cash buffer (10-20% of capital)
            min_cash_buffer = config.TOTAL_CAPITAL * 0.10  # 10% minimum
            optimal_cash_buffer = config.TOTAL_CAPITAL * 0.15  # 15% optimal
            
            # Cash allocation recommendations
            if cash_available < min_cash_buffer:
                recommendation = 'REDUCE_POSITIONS'
                urgency = 'HIGH'
            elif cash_available < optimal_cash_buffer:
                recommendation = 'MAINTAIN_CURRENT'
                urgency = 'MEDIUM'
            elif cash_available > config.TOTAL_CAPITAL * 0.30:  # Over 30% cash
                recommendation = 'INCREASE_POSITIONS'
                urgency = 'MEDIUM'
            else:
                recommendation = 'OPTIMAL'
                urgency = 'LOW'
            
            return {
                'total_capital': config.TOTAL_CAPITAL,
                'total_invested': round(total_invested, 2),
                'cash_available': round(cash_available, 2),
                'cash_percentage': round((cash_available / config.TOTAL_CAPITAL) * 100, 2),
                'min_cash_buffer': round(min_cash_buffer, 2),
                'optimal_cash_buffer': round(optimal_cash_buffer, 2),
                'recommendation': recommendation,
                'urgency': urgency,
                'capacity_for_new_positions': max(0, cash_available - optimal_cash_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"Cash allocation optimization failed: {e}")
            return {'error': str(e)}

    def get_portfolio_diversification_score(self) -> Dict:
        """Calculate portfolio diversification score"""
        
        try:
            current_positions = self.get_current_positions()
            
            if not current_positions:
                return {'diversification_score': 0, 'sectors': 0, 'recommendations': ['ADD_POSITIONS']}
            
            # Sector analysis
            sectors = {}
            total_value = 0
            
            for pos in current_positions:
                sector = self._get_symbol_sector(pos.get('symbol'))
                value = pos.get('position_value', 0)
                sectors[sector] = sectors.get(sector, 0) + value
                total_value += value
            
            # Calculate diversification metrics
            num_sectors = len(sectors)
            
            # Sector concentration (Herfindahl index)
            sector_weights = [value / total_value for value in sectors.values()]
            herfindahl_index = sum(weight ** 2 for weight in sector_weights)
            
            # Position concentration
            position_weights = [pos.get('position_value', 0) / total_value for pos in current_positions]
            position_concentration = max(position_weights) if position_weights else 0
            
            # Diversification score (0-100)
            sector_score = min(num_sectors * 15, 60)  # Max 60 for sectors
            concentration_score = max(0, 40 - (herfindahl_index * 100))  # Max 40 for concentration
            diversification_score = sector_score + concentration_score
            
            # Recommendations
            recommendations = []
            if num_sectors < 3:
                recommendations.append('ADD_MORE_SECTORS')
            if position_concentration > 0.20:
                recommendations.append('REDUCE_LARGEST_POSITION')
            if herfindahl_index > 0.4:
                recommendations.append('IMPROVE_SECTOR_BALANCE')
            
            return {
                'diversification_score': round(diversification_score, 1),
                'sector_count': num_sectors,
                'sector_distribution': {k: round((v/total_value)*100, 1) for k, v in sectors.items()},
                'herfindahl_index': round(herfindahl_index, 3),
                'max_position_weight': round(position_concentration * 100, 1),
                'recommendations': recommendations,
                'grade': self._get_diversification_grade(diversification_score)
            }
            
        except Exception as e:
            self.logger.error(f"Diversification score calculation failed: {e}")
            return {'diversification_score': 0, 'error': str(e)}

    # Helper methods for optimization

    def _rank_signals_by_opportunity(self, signals: List[Dict], current_positions: List[Dict]) -> List[Dict]:
        """Rank signals by opportunity considering portfolio context"""
        
        try:
            from agents.risk_agent import RiskAgent
            risk_agent = RiskAgent(self.db_manager)
            
            ranked_signals = []
            
            for signal in signals:
                # Calculate opportunity score
                confidence = signal.get('overall_confidence', 0)
                technical_score = signal.get('technical_score', 0)
                
                # Portfolio context bonus
                symbol = signal.get('symbol')
                correlation_data = risk_agent.calculate_portfolio_correlation(current_positions, symbol)
                correlation_bonus = 0.1 if correlation_data.get('correlation_risk') == 'low' else 0
                
                # Sector diversification bonus
                symbol_sector = self._get_symbol_sector(symbol)
                current_sectors = set(self._get_symbol_sector(pos.get('symbol')) for pos in current_positions)
                sector_bonus = 0.1 if symbol_sector not in current_sectors else 0
                
                opportunity_score = confidence + (technical_score * 0.3) + correlation_bonus + sector_bonus
                
                signal['opportunity_score'] = round(opportunity_score, 3)
                ranked_signals.append(signal)
            
            # Sort by opportunity score (descending)
            return sorted(ranked_signals, key=lambda x: x.get('opportunity_score', 0), reverse=True)
            
        except Exception:
            return signals

    def _generate_optimized_allocation_report(self, approved_signals: List[Dict], 
                                            portfolio_risk: Dict, total_capital: float, 
                                            allocated_capital: float) -> Dict:
        """Generate comprehensive allocation report"""
        
        new_allocation = sum(signal.get('enhanced_position_info', {}).get('recommended_position_value', 0) 
                            for signal in approved_signals)
        
        return {
            'optimization_applied': True,
            'total_capital': total_capital,
            'previously_allocated': allocated_capital,
            'new_allocation': round(new_allocation, 2),
            'total_after_allocation': round(allocated_capital + new_allocation, 2),
            'cash_remaining': round(total_capital - allocated_capital - new_allocation, 2),
            'capital_utilization': round(((allocated_capital + new_allocation) / total_capital) * 100, 2),
            'signals_evaluated': len(approved_signals),
            'approved_signals': approved_signals,
            'portfolio_risk_before': portfolio_risk,
            'diversification_improvement': self._calculate_diversification_improvement(approved_signals),
            'expected_portfolio_beta': self._estimate_portfolio_beta(approved_signals)
        }

    def _generate_risk_recommendations(self, alerts: List[str], risk_metrics: Dict) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if 'HIGH_PORTFOLIO_RISK' in alerts:
            recommendations.append('Consider reducing position sizes or closing weak positions')
        
        if 'HIGH_SECTOR_CONCENTRATION' in alerts:
            recommendations.append('Diversify across more sectors to reduce concentration risk')
        
        if 'HIGH_CORRELATION_RISK' in alerts:
            recommendations.append('Avoid adding highly correlated positions')
        
        if 'POSITION_LIMIT_EXCEEDED' in alerts:
            recommendations.append('Close some positions before adding new ones')
        
        if 'RISK_BUDGET_EXHAUSTED' in alerts:
            recommendations.append('Wait for risk to decrease before new positions')
        
        if not recommendations:
            recommendations.append('Portfolio risk is within acceptable limits')
        
        return recommendations

    def _calculate_next_review_time(self, alerts: List[str]) -> str:
        """Calculate when next risk review is needed"""
        
        if any(alert in ['HIGH_PORTFOLIO_RISK', 'RISK_BUDGET_EXHAUSTED'] for alert in alerts):
            return 'IMMEDIATE'
        elif len(alerts) > 0:
            return 'WITHIN_4_HOURS'
        else:
            return 'NEXT_TRADING_DAY'

    def _store_risk_monitoring_result(self, result: Dict):
        """Store risk monitoring result in database"""
        
        try:
            # Store in database for historical tracking
            monitoring_data = {
                'timestamp': result['timestamp'],
                'portfolio_health': result['portfolio_health'],
                'total_risk_percent': result['risk_metrics'].get('total_risk_percent', 0),
                'concentration_risk': result['risk_metrics'].get('concentration_risk', 0),
                'correlation_risk': result['risk_metrics'].get('correlation_risk', 'unknown'),
                'alerts_count': len(result['alerts']),
                'alerts': ','.join(result['alerts'])
            }
            
            # Use existing database storage method
            self.db_manager.store_portfolio_monitoring(monitoring_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to store risk monitoring result: {e}")

    def _get_diversification_grade(self, score: float) -> str:
        """Get diversification grade based on score"""
        
        if score >= 80:
            return 'EXCELLENT'
        elif score >= 60:
            return 'GOOD'
        elif score >= 40:
            return 'FAIR'
        else:
            return 'POOR'

    def _calculate_diversification_improvement(self, new_signals: List[Dict]) -> Dict:
        """Calculate how new signals improve diversification"""
        
        try:
            new_sectors = set(self._get_symbol_sector(signal.get('symbol')) for signal in new_signals)
            return {
                'new_sectors_added': len(new_sectors),
                'sectors': list(new_sectors)
            }
        except:
            return {'new_sectors_added': 0, 'sectors': []}

    def _estimate_portfolio_beta(self, signals: List[Dict]) -> float:
        """Estimate portfolio beta (simplified)"""
        
        try:
            # Simplified beta estimation based on volatility categories
            total_weight = 0
            weighted_beta = 0
            
            for signal in signals:
                volatility = signal.get('volatility_category', 'Medium')
                weight = signal.get('enhanced_position_info', {}).get('recommended_position_value', 0)
                
                # Estimate beta based on volatility
                beta = {'Low': 0.8, 'Medium': 1.0, 'High': 1.3}.get(volatility, 1.0)
                
                weighted_beta += beta * weight
                total_weight += weight
            
            return round(weighted_beta / total_weight, 2) if total_weight > 0 else 1.0
            
        except:
            return 1.0
    
    # Add these methods to existing agents/portfolio_agent.py file
    # Insert at the end of the PortfolioAgent class

    def construct_risk_parity_portfolio(self, available_signals: List[Dict]) -> Dict:
        """Construct risk parity portfolio from available signals"""
        
        try:
            if len(available_signals) < config.MIN_PORTFOLIO_POSITIONS:
                return {'error': 'insufficient_signals', 'min_required': config.MIN_PORTFOLIO_POSITIONS}
            
            # Get symbols from signals
            symbols = [signal.get('symbol') for signal in available_signals]
            
            # Calculate risk parity weights
            from agents.risk_agent import RiskAgent
            risk_agent = RiskAgent(self.db_manager)
            risk_parity_weights = risk_agent.calculate_risk_parity_weights(symbols)
            
            # Apply weights to create portfolio
            total_capital = config.TOTAL_CAPITAL
            available_capital = self._get_available_capital()
            
            portfolio_positions = []
            allocated_capital = 0
            
            for signal in available_signals:
                symbol = signal.get('symbol')
                weight = risk_parity_weights.get(symbol, 0)
                
                if weight > 0:
                    # Calculate position size based on weight
                    target_value = available_capital * weight
                    entry_price = signal.get('entry_price', 0)
                    
                    if entry_price > 0:
                        shares = int(target_value / entry_price)
                        actual_value = shares * entry_price
                        
                        if actual_value > 0:
                            position = signal.copy()
                            position.update({
                                'recommended_shares': shares,
                                'recommended_position_value': actual_value,
                                'portfolio_weight': weight,
                                'risk_contribution': weight  # In risk parity, weight = risk contribution
                            })
                            portfolio_positions.append(position)
                            allocated_capital += actual_value
            
            return {
                'strategy': 'risk_parity',
                'positions': portfolio_positions,
                'total_positions': len(portfolio_positions),
                'allocated_capital': allocated_capital,
                'target_weights': risk_parity_weights,
                'capital_utilization': (allocated_capital / available_capital) * 100,
                'diversification_achieved': len(portfolio_positions) >= config.MIN_PORTFOLIO_POSITIONS
            }
            
        except Exception as e:
            self.logger.error(f"Risk parity portfolio construction failed: {e}")
            return {'error': str(e)}

    def calculate_rebalancing_needs(self) -> Dict:
        """Calculate portfolio rebalancing requirements"""
        
        try:
            current_positions = self.get_current_positions()
            
            if len(current_positions) < 2:
                return {'rebalancing_needed': False, 'reason': 'insufficient_positions'}
            
            # Calculate current weights
            total_value = sum(pos.get('position_value', 0) for pos in current_positions)
            current_weights = {
                pos.get('symbol'): pos.get('position_value', 0) / total_value 
                for pos in current_positions
            }
            
            # Get target weights (risk parity)
            symbols = list(current_weights.keys())
            from agents.risk_agent import RiskAgent
            risk_agent = RiskAgent(self.db_manager)
            target_weights = risk_agent.calculate_risk_parity_weights(symbols)
            
            # Calculate deviations
            rebalancing_actions = []
            max_deviation = 0
            
            for symbol in symbols:
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                deviation = abs(current_weight - target_weight) * 100
                
                if deviation > config.REBALANCING_THRESHOLD_PERCENT:
                    action = 'reduce' if current_weight > target_weight else 'increase'
                    rebalancing_actions.append({
                        'symbol': symbol,
                        'action': action,
                        'current_weight': round(current_weight * 100, 2),
                        'target_weight': round(target_weight * 100, 2),
                        'deviation': round(deviation, 2)
                    })
                
                max_deviation = max(max_deviation, deviation)
            
            rebalancing_needed = len(rebalancing_actions) > 0
            
            return {
                'rebalancing_needed': rebalancing_needed,
                'max_deviation': round(max_deviation, 2),
                'actions_required': len(rebalancing_actions),
                'rebalancing_actions': rebalancing_actions,
                'current_weights': {k: round(v*100, 2) for k, v in current_weights.items()},
                'target_weights': {k: round(v*100, 2) for k, v in target_weights.items()},
                'next_rebalancing_date': self._calculate_next_rebalancing_date()
            }
            
        except Exception as e:
            self.logger.error(f"Rebalancing calculation failed: {e}")
            return {'rebalancing_needed': False, 'error': str(e)}

    def calculate_efficient_frontier(self, signals: List[Dict]) -> Dict:
        """Calculate efficient frontier for portfolio optimization"""
        
        try:
            if len(signals) < config.MIN_PORTFOLIO_POSITIONS:
                return {'error': 'insufficient_signals'}
            
            symbols = [signal.get('symbol') for signal in signals]
            
            # Get historical data for optimization
            returns_data = self._get_returns_matrix(symbols)
            
            if returns_data is None:
                return {'error': 'insufficient_historical_data'}
            
            # Calculate efficient frontier points
            frontier_points = []
            min_return, max_return = config.TARGET_RETURN_RANGE
            return_levels = [min_return + (max_return - min_return) * i / (config.EFFICIENT_FRONTIER_POINTS - 1) 
                            for i in range(config.EFFICIENT_FRONTIER_POINTS)]
            
            for target_return in return_levels:
                try:
                    # Simplified mean-variance optimization
                    optimal_weights = self._optimize_portfolio(returns_data, target_return)
                    portfolio_risk = self._calculate_portfolio_risk(returns_data, optimal_weights)
                    
                    frontier_points.append({
                        'expected_return': round(target_return, 2),
                        'portfolio_risk': round(portfolio_risk, 2),
                        'sharpe_ratio': round((target_return - config.RISK_FREE_RATE) / portfolio_risk, 3),
                        'weights': {symbols[i]: round(weight, 3) for i, weight in enumerate(optimal_weights)}
                    })
                    
                except Exception:
                    continue
            
            # Find optimal portfolio (max Sharpe ratio)
            if frontier_points:
                optimal_portfolio = max(frontier_points, key=lambda x: x['sharpe_ratio'])
            else:
                optimal_portfolio = None
            
            return {
                'frontier_points': frontier_points,
                'optimal_portfolio': optimal_portfolio,
                'symbols': symbols,
                'risk_free_rate': config.RISK_FREE_RATE,
                'optimization_successful': len(frontier_points) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Efficient frontier calculation failed: {e}")
            return {'error': str(e)}

    def calculate_portfolio_attribution(self, period: str = '1M') -> Dict:
        """Calculate performance attribution for portfolio"""
        
        try:
            current_positions = self.get_current_positions()
            
            if not current_positions:
                return {'error': 'no_positions'}
            
            # Calculate period returns (simplified)
            period_mapping = {'1D': 1, '1W': 7, '1M': 30, '3M': 90}
            days_back = period_mapping.get(period, 30)
            
            attribution_data = []
            total_portfolio_return = 0
            total_value = sum(pos.get('position_value', 0) for pos in current_positions)
            
            for position in current_positions:
                symbol = position.get('symbol')
                weight = position.get('position_value', 0) / total_value
                
                # Calculate position return (simplified)
                position_return = self._calculate_position_return(symbol, days_back)
                contribution = weight * position_return
                
                attribution_data.append({
                    'symbol': symbol,
                    'weight': round(weight * 100, 2),
                    'return': round(position_return, 2),
                    'contribution': round(contribution, 2),
                    'sector': self._get_symbol_sector(symbol)
                })
                
                total_portfolio_return += contribution
            
            # Calculate sector attribution
            sector_attribution = self._calculate_sector_attribution(attribution_data)
            
            return {
                'period': period,
                'portfolio_return': round(total_portfolio_return, 2),
                'position_attribution': attribution_data,
                'sector_attribution': sector_attribution,
                'top_contributors': sorted(attribution_data, key=lambda x: x['contribution'], reverse=True)[:3],
                'bottom_contributors': sorted(attribution_data, key=lambda x: x['contribution'])[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Performance attribution failed: {e}")
            return {'error': str(e)}

    def generate_portfolio_optimization_report(self, signals: List[Dict]) -> Dict:
        """Generate comprehensive portfolio optimization report"""
        
        try:
            # Current portfolio analysis
            current_performance = self.calculate_portfolio_performance()
            
            # Risk parity portfolio
            risk_parity_portfolio = self.construct_risk_parity_portfolio(signals)
            
            # Rebalancing analysis
            rebalancing_analysis = self.calculate_rebalancing_needs()
            
            # Efficient frontier
            efficient_frontier = self.calculate_efficient_frontier(signals)
            
            # Performance attribution
            performance_attribution = self.calculate_portfolio_attribution()
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                current_performance, risk_parity_portfolio, rebalancing_analysis, efficient_frontier
            )
            
            return {
                'timestamp': datetime.now(self.ist),
                'current_portfolio': current_performance,
                'risk_parity_portfolio': risk_parity_portfolio,
                'rebalancing_analysis': rebalancing_analysis,
                'efficient_frontier': efficient_frontier,
                'performance_attribution': performance_attribution,
                'recommendations': recommendations,
                'optimization_summary': {
                    'current_positions': current_performance.get('total_positions', 0),
                    'capital_utilization': current_performance.get('capital_utilization', 0),
                    'diversification_score': current_performance.get('diversification_score', 0),
                    'rebalancing_needed': rebalancing_analysis.get('rebalancing_needed', False),
                    'optimization_feasible': efficient_frontier.get('optimization_successful', False)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization report failed: {e}")
            return {'error': str(e)}

    # Helper methods for portfolio optimization

    def _get_available_capital(self) -> float:
        """Get available capital for new positions"""
        
        current_positions = self.get_current_positions()
        allocated_capital = sum(pos.get('position_value', 0) for pos in current_positions)
        return config.TOTAL_CAPITAL - allocated_capital

    def _get_returns_matrix(self, symbols: List[str]) -> dict:
        """Get returns matrix for portfolio optimization"""
        
        try:
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=config.OPTIMIZATION_LOOKBACK_MONTHS * 30)
            
            returns_data = {}
            
            for symbol in symbols:
                df = self.db_manager.get_historical_data(symbol, start_date, end_date)
                if not df.empty and len(df) > 20:
                    df['returns'] = df['close'].pct_change().dropna()
                    returns_data[symbol] = df['returns'].values
            
            # Ensure all return series have same length
            if len(returns_data) >= config.MIN_PORTFOLIO_POSITIONS:
                min_length = min(len(returns) for returns in returns_data.values())
                returns_data = {symbol: returns[-min_length:] for symbol, returns in returns_data.items()}
                return returns_data
            
            return None
            
        except Exception:
            return None

    def _optimize_portfolio(self, returns_data: dict, target_return: float) -> List[float]:
        """Optimize portfolio weights for target return (simplified)"""
        
        import numpy as np
        
        symbols = list(returns_data.keys())
        returns_matrix = np.array([returns_data[symbol] for symbol in symbols]).T
        
        # Calculate mean returns and covariance
        mean_returns = np.mean(returns_matrix, axis=0) * 252  # Annualized
        cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized
        
        # Simplified optimization (equal risk contribution with target return constraint)
        num_assets = len(symbols)
        
        # Start with equal weights
        weights = np.ones(num_assets) / num_assets
        
        # Adjust weights to approximate target return (simplified approach)
        current_return = np.dot(weights, mean_returns)
        if current_return != 0:
            adjustment_factor = target_return / current_return
            weights = weights * adjustment_factor
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Apply constraints
            max_weight = config.MAX_SINGLE_ASSET_WEIGHT / 100
            weights = np.minimum(weights, max_weight)
            weights = weights / np.sum(weights)
        
        return weights.tolist()

    def _calculate_portfolio_risk(self, returns_data: dict, weights: List[float]) -> float:
        """Calculate portfolio risk given weights"""
        
        import numpy as np
        
        symbols = list(returns_data.keys())
        returns_matrix = np.array([returns_data[symbol] for symbol in symbols]).T
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized
        
        # Calculate portfolio risk
        weights_array = np.array(weights)
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        portfolio_risk = np.sqrt(portfolio_variance) * 100  # Convert to percentage
        
        return portfolio_risk

    def _calculate_position_return(self, symbol: str, days_back: int) -> float:
        """Calculate position return over specified period"""
        
        try:
            end_date = datetime.now(self.ist)
            start_date = end_date - timedelta(days=days_back)
            df = self.db_manager.get_historical_data(symbol, start_date, end_date)
            
            if df.empty or len(df) < 2:
                return 0.0
            
            start_price = df.iloc[0]['close']
            end_price = df.iloc[-1]['close']
            
            return ((end_price - start_price) / start_price) * 100
            
        except Exception:
            return 0.0

    def _calculate_sector_attribution(self, attribution_data: List[Dict]) -> Dict:
        """Calculate sector-level performance attribution"""
        
        sector_data = {}
        
        for position in attribution_data:
            sector = position.get('sector', 'Unknown')
            
            if sector not in sector_data:
                sector_data[sector] = {
                    'total_weight': 0,
                    'total_contribution': 0,
                    'position_count': 0
                }
            
            sector_data[sector]['total_weight'] += position['weight']
            sector_data[sector]['total_contribution'] += position['contribution']
            sector_data[sector]['position_count'] += 1
        
        # Calculate sector returns
        for sector, data in sector_data.items():
            data['sector_return'] = (data['total_contribution'] / data['total_weight'] * 100) if data['total_weight'] > 0 else 0
            data['total_weight'] = round(data['total_weight'], 2)
            data['total_contribution'] = round(data['total_contribution'], 2)
            data['sector_return'] = round(data['sector_return'], 2)
        
        return sector_data

    def _calculate_next_rebalancing_date(self) -> str:
        """Calculate next rebalancing date"""
        
        next_date = datetime.now(self.ist) + timedelta(days=config.REBALANCING_FREQUENCY_DAYS)
        return next_date.strftime('%Y-%m-%d')

    def _generate_optimization_recommendations(self, current_portfolio: Dict, 
                                            risk_parity: Dict, rebalancing: Dict, 
                                            efficient_frontier: Dict) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Current portfolio analysis
        if current_portfolio.get('total_positions', 0) < config.MIN_PORTFOLIO_POSITIONS:
            recommendations.append(f"Increase positions to minimum {config.MIN_PORTFOLIO_POSITIONS} for better diversification")
        
        # Diversification
        diversification_score = current_portfolio.get('diversification_score', 0)
        if diversification_score < 60:
            recommendations.append("Improve diversification across sectors and asset types")
        
        # Rebalancing
        if rebalancing.get('rebalancing_needed', False):
            max_deviation = rebalancing.get('max_deviation', 0)
            recommendations.append(f"Rebalancing recommended - maximum deviation {max_deviation:.1f}%")
        
        # Risk parity
        if 'error' not in risk_parity and risk_parity.get('diversification_achieved', False):
            recommendations.append("Consider risk parity approach for better risk distribution")
        
        # Efficient frontier analysis
        if efficient_frontier.get('optimization_successful', False):
            optimal_portfolio = efficient_frontier.get('optimal_portfolio')
            if optimal_portfolio:
                sharpe_ratio = optimal_portfolio.get('sharpe_ratio', 0)
                recommendations.append(f"Optimal portfolio achieves Sharpe ratio of {sharpe_ratio:.2f}")
        
        # Cash utilization
        cash_utilization = current_portfolio.get('capital_utilization', 0)
        if cash_utilization < 70:
            recommendations.append("Consider increasing capital utilization for better returns")
        elif cash_utilization > 95:
            recommendations.append("Maintain cash buffer for opportunities and risk management")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _optimize_portfolio_fallback(self, returns_data: dict, target_return: float) -> List[float]:
        """Fallback portfolio optimization without numpy"""
        
        symbols = list(returns_data.keys())
        num_assets = len(symbols)
        
        # Simple equal-weight approach as fallback
        weights = [1.0 / num_assets] * num_assets
        
        # Apply maximum weight constraint
        max_weight = self.config.MAX_SINGLE_ASSET_WEIGHT / 100
        
        for i in range(num_assets):
            if weights[i] > max_weight:
                weights[i] = max_weight
        
        # Renormalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights

    def _calculate_portfolio_risk_fallback(self, returns_data: dict, weights: List[float]) -> float:
        """Fallback portfolio risk calculation without numpy"""
        
        # Simplified risk calculation
        # Equal weight average of individual volatilities
        symbols = list(returns_data.keys())
        
        weighted_vol = 0
        for i, symbol in enumerate(symbols):
            returns = returns_data[symbol]
            # Calculate volatility
            if len(returns) > 1:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                volatility = (variance ** 0.5) * (252 ** 0.5)  # Annualized
            else:
                volatility = 0.20  # Default 20% volatility
            
            weighted_vol += weights[i] * volatility
        
        return weighted_vol * 100  # Return as percentage
    def _safe_import_numpy(self):
        """Safe numpy import with fallback"""
        
        try:
            import numpy as np
            return np
        except ImportError:
            # Create a simple fallback for basic operations
            class NumpyFallback:
                @staticmethod
                def array(data):
                    return data
                
                @staticmethod
                def ones(size):
                    return [1.0] * size
                
                @staticmethod
                def mean(data, axis=None):
                    if isinstance(data[0], (list, tuple)):
                        return [sum(col)/len(col) for col in zip(*data)]
                    return sum(data) / len(data)
                
                @staticmethod
                def dot(a, b):
                    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                        return sum(x * y for x, y in zip(a, b))
                    return a * b
                
                @staticmethod
                def sqrt(x):
                    return x ** 0.5
                
                @staticmethod
                def sum(data):
                    return sum(data)
                
                @staticmethod
                def minimum(a, b):
                    if isinstance(a, (list, tuple)):
                        return [min(x, b) for x in a]
                    return min(a, b)
                
                @staticmethod
                def cov(data):
                    # Simplified covariance calculation
                    n = len(data)
                    if n < 2:
                        return [[1.0]]
                    
                    # Very simplified - return identity-like matrix
                    cov_matrix = []
                    for i in range(n):
                        row = [0.0] * n
                        row[i] = 1.0  # Diagonal elements
                        cov_matrix.append(row)
                    return cov_matrix
            
            return NumpyFallback()
    
    # Add these methods to the existing PortfolioAgent class in agents/portfolio_agent.py

    def execute_paper_trade(self, signal: Dict) -> Dict:
        """Execute paper trade from signal"""
        
        try:
            import config
            from datetime import datetime
            
            if not config.PAPER_TRADING_MODE:
                return {'error': 'Paper trading disabled'}
            
            # Simulate execution delay
            import time
            time.sleep(1)
            
            # Calculate execution details
            execution_price = signal.get('entry_price', 0)
            slippage = execution_price * config.PAPER_TRADING_SLIPPAGE / 100
            final_price = execution_price + slippage
            
            commission = signal.get('recommended_position_value', 0) * config.PAPER_TRADING_COMMISSION / 100
            
            # Create paper position
            position_data = {
                'symbol': signal.get('symbol'),
                'signal_id': signal.get('id'),
                'entry_time': datetime.now(),
                'entry_price': final_price,
                'quantity': signal.get('recommended_shares', 0),
                'position_value': signal.get('recommended_position_value', 0),
                'stop_loss': signal.get('stop_loss'),
                'target_price': signal.get('target_price'),
                'commission_paid': commission,
                'status': 'OPEN'
            }
            
            # Store in database
            position_id = self._store_paper_position(position_data)
            
            self.logger.info(f"Paper trade executed: {signal.get('symbol')} @ {final_price}")
            
            return {
                'status': 'executed',
                'position_id': position_id,
                'execution_price': final_price,
                'commission': commission,
                'slippage': slippage
            }
            
        except Exception as e:
            self.logger.error(f"Paper trade execution failed: {e}")
            return {'error': str(e)}
    
    def update_paper_positions(self) -> Dict:
        """Update all open paper positions with current prices"""
        
        try:
            positions = self._get_open_paper_positions()
            updated_count = 0
            
            for position in positions:
                symbol = position.get('symbol')
                current_price = self._get_current_price(symbol)
                
                if current_price:
                    # Calculate unrealized P&L
                    entry_price = position.get('entry_price', 0)
                    quantity = position.get('quantity', 0)
                    unrealized_pnl = (current_price - entry_price) * quantity
                    unrealized_pnl_percent = (unrealized_pnl / position.get('position_value', 1)) * 100
                    
                    # Check stop loss and target
                    position_update = {
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_percent': unrealized_pnl_percent
                    }
                    
                    # Auto-close if stop loss hit
                    if current_price <= position.get('stop_loss', 0):
                        position_update.update({
                            'status': 'STOPPED_OUT',
                            'exit_time': datetime.now(),
                            'exit_price': current_price,
                            'realized_pnl': unrealized_pnl
                        })
                    
                    # Auto-close if target hit
                    elif current_price >= position.get('target_price', float('inf')):
                        position_update.update({
                            'status': 'TARGET_HIT',
                            'exit_time': datetime.now(),
                            'exit_price': current_price,
                            'realized_pnl': unrealized_pnl
                        })
                    
                    self._update_paper_position(position.get('id'), position_update)
                    updated_count += 1
            
            return {'positions_updated': updated_count}
            
        except Exception as e:
            self.logger.error(f"Position update failed: {e}")
            return {'error': str(e)}
    
    def get_paper_portfolio_summary(self) -> Dict:
        """Get paper trading portfolio summary"""
        
        try:
            positions = self._get_all_paper_positions()
            open_positions = [p for p in positions if p.get('status') == 'OPEN']
            closed_positions = [p for p in positions if p.get('status') in ['STOPPED_OUT', 'TARGET_HIT', 'CLOSED']]
            
            # Calculate totals
            total_invested = sum(p.get('position_value', 0) for p in open_positions)
            total_unrealized = sum(p.get('unrealized_pnl', 0) for p in open_positions)
            total_realized = sum(p.get('realized_pnl', 0) for p in closed_positions)
            
            # Win rate calculation
            winning_trades = len([p for p in closed_positions if p.get('realized_pnl', 0) > 0])
            total_trades = len(closed_positions)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            import config
            portfolio_value = config.PAPER_TRADING_INITIAL_CAPITAL + total_realized + total_unrealized
            total_return = ((portfolio_value / config.PAPER_TRADING_INITIAL_CAPITAL) - 1) * 100
            
            return {
                'portfolio_value': round(portfolio_value, 2),
                'total_invested': round(total_invested, 2),
                'unrealized_pnl': round(total_unrealized, 2),
                'realized_pnl': round(total_realized, 2),
                'total_return_percent': round(total_return, 2),
                'open_positions': len(open_positions),
                'total_trades': total_trades,
                'win_rate': round(win_rate, 1),
                'available_cash': round(config.PAPER_TRADING_INITIAL_CAPITAL - total_invested + total_realized, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary failed: {e}")
            return {'error': str(e)}
    
    def _store_paper_position(self, position_data: Dict) -> str:
        """Store paper position in database"""
        try:
            query = """
                INSERT INTO agent_portfolio_positions 
                (symbol, signal_id, entry_time, entry_price, quantity, position_value, 
                 current_stop_loss, target_price, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        position_data.get('symbol'),
                        position_data.get('signal_id'),
                        position_data.get('entry_time'),
                        position_data.get('entry_price'),
                        position_data.get('quantity'),
                        position_data.get('position_value'),
                        position_data.get('stop_loss'),
                        position_data.get('target_price'),
                        'OPEN'
                    ))
                    position_id = cursor.fetchone()[0]
                    return str(position_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to store paper position: {e}")
            return None
    
    def _get_open_paper_positions(self) -> List[Dict]:
        """Get all open paper positions"""
        try:
            query = "SELECT * FROM agent_portfolio_positions WHERE status = 'OPEN'"
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            return []
    
    def _update_paper_position(self, position_id: str, updates: Dict):
        """Update paper position with current data"""
        try:
            set_clause = ", ".join([f"{k} = %s" for k in updates.keys()])
            query = f"UPDATE agent_portfolio_positions SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, list(updates.values()) + [position_id])
                    
        except Exception as e:
            self.logger.error(f"Failed to update position: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (simulated)"""
        try:
            # Get latest price from historical data
            query = """
                SELECT close FROM historical_data_3m_2025_q3 
                WHERE symbol = %s ORDER BY timestamp DESC LIMIT 1
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (symbol,))
                    result = cursor.fetchone()
                    return float(result[0]) if result else None
                    
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    # Add these methods to the existing PortfolioAgent class in agents/portfolio_agent.py
# Add them at the END of the class, before the closing of the class definition

    def _get_kite_connection(self):
        """Get authenticated Kite connection with automatic token management"""
        
        try:
            import config
            from kiteconnect import KiteConnect
            import json
            import os
            from datetime import datetime
            
            if not config.KITE_API_KEY or not config.KITE_API_SECRET:
                return None
            
            # Check if token exists and is valid for today
            if os.path.exists(config.KITE_TOKEN_FILE):
                try:
                    with open(config.KITE_TOKEN_FILE, "r") as f:
                        token_data = json.loads(f.read().strip())
                        today = datetime.now().strftime("%Y-%m-%d")
                        
                        if token_data.get("date") == today and token_data.get("access_token"):
                            access_token = token_data["access_token"]
                            
                            # Test the token
                            kite = KiteConnect(api_key=config.KITE_API_KEY)
                            kite.set_access_token(access_token)
                            
                            try:
                                profile = kite.profile()
                                self.logger.info(f"Kite connected as: {profile.get('user_name', 'Unknown')}")
                                return kite
                            except Exception:
                                # Token invalid, remove file
                                os.remove(config.KITE_TOKEN_FILE)
                except Exception:
                    # File corrupted, remove it
                    if os.path.exists(config.KITE_TOKEN_FILE):
                        os.remove(config.KITE_TOKEN_FILE)
            
            # Need to generate new token
            self.logger.warning("Kite token missing or expired - manual token generation required")
            self.logger.info("Run: python kite_token_generator.py to generate new token")
            return None
                
        except Exception as e:
            self.logger.error(f"Kite connection failed: {e}")
            return None

    def execute_live_trade(self, signal: Dict) -> Dict:
        """Execute live trade with trade mode control"""
        
        try:
            import config
            
            # Check trade mode first
            if not config.TRADE_MODE:
                self.logger.info(f"TRADE_MODE=no: Signal generated for {signal.get('symbol')} but no order placed")
                return {
                    'status': 'signal_only',
                    'reason': 'TRADE_MODE is disabled',
                    'signal_confidence': signal.get('overall_confidence', 0),
                    'would_execute': 'yes' if self._would_execute_live_trade(signal) else 'no'
                }
            
            # Original live trading logic continues only if TRADE_MODE = yes
            if not config.LIVE_TRADING_MODE:
                return self.execute_paper_trade(signal)
            
            if not self._is_market_open():
                return {'error': 'Market is closed'}
            
            # Initialize Kite connection
            kite = self._get_kite_connection()
            if not kite:
                self.logger.warning("Kite connection failed, falling back to paper trading")
                return self.execute_paper_trade(signal)
            
            # Validate live trading conditions
            validation_result = self._validate_live_trade_conditions(signal)
            if 'error' in validation_result:
                return validation_result
            
            # Place live order
            order_result = self._place_live_order(kite, signal)
            
            if 'error' in order_result:
                self.logger.error(f"Live order failed: {order_result['error']}")
                return self.execute_paper_trade(signal)
            
            # Store live position
            position_id = self._store_live_position(signal, order_result)
            
            self.logger.info(f"Live trade executed: {signal.get('symbol')} @ {order_result.get('price', 0)}")
            
            return {
                'status': 'live_executed',
                'position_id': position_id,
                'order_id': order_result.get('order_id'),
                'execution_price': order_result.get('price'),
                'execution_type': 'LIVE'
            }
            
        except Exception as e:
            self.logger.error(f"Live trade execution failed: {e}")
            return self.execute_paper_trade(signal)

    def _would_execute_live_trade(self, signal: Dict) -> bool:
        """Check if signal would be executed in live trading"""
        
        try:
            import config
            
            if not config.LIVE_TRADING_MODE:
                return False
            
            if not self._is_market_open():
                return False
            
            if signal.get('overall_confidence', 0) < config.LIVE_MIN_CONFIDENCE_THRESHOLD:
                return False
            
            symbol = signal.get('symbol')
            if symbol not in config.LIVE_TRADING_APPROVED_SYMBOLS:
                return False
            
            live_positions = self._get_live_positions()
            if len(live_positions) >= config.LIVE_MAX_POSITIONS:
                return False
            
            if any(pos.get('symbol') == symbol for pos in live_positions):
                return False
            
            return True
            
        except Exception:
            return False

    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        
        try:
            import config
            from datetime import datetime
            import pytz
            
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            
            # Skip weekends
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            market_open = now.replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0)
            market_close = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0)
            
            return market_open <= now <= market_close
            
        except Exception:
            return False

    def _place_live_order(self, kite, signal: Dict) -> Dict:
        """Place live order through Kite API"""
        
        try:
            import config
            
            order_params = {
                'exchange': 'NSE',
                'tradingsymbol': signal.get('symbol'),
                'transaction_type': 'BUY',
                'quantity': signal.get('recommended_shares', 0),
                'product': 'MIS',  # Intraday
                'order_type': 'LIMIT',
                'price': signal.get('entry_price'),
                'validity': 'DAY',
                'tag': 'nexus_trading'
            }
            
            order_id = kite.place_order(**order_params)
            
            # Wait for order confirmation
            import time
            time.sleep(config.LIVE_EXECUTION_DELAY_SECONDS)
            
            # Check order status
            order_status = kite.order_history(order_id)[-1]
            
            if order_status['status'] == 'COMPLETE':
                return {
                    'order_id': order_id,
                    'price': float(order_status['average_price']),
                    'quantity': int(order_status['filled_quantity']),
                    'status': 'executed'
                }
            else:
                return {'error': f"Order not filled: {order_status['status']}"}
                
        except Exception as e:
            self.logger.error(f"Live order placement failed: {e}")
            return {'error': str(e)}

    def _validate_live_trade_conditions(self, signal: Dict) -> Dict:
        """Enhanced validation with approved symbols check"""
        
        try:
            import config
            
            # Check confidence threshold
            if signal.get('overall_confidence', 0) < config.LIVE_MIN_CONFIDENCE_THRESHOLD:
                return {'error': 'Confidence too low for live trading'}
            
            # Check approved symbols
            symbol = signal.get('symbol')
            if symbol not in config.LIVE_TRADING_APPROVED_SYMBOLS:
                return {'error': f'Symbol {symbol} not approved for live trading'}
            
            # Check position limits
            live_positions = self._get_live_positions()
            open_positions = [p for p in live_positions if p.get('status') == 'OPEN']
            
            if len(open_positions) >= config.LIVE_MAX_POSITIONS:
                return {'error': 'Maximum live positions reached'}
            
            # Check symbol duplication
            if any(pos.get('symbol') == symbol for pos in open_positions):
                return {'error': f'Already have position in {symbol}'}
            
            # Check daily loss limit
            daily_pnl = sum(p.get('unrealized_pnl', 0) for p in open_positions)
            if abs(min(0, daily_pnl)) >= config.LIVE_MAX_LOSS_PER_DAY:
                return {'error': 'Daily loss limit reached'}
            
            return {'status': 'validated'}
            
        except Exception as e:
            return {'error': f'Validation failed: {e}'}

    def _store_live_position(self, signal: Dict, order_result: Dict) -> str:
        """Store live position in database"""
        
        try:
            from datetime import datetime
            
            query = """
                INSERT INTO agent_portfolio_positions 
                (symbol, signal_id, entry_time, entry_price, quantity, position_value, 
                 current_stop_loss, target_price, status, execution_type, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        signal.get('symbol'),
                        signal.get('id'),
                        datetime.now(),
                        order_result.get('price'),
                        order_result.get('quantity'),
                        order_result.get('price', 0) * order_result.get('quantity', 0),
                        signal.get('stop_loss'),
                        signal.get('target_price'),
                        'OPEN',
                        'LIVE',
                        order_result.get('order_id')
                    ))
                    position_id = cursor.fetchone()[0]
                    return str(position_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to store live position: {e}")
            return None

    def _get_live_positions(self) -> list:
        """Get live positions from database"""
        
        try:
            query = """
                SELECT * FROM agent_portfolio_positions 
                WHERE execution_type = 'LIVE' AND status = 'OPEN'
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            self.logger.error(f"Failed to get live positions: {e}")
            return []