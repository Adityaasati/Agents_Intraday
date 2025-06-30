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
from datetime import datetime
import pytz

import config

class PortfolioAgent:
    """Portfolio management and tracking agent"""
    
    def __init__(self, db_manager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.ist = pytz.timezone('Asia/Kolkata')
    
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