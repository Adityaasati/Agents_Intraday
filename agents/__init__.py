"""
Nexus Trading System - Agents Package
Technical Analysis and Trading Signal Generation
"""

from .technical_agent import TechnicalAgent
from .fundamental_agent import FundamentalAgent
from .signal_agent import SignalAgent
from .risk_agent import RiskAgent
from .portfolio_agent import PortfolioAgent

__version__ = "1.0.0"
__all__ = ['TechnicalAgent', 'FundamentalAgent', 'SignalAgent', 'RiskAgent', 'PortfolioAgent']

# Package metadata
PACKAGE_INFO = {
    'name': 'nexus_trading_agents',
    'version': __version__,
    'description': 'AI trading agents for Day 1/2 implementation',
    'agents': {
        'TechnicalAgent': {
            'description': 'Technical analysis and indicator calculations',
            'status': 'implemented',
            'day': 1
        },
        'FundamentalAgent': {
            'description': 'Fundamental analysis using 30-column data',
            'status': 'implemented',
            'day': 2
        },
        'SignalAgent': {
            'description': 'Master signal coordinator',
            'status': 'implemented', 
            'day': 2
        },
        'RiskAgent': {
            'description': 'Risk management and position sizing',
            'status': 'implemented',
            'day': 1
        },
        'PortfolioAgent': {
            'description': 'Portfolio tracking and management',
            'status': 'implemented',
            'day': 2
        }
    }
}

def get_package_info():
    """Get package information"""
    return PACKAGE_INFO

def list_available_agents():
    """List all available agents"""
    return list(PACKAGE_INFO['agents'].keys())

def get_agent_capabilities(agent_name: str):
    """Get capabilities of a specific agent"""
    return PACKAGE_INFO['agents'].get(agent_name, {})