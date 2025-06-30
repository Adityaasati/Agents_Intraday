"""
FILE: agents/__init__.py
LOCATION: /agents/ directory
PURPOSE: Agents Package Initialization - Makes agents directory a Python package

DESCRIPTION:
- Initializes the agents package for import
- Exports TechnicalAgent class for easy importing
- Contains package metadata and information about available agents
- Provides utility functions to list agent capabilities
- Part of the modular agent architecture for trading system

DEPENDENCIES:
- agents/technical_agent.py (imports TechnicalAgent)

USAGE:
- Allows importing: from agents import TechnicalAgent
- Provides package information and agent capabilities
- Used by main system to discover available agents
"""

"""
Nexus Trading System - Agents Package
Technical Analysis and Trading Signal Generation
"""

from .technical_agent import TechnicalAgent

__version__ = "1.0.0"
__all__ = ['TechnicalAgent']

# Package metadata
PACKAGE_INFO = {
    'name': 'nexus_trading_agents',
    'version': __version__,
    'description': 'AI trading agents for technical analysis and signal generation',
    'agents': {
        'TechnicalAgent': {
            'description': 'Technical analysis and signal generation',
            'capabilities': [
                'RSI calculation and analysis',
                'MACD signal detection',
                'Bollinger Bands analysis',
                'Moving average trend detection',
                'Volume analysis',
                'Support/resistance identification',
                'Buy/sell signal generation'
            ],
            'requirements': ['pandas', 'numpy', 'pandas_ta (optional)']
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
    return PACKAGE_INFO['agents'].get(agent_name, {}).get('capabilities', [])