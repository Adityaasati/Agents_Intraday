"""Centralized database connection configuration"""
import os
from typing import Dict

def get_connection_params() -> Dict[str, str]:
    """Get database connection parameters from environment
    
    Returns:
        Dict containing host, port, database, user, password
    """
    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5432'),
        'database': os.getenv('DATABASE_NAME'),
        'user': os.getenv('DATABASE_USER'),
        'password': os.getenv('DATABASE_PASSWORD')
    }

def get_pool_config() -> Dict[str, int]:
    """Get connection pool configuration optimized for concurrent usage
    
    Returns:
        Dict containing pool size and timeout settings
    """
    # Determine pool size based on environment or defaults
    # For trading system with multiple concurrent agents, need larger pool
    
    # Base pool size on expected concurrent operations:
    # - TechnicalAgent: 2-3 connections
    # - SentimentAgent: 2-3 connections  
    # - HistoricalDataAgent: 1-2 connections
    # - Main system operations: 2-3 connections
    # - Test execution: 5-10 connections
    # - Buffer for spikes: 5-10 additional
    
    return {
        'minconn': int(os.getenv('DB_POOL_MIN', '8')),     # Increased from 5
        'maxconn': int(os.getenv('DB_POOL_MAX', '25')),    # Increased from 20
        'timeout': int(os.getenv('DB_TIMEOUT', '30'))      # Connection timeout
    }
    
def get_performance_config() -> Dict[str, int]:
    """Get performance-related database configuration"""
    return {
        'query_timeout': int(os.getenv('DB_QUERY_TIMEOUT', '60')),
        'connection_retry_count': int(os.getenv('DB_RETRY_COUNT', '3')),
        'retry_delay': float(os.getenv('DB_RETRY_DELAY', '0.5')),
        'pool_exhaustion_retry_timeout': int(os.getenv('DB_POOL_RETRY_TIMEOUT', '30'))
    }

def print_config_summary():
    """Print current database configuration for debugging"""
    params = get_connection_params()
    pool_config = get_pool_config()
    
    print("=== Database Configuration Summary ===")
    print(f"Host: {params['host']}:{params['port']}")
    print(f"Database: {params['database']}")
    print(f"User: {params['user']}")
    print(f"Password: {'*' * len(params['password']) if params['password'] else 'NOT SET'}")
    print(f"Pool Size: {pool_config['minconn']}-{pool_config['maxconn']} connections")
    print(f"Timeout: {pool_config['timeout']} seconds")
    print("=" * 40)

def get_performance_config() -> Dict[str, int]:
    """Get performance-related database configuration"""
    return {
        'query_timeout': int(os.getenv('DB_QUERY_TIMEOUT', '60')),
        'connection_retry_count': int(os.getenv('DB_RETRY_COUNT', '3')),
        'retry_delay': float(os.getenv('DB_RETRY_DELAY', '0.5')),
        'max_connections_per_agent': int(os.getenv('DB_MAX_CONN_PER_AGENT', '5')),
        'pool_exhaustion_retry_timeout': int(os.getenv('DB_POOL_RETRY_TIMEOUT', '30'))
    }

def validate_connection_params() -> bool:
    """Validate that all required connection parameters are set
    
    Returns:
        True if all parameters are set, False otherwise
    """
    params = get_connection_params()
    required = ['database', 'user', 'password']
    
    missing_params = []
    for param in required:
        if not params.get(param):
            missing_params.append(param)
    
    if missing_params:
        print(f"Missing required database parameters: {', '.join(missing_params)}")
        print("Please set the following environment variables:")
        for param in missing_params:
            env_var = f"DATABASE_{param.upper()}"
            print(f"  {env_var}")
        return False
    
    return True

def get_connection_string() -> str:
    """Get PostgreSQL connection string"""
    params = get_connection_params()
    return f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"

def print_config_summary():
    """Print current database configuration for debugging"""
    params = get_connection_params()
    pool_config = get_pool_config()
    
    print("=== Database Configuration Summary ===")
    print(f"Host: {params['host']}:{params['port']}")
    print(f"Database: {params['database']}")
    print(f"User: {params['user']}")
    print(f"Password: {'*' * len(params['password']) if params['password'] else 'NOT SET'}")
    print(f"Pool Size: {pool_config['minconn']}-{pool_config['maxconn']} connections")
    print(f"Timeout: {pool_config['timeout']} seconds")
    print("=" * 40)

def get_optimal_pool_size_for_workload(concurrent_agents: int = 4, test_mode: bool = False) -> Dict[str, int]:
    """Calculate optimal pool size based on workload"""
    
    # Base calculation
    base_connections = concurrent_agents * 2  # 2 connections per agent average
    
    if test_mode:
        # Test mode needs more connections due to concurrent test execution
        base_connections += 10
    
    # Add buffer
    buffer = max(5, base_connections // 2)
    
    min_conn = max(5, base_connections // 2)
    max_conn = min(50, base_connections + buffer)  # Cap at 50
    
    return {
        'minconn': min_conn,
        'maxconn': max_conn,
        'timeout': 30
    }

def validate_pool_configuration() -> bool:
    """Validate pool configuration is reasonable"""
    pool_config = get_pool_config()
    
    min_conn = pool_config['minconn']
    max_conn = pool_config['maxconn']
    
    # Validation checks
    if min_conn <= 0:
        print("ERROR: Minimum connections must be > 0")
        return False
    
    if max_conn <= min_conn:
        print("ERROR: Maximum connections must be > minimum connections")
        return False
    
    if max_conn > 100:
        print("WARNING: Very large connection pool (>100). This may cause resource issues.")
        print("Consider reducing DB_POOL_MAX environment variable.")
    
    if min_conn < 3:
        print("WARNING: Very small minimum connection pool (<3). May cause connection contention.")
        print("Consider increasing DB_POOL_MIN environment variable.")
    
    return True