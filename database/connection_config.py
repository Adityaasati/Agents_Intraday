"""Centralized database connection configuration with .env support"""
import os
from typing import Dict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Look for .env file in current directory and parent directories
    env_path = Path('.env')
    if not env_path.exists():
        # Try parent directory
        env_path = Path('../.env')
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from: {env_path.absolute()}")
    else:
        print("⚠️ No .env file found. Using environment variables only.")
        
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("Using environment variables only.")

def get_connection_params() -> Dict[str, str]:
    """Get database connection parameters from environment
    
    Returns:
        Dict containing host, port, database, user, password
    """
    params = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5435'),  # Changed default to 5435
        'database': os.getenv('DATABASE_NAME'),  # No default - must be set in .env
        'user': os.getenv('DATABASE_USER'),  # No default - must be set in .env
        'password': os.getenv('DATABASE_PASSWORD', '')
    }
    
    # Debug print (remove in production)
    print("Database Connection Parameters:")
    for key, value in params.items():
        if key == 'password':
            print(f"  {key}: {'*' * len(value) if value else 'NOT SET'}")
        else:
            print(f"  {key}: {value}")
    
    return params

def get_pool_config() -> Dict[str, int]:
    """Get connection pool configuration
    
    Returns:
        Dict containing pool size and timeout settings
    """
    return {
        'minconn': int(os.getenv('DB_POOL_MIN', '5')),
        'maxconn': int(os.getenv('DB_POOL_MAX', '20')),
        'timeout': int(os.getenv('DB_POOL_TIMEOUT', '30'))
    }

def get_connection_string() -> str:
    """Get full PostgreSQL connection string"""
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

def validate_connection_config() -> bool:
    """Validate that all required connection parameters are set"""
    params = get_connection_params()
    
    required_params = ['host', 'port', 'database', 'user', 'password']
    missing_params = []
    
    for param in required_params:
        if not params.get(param):
            missing_params.append(param)
    
    if missing_params:
        print(f"❌ Missing required database parameters: {missing_params}")
        print("Please set these in your .env file or environment variables:")
        for param in missing_params:
            env_var = f"DATABASE_{param.upper()}"
            print(f"  {env_var}=your_value")
        return False
    
    print("✅ All database connection parameters are set")
    return True

def test_env_loading():
    """Test if .env file is properly loaded"""
    print("Testing .env file loading...")
    
    # Check if .env file exists
    env_files = [Path('.env'), Path('../.env')]
    env_found = False
    
    for env_path in env_files:
        if env_path.exists():
            print(f"✅ Found .env file: {env_path.absolute()}")
            env_found = True
            
            # Read and display content (without passwords)
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            print("Contents:")
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if 'PASSWORD' in line:
                        key = line.split('=')[0]
                        print(f"  {key}=*****")
                    else:
                        print(f"  {line}")
            break
    
    if not env_found:
        print("❌ No .env file found in current or parent directory")
        print("Create a .env file with your database configuration:")
        print("""
DATABASE_HOST=localhost
DATABASE_PORT=5435
DATABASE_NAME=your_database_name
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password
""")
    
    return env_found

if __name__ == "__main__":
    test_env_loading()
    print_config_summary()
    validate_connection_config()