#!/usr/bin/env python3
"""
FILE: kite_token_generator.py
PURPOSE: Generate and manage Kite API tokens

DESCRIPTION:
- Generates daily Kite API tokens
- Saves tokens with date validation
- Provides authenticated Kite client

USAGE:
- python kite_token_generator.py
"""

from kiteconnect import KiteConnect
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

KITE_API_KEY = os.getenv('KITE_API_KEY')
KITE_API_SECRET = os.getenv('KITE_API_SECRET')
TOKEN_FILE = "kite_token.txt"

def first_run():
    """Generate and save Kite Connect access token"""
    
    if not KITE_API_KEY or not KITE_API_SECRET:
        print("Error: KITE_API_KEY and KITE_API_SECRET must be set in .env file")
        return None
    
    # Check if token exists and is valid for today
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                token_data = json.loads(f.read().strip())
                today = datetime.now().strftime("%Y-%m-%d")
                
                if token_data.get("date") == today and token_data.get("access_token"):
                    print(f"Valid token found for {today}")
                    return token_data["access_token"]
                else:
                    os.remove(TOKEN_FILE)  # Delete old token
        except:
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
    
    # Generate new token
    print("Generating new Kite API token...")
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url()
    
    print("\nSTEPS TO GENERATE TOKEN:")
    print("1. Open this URL in browser:")
    print(f"   {login_url}")
    print("2. Login with your Zerodha credentials")
    print("3. Copy the request_token from the redirected URL")
    print("4. Paste it below\n")
    
    request_token = input("Paste request_token: ").strip()
    
    if not request_token:
        print("Error: No request token provided")
        return None
    
    try:
        session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = session["access_token"]
        
        # Save with today's date
        token_data = {
            "access_token": access_token, 
            "date": datetime.now().strftime("%Y-%m-%d"),
            "user_id": session.get("user_id", "unknown")
        }
        
        with open(TOKEN_FILE, "w") as f:
            f.write(json.dumps(token_data, indent=2))
        
        print(f"✓ Token generated and saved for {token_data['date']}")
        print(f"✓ User: {token_data['user_id']}")
        
        return access_token
        
    except Exception as e:
        print(f"Failed to generate session: {e}")
        return None

def get_authenticated_kite_client():
    """Get authenticated Kite client ready for use"""
    
    token = first_run()
    if not token:
        print("Failed to get access token")
        return None
    
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(token)
    
    try:
        profile = kite.profile()
        print(f"✓ Connected as: {profile['user_name']} ({profile['user_id']})")
        return kite
    except Exception as e:
        print(f"Connection test failed: {e}")
        return None

def main():
    """Main token generation function"""
    
    print("Kite API Token Generator")
    print("=" * 40)
    
    token = first_run()
    if token:
        print("\n✅ Token ready for live trading")
        print("\nNext steps:")
        print("1. Set TRADE_MODE=yes in .env to enable trading")
        print("2. Run: python live_trading_manager.py --mode validate")
        print("3. Run: python main.py --mode live")
    else:
        print("\n❌ Token generation failed")
        print("\nCheck:")
        print("1. KITE_API_KEY and KITE_API_SECRET in .env")
        print("2. Valid Zerodha account with API access")
        print("3. Correct request_token from login URL")

if __name__ == "__main__":
    main()