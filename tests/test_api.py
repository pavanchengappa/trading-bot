from binance.client import Client
import os
     
     # Replace with your actual API Key and Secret
API_KEY = "mx2aa7I4hMhg77D0dDBLihnZBsRm6iYVDp5bmyHBg47dy1wnJoZPLXfp4O8EKHwY"
API_SECRET = "Sjq68XkbOJBcjodq2GYQ1rwNNnIcXhwCP19Y9o6IZrrVIKx7sA22uHsta7oq59GV"
     
     # Set to True if using Binance Testnet keys, False for Live keys
TESTNET = True
    
try:
        if TESTNET:
            client = Client(API_KEY, API_SECRET, tld='us', testnet=True)
            print("Attempting to connect to Binance Testnet...")
        else:
            client = Client(API_KEY, API_SECRET)
            print("Attempting to connect to Binance Live...")
   
        # Try to get account information (requires a signed request)
        info = client.get_account()
        print("\nSuccessfully connected to Binance API!")
        print("Account Info (partial):")
        print(f"  Can Trade: {info['canTrade']}")
        print(f"  Balances: {info['balances'][:3]}...") # Show first 3 balances
   
except Exception as e:
        print(f"\nError connecting to Binance API: {e}")
        if "APIError(code=-1022): Signature for this request is invalid." in str(e):
            print("\nThis confirms the -1022 error. Please re-check your API key, secret, permissions, and system time.")
        elif "APIError(code=-2015): Invalid API-key, IP, or permissions for action." in str(e):
            print("\nThis indicates an issue with the API key itself, IP restrictions, or permissions.")
        else:
            print("\nAn unexpected error occurred. Check your internet connection or other factors.")