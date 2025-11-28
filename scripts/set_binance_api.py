#!/usr/bin/env python3
"""
Script to set and encrypt Binance API key and secret for the trading bot.
"""
from config.settings import Settings

def main():
    print("=== Set Binance API Credentials ===")
    api_key = input("Enter your Binance API Key: ").strip()
    api_secret = input("Enter your Binance API Secret: ").strip()
    if not api_key or not api_secret:
        print("API key and secret cannot be empty.")
        return 1
    settings = Settings()
    # Ensure binance_config section exists and is a dict
    if 'binance_config' not in settings.config or not isinstance(settings.config['binance_config'], dict):
        settings.config['binance_config'] = {"api_key": "", "api_secret": "", "testnet": True}
    settings.save_config()
    settings.set_api_credentials(api_key, api_secret)
    print("API credentials saved and encrypted successfully!")
    return 0

if __name__ == "__main__":
    main() 