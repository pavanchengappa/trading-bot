#!/usr/bin/env python3
"""
Setup script for Cryptocurrency Trading Bot
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_config_template():
    """Create a template configuration file"""
    config_template = """{
  "api_config": {
    "api_key": "",
    "api_secret": "",
    "testnet": true,
    "base_url": "https://api.binance.com"
  },
  "trading_config": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "investment_amount": 100.0,
    "max_daily_loss": 50.0,
    "stop_loss_percentage": 5.0,
    "take_profit_percentage": 10.0,
    "max_drawdown": 20.0
  },
  "strategy_config": {
    "name": "moving_average_crossover",
    "parameters": {
      "short_window": 10,
      "long_window": 30,
      "rsi_period": 14,
      "rsi_overbought": 70,
      "rsi_oversold": 30
    },
    "enabled": true
  },
  "notification_config": {
    "email_enabled": false,
    "email_address": "",
    "telegram_enabled": false,
    "telegram_bot_token": "",
    "telegram_chat_id": ""
  },
  "database_config": {
    "path": "trading_bot.db",
    "backup_enabled": true,
    "backup_interval_hours": 24
  }
}"""
    
    config_file = Path("config.json")
    if not config_file.exists():
        with open(config_file, "w") as f:
            f.write(config_template)
        print("✓ Created config.json template")
    else:
        print("✓ config.json already exists")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "binance",
        "pandas",
        "numpy",
        "click",
        "APScheduler",
        "cryptography",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Cryptocurrency Trading Bot")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create config template
    print("\n⚙️ Creating configuration template...")
    create_config_template()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    if deps_ok:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Configure your API keys: python crypto_trading_bot/main.py --mode config")
        print("2. Test the bot: python crypto_trading_bot/main.py --mode backtest")
        print("3. Start trading: python crypto_trading_bot/main.py --mode trade")
    else:
        print("⚠️ Setup completed with warnings.")
        print("Please install missing dependencies before using the bot.")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 