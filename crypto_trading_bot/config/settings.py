# config/settings.py - Configuration management for the trading bot
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Binance API configuration"""
    api_key: str
    api_secret: str
    testnet: bool = False
    base_url: str = "https://api.binance.com"

@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    symbols: List[str]  # e.g., ['BTCUSDT', 'ETHUSDT']
    investment_amount: float  # Amount per trade
    max_daily_loss: float  # Maximum daily loss limit
    stop_loss_percentage: float  # Stop loss percentage
    take_profit_percentage: float  # Take profit percentage
    max_drawdown: float  # Maximum drawdown limit

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str  # Strategy name
    parameters: Dict  # Strategy-specific parameters
    enabled: bool = True

@dataclass
class NotificationConfig:
    """Notification settings"""
    email_enabled: bool = False
    email_address: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "trading_bot.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

class Settings:
    """Main settings class that manages all configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.encryption_key_path = "encryption.key"

        # Default configuration
        self.default_config = {
            "binance_config": {
                "api_key": "",
                "api_secret": "",
                "testnet": True
            },
            "trading_config": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "investment_amount": 100.0,
                "max_daily_loss": 0.05,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.10,
                "max_drawdown": 0.20
            },
            "strategy_config": {
                "name": "moving_average_crossover",
                "parameters": {
                    "short_window": 3,
                    "long_window": 10,
                    "min_crossover_strength": 0.00001
                }
            },
            "risk_config": {
                "max_position_size": 0.1,
                "max_daily_trades": 10,
                "max_daily_loss": 0.05
            },
            "notification_config": {
                "email_enabled": False,
                "email": "",
                "telegram_enabled": False,
                "telegram_bot_token": "",
                "telegram_chat_id": ""
            },
            "database_config": {
                "path": "trading_bot.db"
            },
            "backtest_config": {
                "use_real_data": False,
                "data_interval": "1h",
                "fallback_to_synthetic": True
            }
        }
        
        self.config = self.load_config()
        print("Loaded config:", self.config)  # DEBUG: Show loaded config
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info("Configuration loaded successfully")
                return config
            else:
                logger.info("No configuration file found, using defaults")
                return self.default_config.copy()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self.default_config.copy()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_binance_config(self) -> Dict[str, str]:
        """Get Binance API configuration"""
        return self.config.get("binance_config", {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.config.get("trading_config", {})
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.config.get("strategy_config", {})
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.config.get("risk_config", {})
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration"""
        return self.config.get("notification_config", {})
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration"""
        return self.config.get("database_config", {})
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        return self.config.get("backtest_config", {})
    
    def get_portfolio_config(self):
        return self.config.get("portfolio_config", {})
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
    
    def get_api_key(self) -> Optional[str]:
        """Get decrypted API key"""
        try:
            encrypted_key = self.config.get("binance_config", {}).get("api_key", "")
            if not encrypted_key:
                return None
            
            key = self._load_encryption_key()
            if not key:
                return None
            
            fernet = Fernet(key)
            return fernet.decrypt(encrypted_key.encode()).decode()
        except Exception as e:
            logger.error(f"Error decrypting API key: {e}")
            return None
    
    def get_api_secret(self) -> Optional[str]:
        """Get decrypted API secret"""
        try:
            encrypted_secret = self.config.get("binance_config", {}).get("api_secret", "")
            if not encrypted_secret:
                return None
            
            key = self._load_encryption_key()
            if not key:
                return None
            
            fernet = Fernet(key)
            return fernet.decrypt(encrypted_secret.encode()).decode()
        except Exception as e:
            logger.error(f"Error decrypting API secret: {e}")
            return None
    
    def set_api_credentials(self, api_key: str, api_secret: str):
        """Set and encrypt API credentials"""
        try:
            key = self._load_or_create_encryption_key()
            fernet = Fernet(key)
            
            encrypted_key = fernet.encrypt(api_key.encode()).decode()
            encrypted_secret = fernet.encrypt(api_secret.encode()).decode()
            
            self.config["binance_config"]["api_key"] = encrypted_key
            self.config["binance_config"]["api_secret"] = encrypted_secret
            self.save_config()
            
            logger.info("API credentials encrypted and saved")
        except Exception as e:
            logger.error(f"Error encrypting API credentials: {e}")
    
    def _load_or_create_encryption_key(self) -> bytes:
        """Load existing encryption key or create a new one"""
        if os.path.exists(self.encryption_key_path):
            with open(self.encryption_key_path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.encryption_key_path, 'wb') as f:
                f.write(key)
            return key
    
    def _load_encryption_key(self) -> Optional[bytes]:
        """Load encryption key"""
        try:
            if os.path.exists(self.encryption_key_path):
                with open(self.encryption_key_path, 'rb') as f:
                    return f.read()
            return None
        except Exception as e:
            logger.error(f"Error loading encryption key: {e}")
            return None
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate API configuration
        if not self.get_api_key() or not self.get_api_secret():
            errors.append("API key and secret are required")
        
        # Validate trading configuration
        if self.get_trading_config().get("investment_amount", 0) <= 0:
            errors.append("Investment amount must be positive")
        
        if not self.get_trading_config().get("symbols", []):
            errors.append("At least one trading symbol is required")
        
        if self.get_trading_config().get("max_daily_loss", 0) <= 0:
            errors.append("Maximum daily loss must be positive")
        
        # Validate strategy configuration
        if not self.get_strategy_config().get("name"):
            errors.append("Strategy name is required")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_binance_url(self) -> str:
        """Get appropriate Binance URL based on testnet setting"""
        if self.get_binance_config().get('testnet', True):
            return "https://testnet.binance.vision"
        return self.get_binance_config().get('base_url', 'https://api.binance.com') 