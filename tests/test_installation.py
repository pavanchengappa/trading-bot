#!/usr/bin/env python3
"""
Test script to verify trading bot installation
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        "crypto_trading_bot.config.settings",
        "crypto_trading_bot.core.bot",
        "crypto_trading_bot.core.strategies",
        "crypto_trading_bot.core.risk_manager",
        "crypto_trading_bot.database.models",
        "crypto_trading_bot.notifications.notifier",
        "crypto_trading_bot.ui.cli",
        "crypto_trading_bot.ui.gui",
        "crypto_trading_bot.backtesting.backtest_engine",
        "crypto_trading_bot.utils.logger"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_dependencies():
    """Test if external dependencies are available"""
    print("\n📦 Testing external dependencies...")
    
    dependencies = [
        ("binance", "python-binance"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("click", "click"),
        ("apscheduler", "APScheduler"),
        ("cryptography", "cryptography"),
        ("requests", "requests")
    ]
    
    failed_deps = []
    
    for module, package in dependencies:
        try:
            importlib.import_module(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed_deps.append(package)
    
    return len(failed_deps) == 0

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing configuration system...")
    
    try:
        from crypto_trading_bot.config.settings import Settings
        
        # Test creating settings
        settings = Settings()
        print("✓ Settings created successfully")
        
        # Test configuration validation
        if settings.validate_config():
            print("✓ Configuration validation passed")
        else:
            print("⚠ Configuration validation failed (expected for empty config)")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n🗄️ Testing database system...")
    
    try:
        from crypto_trading_bot.database.models import DatabaseManager
        
        # Test database initialization
        db_manager = DatabaseManager("test.db")
        print("✓ Database manager created")
        
        # Test database stats
        stats = db_manager.get_database_stats()
        print(f"✓ Database stats retrieved: {stats}")
        
        # Clean up test database
        Path("test.db").unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_strategies():
    """Test trading strategies"""
    print("\n📊 Testing trading strategies...")
    
    try:
        from crypto_trading_bot.core.strategies import StrategyFactory
        
        factory = StrategyFactory()
        strategies = factory.get_available_strategies()
        print(f"✓ Available strategies: {strategies}")
        
        # Test creating a strategy
        strategy = factory.create_strategy("moving_average_crossover", {
            "short_window": 10,
            "long_window": 30
        })
        print("✓ Strategy created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy test failed: {e}")
        return False

def test_notifications():
    """Test notification system"""
    print("\n🔔 Testing notification system...")
    
    try:
        from crypto_trading_bot.notifications.notifier import Notifier
        from crypto_trading_bot.config.settings import NotificationConfig
        
        config = NotificationConfig()
        notifier = Notifier(config)
        
        status = notifier.get_notification_status()
        print(f"✓ Notification system initialized: {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Notification test failed: {e}")
        return False

def test_logging():
    """Test logging system"""
    print("\n📝 Testing logging system...")
    
    try:
        from crypto_trading_bot.utils.logger import setup_logging, get_logger
        
        # Setup logging
        setup_logging()
        print("✓ Logging system initialized")
        
        # Test logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✓ Logger test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Cryptocurrency Trading Bot Installation")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Strategies", test_strategies),
        ("Notifications", test_notifications),
        ("Logging", test_logging)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The trading bot is ready to use.")
        print("\nNext steps:")
        print("1. Run setup: python setup.py")
        print("2. Configure API keys: python crypto_trading_bot/main.py --mode config")
        print("3. Test with backtesting: python crypto_trading_bot/main.py --mode backtest")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (requires 3.8+)")
        print("3. Verify all files are in the correct locations")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 