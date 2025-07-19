# test_binance_data.py - Test script for Binance data fetching
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.binance_data import BinanceDataFetcher, fetch_historical_data_for_backtest
from config.settings import Settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_binance_data_fetching():
    """Test Binance data fetching functionality"""
    print("=== Testing Binance Data Fetching ===\n")
    
    try:
        # Initialize settings
        settings = Settings()
        
        # Test with public data (no API keys required)
        print("1. Testing with public data (no API keys)...")
        fetcher = BinanceDataFetcher()
        
        # Test symbol validation
        print("   Testing symbol validation...")
        is_valid = fetcher.validate_symbol("BTCUSDT")
        print(f"   BTCUSDT is valid: {is_valid}")
        
        # Test recent klines
        print("   Testing recent klines...")
        recent_data = fetcher.get_recent_klines("BTCUSDT", limit=10)
        if not recent_data.empty:
            print(f"   ✓ Successfully fetched {len(recent_data)} recent klines")
            print(f"   Sample data:")
            print(recent_data.head(3))
        else:
            print("   ✗ Failed to fetch recent klines")
        
        # Test historical data (last 7 days)
        print("\n2. Testing historical data fetching...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        historical_data = fetch_historical_data_for_backtest(
            symbols=["BTCUSDT"],
            start_date=start_date,
            end_date=end_date,
            interval="1h"
        )
        
        if historical_data and "BTCUSDT" in historical_data:
            data = historical_data["BTCUSDT"]
            print(f"   ✓ Successfully fetched {len(data)} historical klines")
            print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
            print(f"   Sample data:")
            print(data.head(3))
        else:
            print("   ✗ Failed to fetch historical data")
        
        # Test with API credentials if available
        print("\n3. Testing with API credentials...")
        api_key = settings.get_api_key()
        api_secret = settings.get_api_secret()
        
        if api_key and api_secret:
            print("   API credentials found, testing authenticated access...")
            auth_fetcher = BinanceDataFetcher(api_key, api_secret)
            
            # Test account info (requires API keys)
            try:
                account_info = auth_fetcher.client.get_account()
                print(f"   ✓ Successfully authenticated with Binance")
                print(f"   Account status: {account_info.get('status', 'Unknown')}")
            except Exception as e:
                print(f"   ⚠ Authentication test failed: {e}")
        else:
            print("   No API credentials found, skipping authenticated tests")
        
        print("\n=== Test completed ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"✗ Test failed: {e}")

def test_backtest_configuration():
    """Test backtest configuration"""
    print("\n=== Testing Backtest Configuration ===\n")
    
    try:
        settings = Settings()
        
        # Show current backtest config
        backtest_config = settings.get_backtest_config()
        print("Current backtest configuration:")
        print(f"  Use Real Data: {backtest_config.get('use_real_data', False)}")
        print(f"  Data Interval: {backtest_config.get('data_interval', '1h')}")
        print(f"  Fallback to Synthetic: {backtest_config.get('fallback_to_synthetic', True)}")
        
        # Test configuration update
        print("\nTesting configuration update...")
        settings.update_config("backtest_config", "use_real_data", True)
        settings.update_config("backtest_config", "data_interval", "4h")
        
        updated_config = settings.get_backtest_config()
        print("Updated backtest configuration:")
        print(f"  Use Real Data: {updated_config.get('use_real_data', False)}")
        print(f"  Data Interval: {updated_config.get('data_interval', '1h')}")
        
        print("✓ Configuration test completed")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        print(f"✗ Configuration test failed: {e}")

if __name__ == "__main__":
    test_binance_data_fetching()
    test_backtest_configuration() 