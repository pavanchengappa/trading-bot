#!/usr/bin/env python3
"""
Test script to diagnose and fix backtesting issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.backtesting.backtest_engine import BacktestEngine
from crypto_trading_bot.core.strategies import StrategyFactory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_different_strategies():
    """Test different strategies to see which ones generate signals"""
    settings = Settings()
    
    # Test strategies
    strategies_to_test = [
        {
            'name': 'random_strategy',
            'parameters': {'signal_probability': 0.05}  # 5% chance per check
        },
        {
            'name': 'moving_average_crossover',
            'parameters': {
                'short_window': 3,
                'long_window': 8,
                'min_crossover_strength': 0.00001
            }
        },
        {
            'name': 'rsi_strategy',
            'parameters': {
                'rsi_period': 5,
                'rsi_overbought': 75,
                'rsi_oversold': 25
            }
        },
        {
            'name': 'bollinger_bands',
            'parameters': {
                'bb_window': 10,
                'bb_std_dev': 1.5,
                'min_breakout_strength': 0.001
            }
        }
    ]
    
    print("="*60)
    print("TESTING DIFFERENT STRATEGIES FOR BACKTESTING")
    print("="*60)
    
    for strategy_config in strategies_to_test:
        print(f"\nTesting strategy: {strategy_config['name']}")
        print(f"Parameters: {strategy_config['parameters']}")
        
        # Update settings with this strategy
        settings.update_config("strategy_config", "name", strategy_config['name'])
        settings.update_config("strategy_config", "parameters", strategy_config['parameters'])
        
        # Create backtest engine
        engine = BacktestEngine(settings)
        
        # Run backtest for a short period
        results = engine.run(
            start_date="2024-01-01",
            end_date="2024-01-07",  # 1 week
            strategy_name=strategy_config['name']
        )
        
        if results:
            print(f"✅ Strategy generated {results.total_trades} trades")
            print(f"   Win rate: {results.win_rate:.2%}")
            print(f"   Total return: {results.total_return:.2%}")
        else:
            print("❌ Strategy generated no trades")
        
        print("-" * 40)

def test_strategy_directly():
    """Test strategy signal generation directly"""
    print("\n" + "="*60)
    print("TESTING STRATEGY SIGNAL GENERATION DIRECTLY")
    print("="*60)
    
    settings = Settings()
    strategy_factory = StrategyFactory()
    
    # Test with random strategy first (most likely to generate signals)
    strategy = strategy_factory.create_strategy('random_strategy', {'signal_probability': 0.1})
    
    # Create some dummy klines data
    import numpy as np
    klines = []
    base_price = 50000
    
    for i in range(100):
        price = base_price + np.random.normal(0, 100)
        timestamp_ms = int(1704067200000 + i * 3600000)  # 1 hour intervals
        
        kline = [
            timestamp_ms,  # Open time
            str(price),    # Open
            str(price * 1.01),  # High
            str(price * 0.99),  # Low
            str(price),    # Close
            str(1000),     # Volume
            timestamp_ms,  # Close time
            "0", "0", "0", "0"  # Other fields
        ]
        klines.append(kline)
    
    print(f"Testing with {len(klines)} klines")
    
    # Test signal generation
    signal_count = 0
    for i in range(10):
        signal = strategy.generate_signal('BTCUSDT', base_price, klines)
        if signal:
            signal_count += 1
            print(f"Signal {signal_count}: {signal.action} @ {signal.price}")
    
    print(f"Generated {signal_count} signals out of 10 attempts")

def main():
    """Main test function"""
    print("Backtesting Diagnosis and Fix Tool")
    print("This tool will help identify why your backtest shows 0 trades")
    
    # Test 1: Direct strategy testing
    test_strategy_directly()
    
    # Test 2: Full backtest with different strategies
    test_different_strategies()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. If random_strategy works, the backtest engine is functioning correctly")
    print("2. If other strategies don't work, try adjusting their parameters")
    print("3. For moving_average_crossover, try shorter windows and lower crossover strength")
    print("4. For RSI, try shorter periods and less extreme thresholds")
    print("5. For Bollinger Bands, try shorter windows and lower std_dev")
    print("\nTo run a backtest with the random strategy:")
    print("python crypto_trading_bot/main.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31")

if __name__ == "__main__":
    main() 