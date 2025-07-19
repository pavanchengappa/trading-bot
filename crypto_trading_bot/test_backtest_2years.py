#!/usr/bin/env python3
"""
Test script to run backtest with 2 years of data and aggressive parameters
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import Settings
from backtesting.backtest_engine import BacktestEngine
from utils.logger import setup_logging

def main():
    """Run backtest with 2 years of data and aggressive parameters"""
    
    # Setup logging with DEBUG level to see all the debug messages
    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting 2-year backtest with aggressive parameters")
    
    try:
        # Load configuration
        settings = Settings()
        
        # Modify settings for more aggressive trading
        # Update strategy parameters to be more sensitive
        settings.update_config("strategy_config", "name", "moving_average_crossover")
        settings.update_config("strategy_config", "parameters", {
            "short_window": 3,  # Very short window
            "long_window": 7,   # Very short window
            "min_crossover_strength": 0.00001  # Very low threshold
        })
        
        # Enable real data only, do not use synthetic
        settings.update_config("backtest_config", "use_real_data", True)
        settings.update_config("backtest_config", "fallback_to_synthetic", False)
        settings.update_config("backtest_config", "data_interval", "1h")
        
        # Set trading symbols
        settings.update_config("trading_config", "symbols", ["BTCUSDT"])
        settings.update_config("trading_config", "investment_amount", 100.0)
        
        # Calculate 2-year date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 years
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Running backtest from {start_date_str} to {end_date_str}")
        logger.info(f"Strategy: {settings.get_strategy_config()}")
        logger.info(f"Symbols: {settings.get_trading_config().get('symbols')}")
        
        # Create and run backtest engine
        engine = BacktestEngine(settings)
        results = engine.run(start_date=start_date_str, end_date=end_date_str)
        
        if results:
            logger.info("Backtest completed successfully!")
            logger.info(f"Total trades: {results.total_trades}")
            logger.info(f"Win rate: {results.win_rate:.2%}")
            logger.info(f"Total P&L: ${results.total_pnl:.2f}")
        else:
            logger.error("Backtest failed or returned no results")
            
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 