#!/usr/bin/env python3
"""
Test script to run backtest with random strategy to ensure trades are generated
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
    """Run backtest with random strategy to test the engine"""
    
    # Setup logging with DEBUG level
    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting backtest with random strategy")
    
    try:
        # Load configuration
        settings = Settings()
        
        # Use random strategy with high probability of signals
        settings.update_config("strategy_config", "name", "random_strategy")
        settings.update_config("strategy_config", "parameters", {
            "signal_probability": 0.1  # 10% chance per check (very high)
        })
        
        # Use synthetic data for testing
        settings.update_config("backtest_config", "use_real_data", False)
        settings.update_config("backtest_config", "fallback_to_synthetic", True)
        settings.update_config("backtest_config", "data_interval", "1h")
        
        # Set trading symbols
        settings.update_config("trading_config", "symbols", ["BTCUSDT"])
        settings.update_config("trading_config", "investment_amount", 100.0)
        
        # Calculate 2-year date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)  # 2 years
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Running random strategy backtest from {start_date_str} to {end_date_str}")
        logger.info(f"Strategy: {settings.get_strategy_config()}")
        logger.info(f"Signal probability: 10% per check")
        
        # Create and run backtest engine
        engine = BacktestEngine(settings)
        results = engine.run(start_date=start_date_str, end_date=end_date_str)
        
        if results:
            logger.info("Random strategy backtest completed!")
            logger.info(f"Total trades: {results.total_trades}")
            logger.info(f"Win rate: {results.win_rate:.2%}")
            logger.info(f"Total P&L: ${results.total_pnl:.2f}")
            
            if results.total_trades == 0:
                logger.warning("Random strategy generated 0 trades - this indicates a problem with the backtest engine")
            else:
                logger.info("Random strategy generated trades successfully - backtest engine is working")
        else:
            logger.error("Random strategy backtest failed")
            
    except Exception as e:
        logger.error(f"Error running random strategy backtest: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 