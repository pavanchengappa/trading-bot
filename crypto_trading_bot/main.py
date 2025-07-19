# main.py - Entry point for the trading bot
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.core.bot import TradingBot
from crypto_trading_bot.ui.cli import CLI
from crypto_trading_bot.utils.logger import setup_logging

def main():
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--mode', '-m', choices=['trade', 'backtest', 'config'], 
                       default='trade', help='Bot operation mode')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Cryptocurrency Trading Bot")
    
    try:
        # Load configuration
        settings = Settings(config_path=args.config)
        
        if args.mode == 'config':
            # Configuration mode
            cli = CLI(settings)
            cli.configure()
        elif args.mode == 'backtest':
            # Backtesting mode
            from crypto_trading_bot.backtesting.backtest_engine import BacktestEngine
            engine = BacktestEngine(settings)
            
            # Use provided dates or defaults
            start_date = args.start_date or "2024-01-01"
            end_date = args.end_date or "2024-12-31"
            
            logger.info(f"Running backtest from {start_date} to {end_date}")
            engine.run(start_date=start_date, end_date=end_date)
        else:
            # Trading mode
            if args.gui:
                from crypto_trading_bot.ui.gui import GUI
                try:
                    gui = GUI(settings)
                    gui.run()
                except Exception as gui_e:
                    logger.error(f"Error launching GUI: {gui_e}")
            else:
                bot = TradingBot(settings)
                bot.start()
                
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()