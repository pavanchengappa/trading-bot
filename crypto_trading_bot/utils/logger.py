# utils/logger.py - Logging configuration
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration for the trading bot"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Default log file if not specified
    if log_file is None:
        log_file = logs_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create specific loggers for different components
    loggers = [
        'core.bot',
        'core.strategies',
        'core.risk_manager',
        'database.models',
        'notifications.notifier',
        'backtesting.backtest_engine',
        'ui.cli',
        'ui.gui'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
    
    # Log startup message
    logging.info("Logging system initialized")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Log level: {logging.getLevelName(log_level)}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)

def log_trade(logger: logging.Logger, trade_data: dict):
    """Log trade information"""
    logger.info(
        f"TRADE: {trade_data['symbol']} {trade_data['side']} "
        f"{trade_data['quantity']:.6f} @ {trade_data['price']:.2f} "
        f"(P&L: {trade_data.get('pnl', 0):.2f})"
    )

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context"""
    logger.error(f"ERROR in {context}: {str(error)}", exc_info=True)

def log_performance(logger: logging.Logger, performance_data: dict):
    """Log performance metrics"""
    logger.info(
        f"PERFORMANCE: Trades={performance_data.get('total_trades', 0)}, "
        f"Win Rate={performance_data.get('win_rate', 0):.1%}, "
        f"P&L={performance_data.get('total_pnl', 0):.2f}"
    ) 