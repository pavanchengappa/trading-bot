# core/bot.py - Main trading bot class
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from apscheduler.schedulers.background import BackgroundScheduler

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.core.strategies import StrategyFactory
from crypto_trading_bot.core.risk_manager import RiskManager
from crypto_trading_bot.database.models import DatabaseManager
from crypto_trading_bot.notifications.notifier import Notifier
from crypto_trading_bot.core.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    timestamp: datetime
    strategy: str
    confidence: float

class TradingBot:
    """Main trading bot class that orchestrates all trading operations"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        self.client = None
        self.scheduler = BackgroundScheduler()
        self.db_manager = DatabaseManager(settings.get_database_config().get('path', 'trading_bot.db'))
        self.risk_manager = RiskManager(settings)
        self.notifier = Notifier(settings.get_notification_config())
        self.strategy_factory = StrategyFactory()
        self.current_strategy = None
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.total_invested = 0.0
        self.total_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Position tracking for P&L calculation
        self.positions = {}  # {symbol: {'side': 'BUY', 'entry_price': float, 'quantity': float}}
        
        # Initialize components
        self._initialize_binance_client()
        self._initialize_strategy()
        
    def _initialize_binance_client(self):
        """Initialize Binance API client"""
        try:
            self.client = Client(
                self.settings.get_api_key(),
                self.settings.get_api_secret(),
                testnet=self.settings.get_binance_config().get('testnet', True)
            )
            self.client.REQUEST_RECVWINDOW = 10000 # Increase recvWindow to 10 seconds
            logger.info(f"Binance client initialized with testnet: {self.settings.get_binance_config().get('testnet', True)} and recvWindow: {self.client.REQUEST_RECVWINDOW}")
            
            # Test connection
            server_time = self.client.get_server_time()
            logger.info(f"Connected to Binance API. Server time: {server_time}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def _initialize_strategy(self):
        """Initialize trading strategy"""
        try:
            strategy_config = self.settings.get_strategy_config()
            self.current_strategy = self.strategy_factory.create_strategy(
                strategy_config.get('name', 'moving_average_crossover'),
                strategy_config.get('parameters', {})
            )
            logger.info(f"Strategy '{strategy_config.get('name', 'Unknown')}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            raise
    
    def start(self):
        """Start the trading bot"""
        if not self.settings.validate_config():
            logger.error("Configuration validation failed")
            return
        
        logger.info("Starting trading bot...")
        self.running = True
        
        # Start scheduler
        self.scheduler.start()
        
        # Schedule price polling
        self.scheduler.add_job(
            self._poll_prices,
            'interval',
            seconds=5,  # Poll every 5 seconds
            id='price_polling'
        )
        
        # Schedule daily reset
        self.scheduler.add_job(
            self._reset_daily_stats,
            'cron',
            hour=0,
            minute=0,
            id='daily_reset'
        )
        
        # Schedule database backup
        if self.settings.get_database_config().get('backup_enabled', True):
            self.scheduler.add_job(
                self._backup_database,
                'interval',
                hours=self.settings.get_database_config().get('backup_interval_hours', 24),
                id='database_backup'
            )
        
        logger.info("Trading bot started successfully")
        
        # Send startup notification
        strategy_config = self.settings.get_strategy_config()
        self.notifier.send_notification(
            "Trading Bot Started",
            f"Bot started with strategy: {strategy_config.get('name', 'Unknown')}"
        )
        
        try:
            # Main loop
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received stop signal")
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.running = False
        
        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown()
        
        # Send shutdown notification
        self.notifier.send_notification(
            "Trading Bot Stopped",
            f"Bot stopped. Total trades: {self.total_trades}, Daily P&L: {self.daily_pnl:.2f}"
        )
        
        logger.info("Trading bot stopped")
    
    def _poll_prices(self):
        """Poll current prices and execute trading logic"""
        try:
            trading_config = self.settings.get_trading_config()
            for symbol in trading_config.get('symbols', ['BTCUSDT']):
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Get historical data for strategy
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1HOUR,
                    limit=100
                )
                
                # Generate trading signal
                signal = self.current_strategy.generate_signal(
                    symbol, current_price, klines
                )
                
                if signal:
                    # Check risk management
                    if self.risk_manager.check_risk_limits(signal):
                        # Execute trade
                        self._execute_trade(signal)
                    else:
                        logger.warning(f"Risk limit exceeded for {symbol}")
                        
        except Exception as e:
            logger.error(f"Error in price polling: {e}")
            self.notifier.send_notification("Error", f"Price polling error: {e}")
    
    def _execute_trade(self, signal: TradeSignal):
        """Execute a trade based on signal"""
        try:
            # Calculate quantity based on investment amount
            trading_config = self.settings.get_trading_config()
            quantity = trading_config.get('investment_amount', 100.0) / signal.price
            
            # Round quantity to appropriate precision using Decimal and step size
            symbol_info = self.client.get_symbol_info(signal.symbol)
            step_size = float([f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['stepSize'])
            step_size_dec = Decimal(str(step_size))
            quantity_dec = (Decimal(str(quantity)) // step_size_dec) * step_size_dec
            quantity = float(quantity_dec)
            
            # Place order
            if signal.action == 'BUY':
                order = self.client.order_market_buy(
                    symbol=signal.symbol,
                    quantity=quantity
                )
            else:  # SELL
                order = self.client.order_market_sell(
                    symbol=signal.symbol,
                    quantity=quantity
                )
            
            # Record trade
            trade_data = {
                'order_id': order['orderId'],
                'symbol': signal.symbol,
                'side': signal.action,
                'quantity': float(order['executedQty']),
                'price': float(order['fills'][0]['price']) if order['fills'] else signal.price,
                'timestamp': datetime.now(),
                'strategy': signal.strategy,
                'status': order['status']
            }

            # --- P&L Calculation and Position Tracking ---
            pnl = 0.0
            symbol = signal.symbol
            side = signal.action
            price = trade_data['price']
            qty = trade_data['quantity']

            if side == 'BUY':
                # If already have a position, add to it (average price)
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    total_qty = pos['quantity'] + qty
                    avg_price = ((pos['entry_price'] * pos['quantity']) + (price * qty)) / total_qty
                    self.positions[symbol] = {
                        'side': 'BUY',
                        'entry_price': avg_price,
                        'quantity': total_qty
                    }
                else:
                    self.positions[symbol] = {
                        'side': 'BUY',
                        'entry_price': price,
                        'quantity': qty
                    }
                self.total_invested += (price * qty) # Add to total invested
            elif side == 'SELL' and symbol in self.positions:
                pos = self.positions[symbol]
                # Calculate P&L for closing (part of) the position
                close_qty = min(qty, pos['quantity'])
                pnl = (price - pos['entry_price']) * close_qty
                self.total_pnl += pnl # Add to total P&L
                # Update or remove position
                if qty >= pos['quantity']:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['quantity'] -= close_qty
            trade_data['pnl'] = pnl
            # --- End P&L Calculation ---

            self.db_manager.record_trade(trade_data)
            
            # Update statistics
            self.total_trades += 1
            self._update_daily_pnl(trade_data)
            
            # Send notification
            self.notifier.send_notification(
                f"Trade Executed - {signal.action}",
                f"{signal.symbol}: {signal.action} {quantity:.6f} @ {signal.price:.2f}"
            )
            
            logger.info(f"Trade executed: {signal.action} {quantity:.6f} {signal.symbol} @ {signal.price:.2f}")
            
        except BinanceOrderException as e:
            logger.error(f"Order execution failed: {e}")
            self.notifier.send_notification("Order Error", f"Failed to execute {signal.action} order: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in trade execution: {e}")
            self.notifier.send_notification("Error", f"Trade execution error: {e}")
    
    def _update_daily_pnl(self, trade_data: Dict):
        """Update daily profit/loss tracking"""
        # This is a simplified P&L calculation
        # In a real implementation, you'd track positions and calculate actual P&L
        if trade_data['side'] == 'SELL':
            # Simplified: assume profit if selling
            self.daily_pnl += trade_data['quantity'] * trade_data['price'] * 0.01  # 1% profit assumption
    
    def _reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info("Daily statistics reset")
    
    def _backup_database(self):
        """Create database backup"""
        try:
            self.db_manager.backup_database()
            logger.info("Database backup completed")
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
    
    def get_current_portfolio_value(self) -> float:
        """Calculate the current total value of all open positions"""
        current_value = 0.0
        for symbol, pos in self.positions.items():
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                current_value += pos['quantity'] * current_price
            except Exception as e:
                logger.warning(f"Could not get current price for {symbol}: {e}")
        return current_value

    def get_status(self) -> Dict:
        """Get current bot status"""
        strategy_config = self.settings.get_strategy_config()
        trading_config = self.settings.get_trading_config()
        current_portfolio_value = self.get_current_portfolio_value()
        return {
            'running': self.running,
            'strategy': strategy_config.get('name', 'Unknown'),
            'symbols': trading_config.get('symbols', []),
            'total_trades': self.total_trades,
            'daily_pnl': self.daily_pnl,
            'total_invested': self.total_invested,
            'total_pnl': self.total_pnl,
            'current_portfolio_value': current_portfolio_value,
            'last_reset_date': self.last_reset_date.isoformat()
        } 