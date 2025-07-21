# core/bot.py - Main trading bot class with portfolio management
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from apscheduler.schedulers.background import BackgroundScheduler

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.core.strategies import StrategyFactory
from crypto_trading_bot.core.risk_manager import RiskManager
from crypto_trading_bot.core.portfolio_manager import PortfolioManager
from crypto_trading_bot.database.models import DatabaseManager
from crypto_trading_bot.notifications.notifier import Notifier
from crypto_trading_bot.core.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

@dataclass
class MarketOpportunity:
    """Represents a trading opportunity with score"""
    symbol: str
    signal: TradeSignal
    score: float  # Higher score = better opportunity
    volume_24h: float
    volatility: float

class TradingBot:
    """Main trading bot class with portfolio management"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.running = False
        self.client = None
        self.scheduler = BackgroundScheduler()
        self.db_manager = DatabaseManager(settings.get_database_config().get('path', 'trading_bot.db'))
        self.risk_manager = RiskManager(settings)
        self.notifier = Notifier(settings.get_notification_config())
        self.strategy_factory = StrategyFactory()
        
        # Initialize portfolio manager with initial investment
        trading_config = settings.get_trading_config()
        initial_investment = trading_config.get('initial_investment', 10000.0)
        max_allocation_per_trade = trading_config.get('max_allocation_per_trade', 0.05)
        
        self.portfolio_manager = PortfolioManager(
            total_investment=initial_investment,
            max_allocation_per_trade=max_allocation_per_trade
        )
        
        # Strategy management - can use different strategies for different coins
        self.strategies = {}  # {symbol: strategy}
        self.default_strategy = None
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Position tracking with enhanced data
        self.positions = {}  # {symbol: {'side': 'BUY', 'entry_price': float, 'quantity': float, 'allocated_amount': float}}
        
        # Market data cache
        self.market_data_cache = {}  # {symbol: {'price': float, 'volume_24h': float, 'volatility': float, 'last_update': datetime}}
        
        # Thread lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Track active symbols and their performance
        self.symbol_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0})
        
        # Initialize components
        self._initialize_binance_client()
        self._initialize_strategies()
        
    def _initialize_binance_client(self):
        """Initialize Binance API client"""
        try:
            api_key = self.settings.get_api_key()
            api_secret = self.settings.get_api_secret()
            
            if not api_key or not api_secret:
                raise ValueError("API key and secret are required")
            
            self.client = Client(
                api_key,
                api_secret,
                testnet=bool(self.settings.get_binance_config().get('testnet', True))
            )
            self.client.REQUEST_RECVWINDOW = 10000
            logger.info(
                f"Binance client initialized with testnet={self.client.testnet} and recvWindow={self.client.REQUEST_RECVWINDOW}"
            )
            
            # Test connection
            server_time = self.client.get_server_time()
            logger.info(f"Connected to Binance API. Server time: {server_time}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize trading strategies for multiple coins"""
        try:
            strategy_config = self.settings.get_strategy_config()
            default_strategy_name = strategy_config.get('name', 'moving_average_crossover')
            default_strategy_params = strategy_config.get('parameters', {})
            
            # Create default strategy
            self.default_strategy = self.strategy_factory.create_strategy(
                default_strategy_name,
                default_strategy_params
            )
            
            # Allow per-symbol strategy override
            symbol_strategies = strategy_config.get('symbol_strategies', {})
            for symbol, strategy_info in symbol_strategies.items():
                strategy_name = strategy_info.get('name', default_strategy_name)
                strategy_params = strategy_info.get('parameters', default_strategy_params)
                self.strategies[symbol] = self.strategy_factory.create_strategy(
                    strategy_name,
                    strategy_params
                )
            
            logger.info(f"Default strategy '{default_strategy_name}' initialized")
            logger.info(f"Symbol-specific strategies: {list(self.strategies.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise
    
    def get_strategy_for_symbol(self, symbol: str):
        """Get the appropriate strategy for a symbol"""
        return self.strategies.get(symbol, self.default_strategy)
    
    def start(self):
        """Start the trading bot"""
        if not self.settings.validate_config():
            logger.error("Configuration validation failed")
            return False
        
        if self.running:
            logger.warning("Bot is already running")
            return True
        
        logger.info("Starting trading bot...")
        
        try:
            # Start scheduler
            if not self.scheduler.running:
                self.scheduler.start()
            
            # Schedule market scanning and trading
            poll_interval = self.settings.get_trading_config().get('polling_interval', 5)
            self.scheduler.add_job(
                self._scan_and_trade,
                'interval',
                seconds=max(5, poll_interval),
                id='market_scanning',
                replace_existing=True
            )
            
            # Schedule portfolio updates
            self.scheduler.add_job(
                self._update_portfolio_values,
                'interval',
                seconds=30,  # Update portfolio values every 30 seconds
                id='portfolio_update',
                replace_existing=True
            )
            
            # Schedule daily reset
            self.scheduler.add_job(
                self._reset_daily_stats,
                'cron',
                hour=0,
                minute=0,
                id='daily_reset',
                replace_existing=True
            )
            
            # Schedule database backup
            if self.settings.get_database_config().get('backup_enabled', True):
                backup_interval = self.settings.get_database_config().get('backup_interval_hours', 24)
                self.scheduler.add_job(
                    self._backup_database,
                    'interval',
                    hours=max(1, backup_interval),
                    id='database_backup',
                    replace_existing=True
                )
            
            self.running = True
            logger.info("Trading bot started successfully")
            
            # Send startup notification with portfolio info
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            self.notifier.send_notification(
                "Trading Bot Started",
                f"Initial investment: ${portfolio_summary['total_investment']:,.2f}\n"
                f"Max per trade: ${portfolio_summary['max_trade_amount']:,.2f}"
            )
            
            # Main loop
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received stop signal")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.stop()
            return False
            
        return True
    
    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            logger.warning("Bot is not running")
            return
            
        logger.info("Stopping trading bot...")
        self.running = False
        
        try:
            # Stop scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            
            # Get final portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Send shutdown notification
            self.notifier.send_notification(
                "Trading Bot Stopped",
                f"Bot stopped.\n"
                f"Total trades: {self.total_trades}\n"
                f"Portfolio value: ${portfolio_summary['current_portfolio_value']:,.2f}\n"
                f"Total P&L: ${portfolio_summary['total_pnl']:,.2f} ({portfolio_summary['roi_percentage']:.2f}%)"
            )
            
            logger.info("Trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _scan_and_trade(self):
        """Scan multiple markets and execute best opportunities"""
        try:
            trading_config = self.settings.get_trading_config()
            symbols = trading_config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
            
            if not symbols:
                logger.warning("No symbols configured for trading")
                return
            
            # Collect all opportunities
            opportunities = []
            
            for symbol in symbols:
                try:
                    opportunity = self._analyze_symbol(symbol)
                    if opportunity:
                        opportunities.append(opportunity)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort opportunities by score (best first)
            opportunities.sort(key=lambda x: x.score, reverse=True)
            
            # Execute best opportunities within portfolio limits
            for opportunity in opportunities:
                if self._should_execute_opportunity(opportunity):
                    self._execute_opportunity(opportunity)
                    
        except Exception as e:
            logger.error(f"Error in market scanning: {e}")
            self.notifier.send_notification("Error", f"Market scanning error: {e}")
    
    def _analyze_symbol(self, symbol: str) -> Optional[MarketOpportunity]:
        """Analyze a symbol and return trading opportunity if found"""
        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Get 24h stats for volume and volatility
            ticker_24h = self.client.get_ticker(symbol=symbol)
            volume_24h = float(ticker_24h.get('volume', 0))
            price_change_percent = float(ticker_24h.get('priceChangePercent', 0))
            
            # Get historical data
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1HOUR,
                limit=100
            )
            
            if not klines:
                logger.warning(f"No klines data received for {symbol}")
                return None
            
            # Calculate volatility
            closes = [float(k[4]) for k in klines[-24:]]  # Last 24 hours
            if len(closes) >= 2:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * 100
            else:
                volatility = abs(price_change_percent)
            
            # Update market data cache
            self.market_data_cache[symbol] = {
                'price': current_price,
                'volume_24h': volume_24h,
                'volatility': volatility,
                'last_update': datetime.now()
            }
            
            # Get appropriate strategy for this symbol
            strategy = self.get_strategy_for_symbol(symbol)
            
            # Generate trading signal
            signal = strategy.generate_signal(symbol, current_price, klines)
            
            if signal:
                # Calculate opportunity score based on multiple factors
                score = self._calculate_opportunity_score(
                    signal=signal,
                    volume_24h=volume_24h,
                    volatility=volatility,
                    symbol_performance=self.symbol_performance[symbol]
                )
                
                return MarketOpportunity(
                    symbol=symbol,
                    signal=signal,
                    score=score,
                    volume_24h=volume_24h,
                    volatility=volatility
                )
            
            return None
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error analyzing {symbol}: {e}")
            return None
    
    def _calculate_opportunity_score(self, signal: TradeSignal, volume_24h: float, 
                                   volatility: float, symbol_performance: Dict) -> float:
        """Calculate a score for trading opportunity (0-100)"""
        score = 0.0
        
        # Signal confidence (0-40 points)
        score += signal.confidence * 40
        
        # Volume score (0-20 points) - higher volume is better
        if volume_24h > 1000000000:  # > $1B
            score += 20
        elif volume_24h > 100000000:  # > $100M
            score += 15
        elif volume_24h > 10000000:  # > $10M
            score += 10
        else:
            score += 5
        
        # Volatility score (0-20 points) - moderate volatility is best
        if 2 <= volatility <= 5:  # Sweet spot
            score += 20
        elif 1 <= volatility < 2 or 5 < volatility <= 10:
            score += 15
        elif 0.5 <= volatility < 1 or 10 < volatility <= 15:
            score += 10
        else:
            score += 5
        
        # Historical performance (0-20 points)
        win_rate = symbol_performance.get('win_rate', 0.5)
        score += win_rate * 20
        
        return min(score, 100)
    
    def _should_execute_opportunity(self, opportunity: MarketOpportunity) -> bool:
        """Determine if an opportunity should be executed"""
        # Check if we already have a position in this symbol
        if opportunity.symbol in self.positions:
            # For now, don't add to existing positions
            # Could implement position averaging logic here
            return False
        
        # Check minimum score threshold
        min_score = self.settings.get_trading_config().get('min_opportunity_score', 50)
        if opportunity.score < min_score:
            return False
        
        # Check risk limits
        if not self.risk_manager.check_risk_limits(opportunity.signal):
            logger.warning(f"Risk limit exceeded for {opportunity.symbol}")
            return False
        
        # Check portfolio allocation limits
        max_trade_amount = self.portfolio_manager.get_max_trade_amount()
        if max_trade_amount <= 0:
            logger.warning("No funds available for new trades")
            return False
        
        return True
    
    def _execute_opportunity(self, opportunity: MarketOpportunity):
        """Execute a trading opportunity"""
        with self._lock:
            try:
                signal = opportunity.signal
                
                if not self.client:
                    logger.error("Binance client is not initialized")
                    return
                
                # Get symbol info for precision
                symbol_info = self.client.get_symbol_info(signal.symbol)
                if not symbol_info or 'filters' not in symbol_info:
                    logger.error(f"Symbol info not found for {signal.symbol}")
                    return
                
                # Calculate investment amount based on portfolio manager
                max_trade_amount = self.portfolio_manager.get_max_trade_amount()
                
                # Adjust based on opportunity score (higher score = larger position)
                score_multiplier = 0.5 + (opportunity.score / 100) * 0.5  # 50% to 100% of max
                investment_amount = max_trade_amount * score_multiplier
                
                if investment_amount <= 0:
                    logger.error("Investment amount must be positive")
                    return
                
                raw_quantity = investment_amount / signal.price
                
                # Get step size and minimum quantity
                lot_size_filter = next(
                    (f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
                    None
                )
                
                if not lot_size_filter:
                    logger.error(f"LOT_SIZE filter not found for {signal.symbol}")
                    return
                
                step_size = Decimal(str(lot_size_filter['stepSize']))
                min_qty = Decimal(str(lot_size_filter['minQty']))
                
                # Round quantity to appropriate precision
                quantity_dec = (Decimal(str(raw_quantity)) // step_size) * step_size
                
                if quantity_dec < min_qty:
                    logger.warning(f"Calculated quantity {quantity_dec} is below minimum {min_qty} for {signal.symbol}")
                    return
                
                quantity = float(quantity_dec)
                
                # Check if portfolio manager can afford this
                can_afford, affordable_quantity = self.portfolio_manager.can_afford_position_size(
                    signal.symbol, signal.price, quantity
                )
                
                if not can_afford:
                    if affordable_quantity * signal.price < 10:  # Minimum $10 position
                        logger.warning(f"Cannot afford minimum position size for {signal.symbol}")
                        return
                    quantity = affordable_quantity
                
                # Check available balance
                if not self._check_balance(signal.symbol, signal.action, quantity, signal.price):
                    logger.warning(f"Insufficient balance for {signal.action} {quantity} {signal.symbol}")
                    return
                
                # Allocate funds in portfolio manager
                allocation_amount = quantity * signal.price
                if signal.action == 'BUY':
                    if not self.portfolio_manager.allocate_funds(signal.symbol, allocation_amount):
                        logger.warning(f"Failed to allocate funds for {signal.symbol}")
                        return
                
                # Place order
                order = self._place_order(signal, quantity)
                
                if order:
                    # Record trade with opportunity data
                    self._record_trade(signal, order, opportunity)
                    
            except BinanceOrderException as e:
                logger.error(f"Order execution failed: {e}")
                self.notifier.send_notification("Order Error", f"Failed to execute {signal.action} order: {e}")
                
            except Exception as e:
                logger.error(f"Unexpected error in trade execution: {e}")
                self.notifier.send_notification("Error", f"Trade execution error: {e}")
    
    def _check_balance(self, symbol: str, action: str, quantity: float, price: float) -> bool:
        """Check if there's sufficient balance for the trade"""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) for b in account['balances']}
            
            if action == 'BUY':
                # For buying, check quote asset balance
                quote_asset = self._get_quote_asset(symbol)
                required_amount = quantity * price * 1.01  # 1% buffer for fees
                return balances.get(quote_asset, 0) >= required_amount
                
            else:  # SELL
                # For selling, check base asset balance
                base_asset = self._get_base_asset(symbol)
                return balances.get(base_asset, 0) >= quantity
                
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return False
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extract base asset from symbol"""
        # Common quote assets
        quote_assets = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']
        for quote in quote_assets:
            if symbol.endswith(quote):
                return symbol[:-len(quote)]
        return symbol[:3]  # Default to first 3 chars
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extract quote asset from symbol"""
        # Common quote assets
        quote_assets = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']
        for quote in quote_assets:
            if symbol.endswith(quote):
                return quote
        return 'USDT'  # Default
    
    def _place_order(self, signal: TradeSignal, quantity: float) -> Optional[Dict]:
        """Place the actual order"""
        try:
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
            
            logger.info(f"Order placed: {signal.action} {quantity:.6f} {signal.symbol}")
            return order
            
        except BinanceOrderException as e:
            logger.error(f"Failed to place {signal.action} order: {e}")
            return None
    
    def _record_trade(self, signal: TradeSignal, order: Dict, opportunity: MarketOpportunity):
        """Record trade and update portfolio"""
        try:
            # Extract order details
            executed_qty = float(order.get('executedQty', 0))
            if executed_qty == 0:
                logger.warning("Order was not executed")
                return
            
            # Calculate average fill price
            fills = order.get('fills', [])
            if fills:
                total_cost = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                avg_price = total_cost / executed_qty
            else:
                avg_price = signal.price
            
            # Update positions and calculate P&L
            pnl = 0.0
            allocated_amount = avg_price * executed_qty
            
            if signal.action == 'BUY':
                # Add to positions
                self.positions[signal.symbol] = {
                    'side': 'BUY',
                    'entry_price': avg_price,
                    'quantity': executed_qty,
                    'allocated_amount': allocated_amount,
                    'entry_time': datetime.now()
                }
            elif signal.action == 'SELL' and signal.symbol in self.positions:
                pos = self.positions[signal.symbol]
                # Calculate P&L
                pnl = (avg_price - pos['entry_price']) * executed_qty
                
                # Update portfolio manager
                self.portfolio_manager.deallocate_funds(
                    signal.symbol, 
                    pos['allocated_amount'],
                    pnl
                )
                
                # Remove position
                del self.positions[signal.symbol]
                
                # Update symbol performance
                self._update_symbol_performance(signal.symbol, pnl > 0, pnl)
            
            # Create trade record
            trade_data = {
                'order_id': order['orderId'],
                'symbol': signal.symbol,
                'side': signal.action,
                'quantity': executed_qty,
                'price': avg_price,
                'timestamp': datetime.now(),
                'strategy': getattr(signal, 'strategy', 'unknown'),
                'status': order['status'],
                'pnl': pnl,
                'opportunity_score': opportunity.score,
                'volume_24h': opportunity.volume_24h,
                'volatility': opportunity.volatility
            }
            
            # Save to database
            self.db_manager.record_trade(trade_data)
            
            # Update statistics
            self.total_trades += 1
            self.daily_pnl += pnl
            
            # Get portfolio summary for notification
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Send notification
            self.notifier.send_notification(
                f"Trade Executed - {signal.action}",
                f"{signal.symbol}: {signal.action} {executed_qty:.6f} @ {avg_price:.2f}\n"
                f"Score: {opportunity.score:.1f} | P&L: ${pnl:.2f}\n"
                f"Portfolio: ${portfolio_summary['current_portfolio_value']:,.2f} "
                f"({portfolio_summary['roi_percentage']:+.2f}%)"
            )
            
            logger.info(
                f"Trade recorded: {signal.action} {executed_qty:.6f} {signal.symbol} @ {avg_price:.2f} | "
                f"P&L: ${pnl:.2f} | Score: {opportunity.score:.1f}"
            )
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _update_symbol_performance(self, symbol: str, is_win: bool, pnl: float):
        """Update performance tracking for a symbol"""
        perf = self.symbol_performance[symbol]
        perf['trades'] += 1
        perf['pnl'] += pnl
        
        # Update win rate (moving average)
        if perf['trades'] == 1:
            perf['win_rate'] = 1.0 if is_win else 0.0
        else:
            # Exponential moving average of win rate
            alpha = 0.1  # Weight for new result
            perf['win_rate'] = alpha * (1.0 if is_win else 0.0) + (1 - alpha) * perf['win_rate']
    
    def _update_portfolio_values(self):
        """Update portfolio with current market values"""
        try:
            current_positions = {}
            
            for symbol, pos in self.positions.items():
                try:
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    current_value = pos['quantity'] * current_price
                    
                    current_positions[symbol] = {
                        'current_value': current_value,
                        'entry_value': pos['allocated_amount'],
                        'unrealized_pnl': current_value - pos['allocated_amount']
                    }
                except Exception as e:
                    logger.warning(f"Could not update price for {symbol}: {e}")
                    # Use entry value as fallback
                    current_positions[symbol] = {
                        'current_value': pos['allocated_amount'],
                        'entry_value': pos['allocated_amount'],
                        'unrealized_pnl': 0
                    }
            
            # Update portfolio manager with current values
            self.portfolio_manager.update_unrealized_pnl(current_positions)
            
        except Exception as e:
            logger.error(f"Error updating portfolio values: {e}")
    
    def _reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today
            self.portfolio_manager.reset_daily_stats()
            logger.info("Daily statistics reset")
    
    def _backup_database(self):
        """Create database backup"""
        try:
            self.db_manager.backup_database()
            logger.info("Database backup completed")
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status with portfolio information"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            strategy_config = self.settings.get_strategy_config()
            trading_config = self.settings.get_trading_config()
            
            # Calculate some additional metrics
            positions_info = []
            for symbol, pos in self.positions.items():
                market_data = self.market_data_cache.get(symbol, {})
                positions_info.append({
                    'symbol': symbol,
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'current_price': market_data.get('price', pos['entry_price']),
                    'allocated': pos['allocated_amount'],
                    'entry_time': pos.get('entry_time', 'Unknown')
                })
            
            status = {
                'running': self.running,
                'strategy': strategy_config.get('name', 'Unknown'),
                'symbols': trading_config.get('symbols', []),
                'total_trades': self.total_trades,
                'daily_pnl': round(self.daily_pnl, 2),
                'portfolio': {
                    'initial_investment': portfolio_summary['total_investment'],
                    'current_value': portfolio_summary['current_portfolio_value'],
                    'available_funds': portfolio_summary['available_funds'],
                    'allocated_funds': portfolio_summary['allocated_funds'],
                    'total_pnl': portfolio_summary['total_pnl'],
                    'roi_percentage': portfolio_summary['roi_percentage'],
                    'max_trade_amount': portfolio_summary['max_trade_amount']
                },
                'positions': positions_info,
                'active_positions': len(self.positions),
                'last_reset_date': self.last_reset_date.isoformat(),
                'scheduler_running': self.scheduler.running if self.scheduler else False,
                'symbol_performance': dict(self.symbol_performance)
            }
            
            logger.info(f"Status requested: {status}")
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'running': self.running,
                'error': str(e)
            }