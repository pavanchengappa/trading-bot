# core/bot.py - Enhanced trading bot with increased trade frequency
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict, deque

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
    """Enhanced trading bot with increased trade generation"""
    
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
        
        # Strategy management - multiple strategies per symbol
        self.strategies = {}  # {symbol: [strategy1, strategy2, ...]}
        self.default_strategies = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Enhanced position tracking - allow multiple positions per symbol
        self.positions = {}  # {f"{symbol}_{timestamp}": position_data}
        
        # Market data cache with trend analysis
        self.market_data_cache = {}
        self.trend_cache = {}  # Store short-term trend data
        
        # Thread lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Track active symbols and their performance
        self.symbol_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0})
        
        # Trade frequency controls
        self.last_trade_time = defaultdict(float)  # {symbol: timestamp}
        self.min_trade_interval = trading_config.get('min_trade_interval_seconds', 30)  # Reduced from default
        
        # Initialize components
        self._initialize_binance_client()
        self._initialize_strategies()
        
        # Track recent opportunities for GUI
        self.recent_opportunities = deque(maxlen=50)
        
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
        """Initialize multiple trading strategies for increased signal generation"""
        try:
            strategy_config = self.settings.get_strategy_config()
            
            # Create multiple default strategies for broader coverage
            default_strategies = [
                ('moving_average_crossover', {'short_window': 3, 'long_window': 8, 'min_crossover_strength': 0.0001}),
                ('moving_average_crossover', {'short_window': 5, 'long_window': 12, 'min_crossover_strength': 0.0001}),
                ('rsi_strategy', {'rsi_period': 7, 'rsi_overbought': 75, 'rsi_oversold': 25}),
                ('rsi_strategy', {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30}),
                ('bollinger_bands', {'bb_window': 15, 'bb_std_dev': 1.5, 'min_breakout_strength': 0.001}),
            ]
            
            self.default_strategies = []
            for strategy_name, params in default_strategies:
                strategy = self.strategy_factory.create_strategy(strategy_name, params)
                self.default_strategies.append(strategy)
            
            # Allow per-symbol strategy override with multiple strategies
            symbol_strategies = strategy_config.get('symbol_strategies', {})
            trading_config = self.settings.get_trading_config()
            symbols = trading_config.get('symbols', ['BTCUSDT'])
            
            for symbol in symbols:
                if symbol in symbol_strategies:
                    strategy_info = symbol_strategies[symbol]
                    strategy_name = strategy_info.get('name', 'moving_average_crossover')
                    strategy_params = strategy_info.get('parameters', {})
                    self.strategies[symbol] = [self.strategy_factory.create_strategy(strategy_name, strategy_params)]
                else:
                    # Use all default strategies for symbols without specific config
                    self.strategies[symbol] = self.default_strategies.copy()
            
            logger.info(f"Initialized {len(self.default_strategies)} default strategies")
            logger.info(f"Symbol-specific strategies: {list(self.strategies.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            raise
    
    def get_strategies_for_symbol(self, symbol: str) -> List:
        """Get all strategies for a symbol"""
        return self.strategies.get(symbol, self.default_strategies)
    
    def start(self):
        """Start the trading bot with enhanced scheduling"""
        if not self.settings.validate_config():
            logger.error("Configuration validation failed")
            return False
        
        if self.running:
            logger.warning("Bot is already running")
            return True
        
        logger.info("Starting enhanced trading bot...")
        
        try:
            # Start scheduler
            if not self.scheduler.running:
                self.scheduler.start()
            
            # Schedule market scanning and trading with higher frequency
            poll_interval = max(10, self.settings.get_trading_config().get('polling_interval', 30))  # Minimum 10 seconds
            self.scheduler.add_job(
                self._scan_and_trade,
                'interval',
                seconds=poll_interval,
                id='market_scanning',
                replace_existing=True,
                max_instances=3
            )
            
            # Add multiple scanning jobs with different intervals for different strategies
            self.scheduler.add_job(
                self._quick_scan,
                'interval',
                seconds=max(5, poll_interval // 2),  # Faster scanning but safe
                id='quick_scan',
                replace_existing=True,
                max_instances=3
            )
            
            # Schedule portfolio updates more frequently
            self.scheduler.add_job(
                self._update_portfolio_values,
                'interval',
                seconds=15,  # Every 15 seconds
                id='portfolio_update',
                replace_existing=True
            )
            
            # Schedule trend analysis updates
            self.scheduler.add_job(
                self._update_trend_analysis,
                'interval',
                seconds=10,  # Every 10 seconds
                id='trend_update',
                replace_existing=True
            )
            
            # Schedule position management (check for exits)
            self.scheduler.add_job(
                self._manage_positions,
                'interval',
                seconds=5,  # Every 5 seconds
                id='position_management',
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
                backup_interval = self.settings.get_database_config().get('backup_interval_hours', 12)
                self.scheduler.add_job(
                    self._backup_database,
                    'interval',
                    hours=max(1, backup_interval),
                    id='database_backup',
                    replace_existing=True
                )
            
            self.running = True
            logger.info("Enhanced trading bot started successfully")
            
            # Send startup notification with portfolio info
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            self.notifier.send_notification(
                "Enhanced Trading Bot Started",
                f"Initial investment: ${portfolio_summary['total_investment']:,.2f}\n"
                f"Max per trade: ${portfolio_summary['max_trade_amount']:,.2f}\n"
                f"Strategies: {len(self.default_strategies)} per symbol"
            )
            
            # Main loop
            while self.running:
                time.sleep(0.5)  # Reduced sleep time
                
        except KeyboardInterrupt:
            logger.info("Received stop signal")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.stop()
            return False
            
        return True
    
    def _quick_scan(self):
        """Quick scan for immediate opportunities using simplified criteria"""
        try:
            trading_config = self.settings.get_trading_config()
            symbols = trading_config.get('symbols', ['BTCUSDT'])
            
            for symbol in symbols:
                try:
                    # Check if enough time has passed since last trade
                    if time.time() - self.last_trade_time[symbol] < self.min_trade_interval:
                        continue
                    
                    # Quick price check and simple signal generation
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Get minimal kline data for quick analysis
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        limit=20  # Reduced for speed
                    )
                    
                    if klines:
                        # Use only the fastest strategy for quick scan
                        strategies = self.get_strategies_for_symbol(symbol)
                        if strategies:
                            signal = strategies[0].generate_signal(symbol, current_price, klines)
                            if signal:
                                opportunity = MarketOpportunity(
                                    symbol=symbol,
                                    signal=signal,
                                    score=60 + signal.confidence * 20,  # Boost score for quick execution
                                    volume_24h=0,  # Skip volume check for speed
                                    volatility=0   # Skip volatility calc for speed
                                )
                                
                                if self._should_execute_opportunity(opportunity, quick_scan=True):
                                    self._execute_opportunity(opportunity)
                                    self.last_trade_time[symbol] = time.time()
                                
                                # Add to recent opportunities for GUI
                                cached_data = self.market_data_cache.get(symbol, {})
                                self.recent_opportunities.append({
                                    'symbol': symbol,
                                    'price': current_price,
                                    'volume_24h': cached_data.get('volume_24h', 0),
                                    'volatility': cached_data.get('volatility', 0),
                                    'score': opportunity.score,
                                    'signal': signal.action,
                                    'timestamp': datetime.now().isoformat()
                                })
                    
                except Exception as e:
                    logger.error(f"Error in quick scan for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in quick scanning: {e}")
    
    def _scan_and_trade(self):
        """Enhanced scan with multiple strategies per symbol"""
        try:
            trading_config = self.settings.get_trading_config()
            symbols = trading_config.get('symbols', ['BTCUSDT'])
            
            if not symbols:
                logger.warning("No symbols configured for trading")
                return
            
            # Collect all opportunities from all strategies
            opportunities = []
            
            for symbol in symbols:
                try:
                    symbol_opportunities = self._analyze_symbol_comprehensive(symbol)
                    opportunities.extend(symbol_opportunities)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort opportunities by score (best first)
            opportunities.sort(key=lambda x: x.score, reverse=True)
            
            # Execute top opportunities
            executed_count = 0
            max_concurrent_trades = trading_config.get('max_concurrent_trades_per_scan', 3)
            
            for opportunity in opportunities:
                if executed_count >= max_concurrent_trades:
                    break
                    
                if self._should_execute_opportunity(opportunity):
                    self._execute_opportunity(opportunity)
                    executed_count += 1
                    self.last_trade_time[opportunity.symbol] = time.time()
                    
        except Exception as e:
            logger.error(f"Error in market scanning: {e}")
            self.notifier.send_notification("Error", f"Market scanning error: {e}")
    
    def _analyze_symbol_comprehensive(self, symbol: str) -> List[MarketOpportunity]:
        """Analyze symbol with all available strategies"""
        opportunities = []
        
        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Get 24h stats
            ticker_24h = self.client.get_ticker(symbol=symbol)
            volume_24h = float(ticker_24h.get('volume', 0))
            price_change_percent = float(ticker_24h.get('priceChangePercent', 0))
            
            # Get historical data with more data points
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=50  # Increased for better analysis
            )
            
            if not klines:
                return opportunities
            
            # Calculate volatility
            closes = [float(k[4]) for k in klines[-24:]]
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
            
            # --- PHASE 2: Multi-Timeframe Analysis ---
            # Fetch 1h klines for trend detection
            htf_trend = 'NEUTRAL'
            try:
                klines_1h = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1HOUR,
                    limit=50
                )
                if klines_1h and len(klines_1h) >= 20:
                    closes_1h = [float(k[4]) for k in klines_1h]
                    # Simple EMA trend on 1h
                    ema_20 = self._calculate_ema(closes_1h, 20)
                    ema_50 = self._calculate_ema(closes_1h, 50)
                    
                    if ema_20 and ema_50:
                        if ema_20 > ema_50:
                            htf_trend = 'BULLISH'
                        elif ema_20 < ema_50:
                            htf_trend = 'BEARISH'
            except Exception as e:
                logger.warning(f"Failed to fetch HTF data for {symbol}: {e}")
            
            context = {
                'htf_trend': htf_trend
            }
            
            # Test all strategies for this symbol
            strategies = self.get_strategies_for_symbol(symbol)
            for i, strategy in enumerate(strategies):
                try:
                    # Pass context to generate_signal
                    signal = strategy.generate_signal(symbol, current_price, klines, context=context)
                    if signal:
                        # Adjust score based on strategy index (prefer diverse signals)
                        score = self._calculate_opportunity_score(
                            signal=signal,
                            volume_24h=volume_24h,
                            volatility=volatility,
                            symbol_performance=self.symbol_performance[symbol],
                            strategy_index=i
                        )
                        opportunity = MarketOpportunity(
                            symbol=symbol,
                            signal=signal,
                            score=score, # Use the calculated score
                            volume_24h=volume_24h,
                            volatility=volatility
                        )
                        opportunities.append(opportunity)
                        
                        # Add to recent opportunities for GUI
                        self.recent_opportunities.append({
                            'symbol': symbol,
                            'price': current_price,
                            'volume_24h': volume_24h,
                            'volatility': volatility,
                            'score': opportunity.score,
                            'signal': signal.action,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    logger.error(f"Error with strategy {i} for {symbol}: {e}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return opportunities
    
    def _calculate_opportunity_score(self, signal: TradeSignal, volume_24h: float, 
                                   volatility: float, symbol_performance: Dict,
                                   strategy_index: int = 0) -> float:
        """Enhanced scoring with strategy diversity bonus"""
        score = 0.0
        
        # Signal confidence (0-30 points) - reduced to make room for other factors
        score += signal.confidence * 30
        
        # Volume score (0-15 points)
        if volume_24h > 1000000000:  # > $1B
            score += 15
        elif volume_24h > 100000000:  # > $100M
            score += 12
        elif volume_24h > 10000000:  # > $10M
            score += 8
        else:
            score += 5
        
        # Volatility score (0-15 points) - favor higher volatility for more trades
        if volatility > 10:  # High volatility
            score += 15
        elif volatility > 5:
            score += 12
        elif volatility > 2:
            score += 8
        else:
            score += 5
        
        # Historical performance (0-15 points)
        win_rate = symbol_performance.get('win_rate', 0.5)
        score += win_rate * 15
        
        # Strategy diversity bonus (0-10 points)
        diversity_bonus = min(strategy_index * 2, 10)  # Bonus for using different strategies
        score += diversity_bonus
        
        # Time-based bonus (0-15 points) - encourage more frequent trading
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 20:  # Trading hours bonus
            score += 10
        else:
            score += 5
        
        return min(score, 100)
    
    def _should_execute_opportunity(self, opportunity: MarketOpportunity, quick_scan: bool = False) -> bool:
        """Enhanced opportunity validation with relaxed criteria"""
        # Relaxed score threshold
        if quick_scan:
            min_score = 40  # Lower threshold for quick opportunities
        else:
            min_score = self.settings.get_trading_config().get('min_opportunity_score', 35)  # Reduced default
        
        if opportunity.score < min_score:
            return False
        
        # Allow multiple positions per symbol with different entry times
        position_key = f"{opportunity.symbol}_{int(time.time() // 300)}"  # 5-minute buckets
        existing_positions = [k for k in self.positions.keys() if k.startswith(f"{opportunity.symbol}_")]
        
        # Limit positions per symbol
        max_positions_per_symbol = self.settings.get_trading_config().get('max_positions_per_symbol', 3)
        if len(existing_positions) >= max_positions_per_symbol:
            return False
        
        # Check risk limits (relaxed)
        try:
            # Gather correlation data if needed
            correlation_data = {}
            if self.risk_manager.max_correlation < 1.0:
                # Get price history for candidate symbol
                klines = self.client.get_klines(
                    symbol=opportunity.symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=50
                )
                correlation_data[opportunity.symbol] = [float(k[4]) for k in klines]
                
                # Get price history for active positions
                active_symbols = set()
                for pos in self.positions.values():
                    active_symbols.add(pos.get('symbol'))
                
                for sym in active_symbols:
                    if sym != opportunity.symbol:
                        klines = self.client.get_klines(
                            symbol=sym,
                            interval=Client.KLINE_INTERVAL_1MINUTE,
                            limit=50
                        )
                        correlation_data[sym] = [float(k[4]) for k in klines]

            if not self.risk_manager.check_risk_limits(opportunity.signal, opportunity.volatility, correlation_data):
                logger.debug(f"Risk limit check failed for {opportunity.symbol}")
                return False
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            pass  # Continue if risk check fails
        
        # Check portfolio allocation limits
        max_trade_amount = self.portfolio_manager.get_max_trade_amount()
        if max_trade_amount <= 10:  # Minimum $10 trade
            return False
        
        return True
    
    def _update_trend_analysis(self):
        """Update short-term trend analysis for faster decision making"""
        try:
            trading_config = self.settings.get_trading_config()
            symbols = trading_config.get('symbols', ['BTCUSDT'])
            
            for symbol in symbols:
                try:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        limit=10
                    )
                    
                    if klines and len(klines) >= 5:
                        closes = [float(k[4]) for k in klines]
                        
                        # Simple trend calculation
                        recent_trend = (closes[-1] - closes[-5]) / closes[-5]
                        momentum = (closes[-1] - closes[-3]) / closes[-3]
                        
                        self.trend_cache[symbol] = {
                            'trend': recent_trend,
                            'momentum': momentum,
                            'last_update': time.time()
                        }
                        
                except Exception as e:
                    logger.debug(f"Error updating trend for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
    
    def _manage_positions(self):
        """Enhanced position management with quicker exits"""
        try:
            current_time = time.time()
            trading_config = self.settings.get_trading_config()
            
            # Quick exit parameters
            quick_profit_threshold = trading_config.get('quick_profit_threshold', 0.02)  # 2% quick profit
            max_hold_time = trading_config.get('max_position_hold_seconds', 600)  # 10 minutes max hold
            stop_loss_threshold = trading_config.get('quick_stop_loss', 0.03)  # 3% stop loss
            
            positions_to_close = []
            
            for position_key, position in self.positions.items():
                try:
                    symbol = position['symbol'] if 'symbol' in position else position_key.split('_')[0]
                    
                    # Get current price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    entry_price = position['entry_price']
                    entry_time = position.get('entry_time', current_time)
                    hold_time = current_time - entry_time.timestamp() if hasattr(entry_time, 'timestamp') else 0
                    
                    # Calculate P&L percentage
                    if position['side'] == 'BUY':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""
                    
                    # Quick profit taking
                    if pnl_pct >= quick_profit_threshold:
                        should_exit = True
                        exit_reason = f"Quick profit: {pnl_pct:.2%}"
                    
                    # Stop loss
                    elif pnl_pct <= -stop_loss_threshold:
                        should_exit = True
                        exit_reason = f"Stop loss: {pnl_pct:.2%}"
                    
                    # Time-based exit
                    elif hold_time > max_hold_time:
                        should_exit = True
                        exit_reason = f"Time limit: {hold_time:.0f}s"
                    
                    if should_exit:
                        positions_to_close.append((position_key, position, exit_reason))
                        
                except Exception as e:
                    logger.error(f"Error managing position {position_key}: {e}")
            
            # Close positions
            for position_key, position, reason in positions_to_close:
                self._close_position(position_key, position, reason)
                
        except Exception as e:
            logger.error(f"Error in position management: {e}")
    
    def _close_position(self, position_key: str, position: Dict, reason: str):
        """Close a position with given reason"""
        try:
            symbol = position.get('symbol', position_key.split('_')[0])
            quantity = position['quantity']
            
            # Determine close action
            close_action = 'SELL' if position['side'] == 'BUY' else 'BUY'
            
            # Get current price for signal
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Create close signal
            close_signal = TradeSignal(
                symbol=symbol,
                action=close_action,
                price=current_price,
                quantity=quantity,
                timestamp=datetime.now(),
                strategy='position_management',
                confidence=0.8
            )
            
            # Place close order
            order = self._place_order(close_signal, quantity)
            
            if order:
                # Calculate P&L
                entry_price = position['entry_price']
                if position['side'] == 'BUY':
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity
                
                # Update portfolio
                self.portfolio_manager.deallocate_funds(
                    symbol,
                    position['allocated_amount'],
                    pnl
                )
                
                # Remove position
                del self.positions[position_key]
                
                # Update performance
                self._update_symbol_performance(symbol, pnl > 0, pnl)
                
                # Record trade in database
                trade_data = {
                    'order_id': order['orderId'],
                    'symbol': symbol,
                    'side': close_action,
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'strategy': 'position_management',
                    'status': order['status'],
                    'pnl': pnl,
                    'fees': 0.0, # Placeholder
                    'position_key': position_key
                }
                self.db_manager.record_trade(trade_data)
                self.total_trades += 1
                
                # Send notification
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                self.notifier.send_notification(
                    f"Position Closed - {symbol}",
                    f"{symbol}: {close_action} {quantity:.6f} @ {current_price:.2f}\n"
                    f"Reason: {reason}\n"
                    f"P&L: ${pnl:.2f} ({(pnl/position['allocated_amount'])*100:+.2f}%)\n"
                    f"Portfolio: ${portfolio_summary['current_portfolio_value']:,.2f}"
                )
                
                logger.info(f"Position closed: {symbol} - {reason} - P&L: ${pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error closing position {position_key}: {e}")
    
    def _calculate_ema(self, prices: List[float], window: int) -> Optional[float]:
        """Helper to calculate EMA for Bot internal use"""
        if len(prices) < window:
            return None
        
        import numpy as np
        prices_array = np.array(prices)
        alpha = 2 / (window + 1)
        ema = [prices_array[0]]
        
        for price in prices_array[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        
        return float(ema[-1])

    def _execute_opportunity(self, opportunity: MarketOpportunity):
        """Enhanced execution with position tracking"""
        with self._lock:
            try:
                signal = opportunity.signal
                
                if not self.client:
                    logger.error("Binance client is not initialized")
                    return
                
                # Handle SELL signals (Close Position Logic)
                if signal.action == 'SELL':
                    # Find active positions for this symbol
                    positions_to_close = []
                    for key, pos in self.positions.items():
                        if pos.get('symbol') == signal.symbol or key.startswith(f"{signal.symbol}_"):
                            positions_to_close.append((key, pos))
                    
                    if not positions_to_close:
                        logger.debug(f"Ignored SELL signal for {signal.symbol} - No active positions")
                        return

                    # Close all positions for this symbol
                    for key, pos in positions_to_close:
                        self._close_position(key, pos, "Strategy Signal")
                    return

                # Handle BUY signals (Open Position Logic)
                # Get symbol info for precision
                symbol_info = self.client.get_symbol_info(signal.symbol)
                if not symbol_info or 'filters' not in symbol_info:
                    logger.error(f"Symbol info not found for {signal.symbol}")
                    return
                
                # Calculate investment amount
                # Use Kelly Criterion if enabled
                win_rate = self.symbol_performance[signal.symbol].get('win_rate', 0.5)
                # Estimate win/loss ratio from recent trades or default to 1.5
                win_loss_ratio = 1.5 
                
                max_trade_amount = self.portfolio_manager.get_max_trade_amount_kelly(win_rate, win_loss_ratio)
                score_multiplier = 0.3 + (opportunity.score / 100) * 0.7  # 30% to 100% of max
                investment_amount = max_trade_amount * score_multiplier
                
                # Minimum trade amount
                min_trade_amount = self.settings.get_trading_config().get('min_trade_amount', 20)
                investment_amount = max(investment_amount, min_trade_amount)
                
                if investment_amount <= 0:
                    return
                
                raw_quantity = investment_amount / signal.price
                
                # Get precision requirements
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
                
                # Portfolio affordability check
                can_afford, affordable_quantity = self.portfolio_manager.can_afford_position_size(
                    signal.symbol, signal.price, quantity
                )
                
                if not can_afford and affordable_quantity * signal.price >= min_trade_amount:
                    quantity = affordable_quantity
                elif not can_afford:
                    return
                
                # Balance check
                if not self._check_balance(signal.symbol, signal.action, quantity, signal.price):
                    return
                
                # Allocate funds
                allocation_amount = quantity * signal.price
                if not self.portfolio_manager.allocate_funds(signal.symbol, allocation_amount):
                    return
                
                # Place order
                order = self._place_order(signal, quantity)
                
                if order:
                    self._record_trade(signal, order, opportunity)
                    
            except Exception as e:
                logger.error(f"Error in enhanced trade execution: {e}")
    
    def _record_trade(self, signal: TradeSignal, order: Dict, opportunity: MarketOpportunity):
        """Enhanced trade recording with position key generation"""
        try:
            executed_qty = float(order.get('executedQty', 0))
            if executed_qty == 0:
                return
            
            # Calculate average fill price
            fills = order.get('fills', [])
            if fills:
                total_cost = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                avg_price = total_cost / executed_qty
            else:
                avg_price = signal.price
            
            allocated_amount = avg_price * executed_qty
            current_time = datetime.now()
            
            # Create unique position key
            position_key = f"{signal.symbol}_{int(current_time.timestamp())}"
            
            if signal.action == 'BUY':
                # Add new position
                self.positions[position_key] = {
                    'symbol': signal.symbol,
                    'side': 'BUY',
                    'entry_price': avg_price,
                    'quantity': executed_qty,
                    'allocated_amount': allocated_amount,
                    'entry_time': current_time
                }
            
            # Create trade record
            trade_data = {
                'order_id': order['orderId'],
                'symbol': signal.symbol,
                'side': signal.action,
                'quantity': executed_qty,
                'price': avg_price,
                'timestamp': current_time,
                'strategy': getattr(signal, 'strategy', 'unknown'),
                'status': order['status'],
                'pnl': 0.0,  # Will be calculated on close
                'opportunity_score': opportunity.score,
                'volume_24h': opportunity.volume_24h,
                'volatility': opportunity.volatility,
                'position_key': position_key
            }
            
            # Save to database
            self.db_manager.record_trade(trade_data)
            
            # Update statistics
            self.total_trades += 1
            
            # Get portfolio summary for notification
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Send notification
            self.notifier.send_notification(
                f"Enhanced Trade - {signal.action}",
                f"{signal.symbol}: {signal.action} {executed_qty:.6f} @ {avg_price:.2f}\n"
                f"Score: {opportunity.score:.1f} | Position: {position_key}\n"
                f"Portfolio: ${portfolio_summary['current_portfolio_value']:,.2f} "
                f"({portfolio_summary['roi_percentage']:+.2f}%)"
            )
            
            logger.info(
                f"Enhanced trade recorded: {signal.action} {executed_qty:.6f} {signal.symbol} @ {avg_price:.2f} | "
                f"Score: {opportunity.score:.1f} | Key: {position_key}"
            )
            
        except Exception as e:
            logger.error(f"Error recording enhanced trade: {e}")
    
    def _check_balance(self, symbol: str, action: str, quantity: float, price: float) -> bool:
        """Check if there's sufficient balance for the trade"""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) for b in account['balances']}
            
            if action == 'BUY':
                quote_asset = self._get_quote_asset(symbol)
                required_amount = quantity * price * 1.01  # 1% buffer for fees
                return balances.get(quote_asset, 0) >= required_amount
            else:  # SELL
                base_asset = self._get_base_asset(symbol)
                return balances.get(base_asset, 0) >= quantity
                
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return False
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extract base asset from symbol"""
        quote_assets = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']
        for quote in quote_assets:
            if symbol.endswith(quote):
                return symbol[:-len(quote)]
        return symbol[:3]
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extract quote asset from symbol"""
        quote_assets = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']
        for quote in quote_assets:
            if symbol.endswith(quote):
                return quote
        return 'USDT'
    
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
            
            logger.info(f"Enhanced order placed: {signal.action} {quantity:.6f} {signal.symbol}")
            return order
            
        except BinanceOrderException as e:
            logger.error(f"Failed to place {signal.action} order: {e}")
            return None
    
    def _update_symbol_performance(self, symbol: str, is_win: bool, pnl: float):
        """Update performance tracking for a symbol"""
        perf = self.symbol_performance[symbol]
        perf['trades'] += 1
        perf['pnl'] += pnl
        
        # Update win rate (moving average)
        if perf['trades'] == 1:
            perf['win_rate'] = 1.0 if is_win else 0.0
        else:
            alpha = 0.15  # Increased weight for new results
            perf['win_rate'] = alpha * (1.0 if is_win else 0.0) + (1 - alpha) * perf['win_rate']
    
    def _update_portfolio_values(self):
        """Update portfolio with current market values"""
        try:
            current_positions = {}
            
            for position_key, pos in self.positions.items():
                try:
                    symbol = pos.get('symbol', position_key.split('_')[0])
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    current_value = pos['quantity'] * current_price
                    
                    current_positions[position_key] = {
                        'current_value': current_value,
                        'entry_value': pos['allocated_amount'],
                        'unrealized_pnl': current_value - pos['allocated_amount']
                    }
                except Exception as e:
                    logger.warning(f"Could not update price for {position_key}: {e}")
                    current_positions[position_key] = {
                        'current_value': pos['allocated_amount'],
                        'entry_value': pos['allocated_amount'],
                        'unrealized_pnl': 0
                    }
            
            # Reconcile portfolio state to ensure sync
            self.portfolio_manager.reconcile_state(self.positions)
            
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
    
    def stop(self):
        """Stop the enhanced trading bot"""
        if not self.running:
            logger.warning("Bot is not running")
            return
            
        logger.info("Stopping enhanced trading bot...")
        self.running = False
        
        try:
            # Stop scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            
            # Close all open positions before shutdown
            self._close_all_positions("Bot shutdown")
            
            # Get final portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Send shutdown notification
            self.notifier.send_notification(
                "Enhanced Trading Bot Stopped",
                f"Bot stopped with {len(self.positions)} positions closed.\n"
                f"Total trades: {self.total_trades}\n"
                f"Portfolio value: ${portfolio_summary['current_portfolio_value']:,.2f}\n"
                f"Total P&L: ${portfolio_summary['total_pnl']:,.2f} ({portfolio_summary['roi_percentage']:.2f}%)"
            )
            
            logger.info("Enhanced trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _close_all_positions(self, reason: str):
        """Close all open positions"""
        positions_to_close = list(self.positions.items())
        for position_key, position in positions_to_close:
            try:
                self._close_position(position_key, position, reason)
            except Exception as e:
                logger.error(f"Error closing position {position_key} on shutdown: {e}")
    
    def get_status(self) -> Dict:
        """Enhanced status with more detailed information"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            strategy_config = self.settings.get_strategy_config()
            trading_config = self.settings.get_trading_config()
            
            # Calculate enhanced metrics
            positions_info = []
            total_unrealized_pnl = 0
            
            for position_key, pos in self.positions.items():
                symbol = pos.get('symbol', position_key.split('_')[0])
                market_data = self.market_data_cache.get(symbol, {})
                current_price = market_data.get('price', pos['entry_price'])
                
                # Calculate unrealized P&L
                if pos['side'] == 'BUY':
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
                
                total_unrealized_pnl += unrealized_pnl
                
                positions_info.append({
                    'position_key': position_key,
                    'symbol': symbol,
                    'side': pos['side'],
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'current_price': current_price,
                    'allocated': pos['allocated_amount'],
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': (unrealized_pnl / pos['allocated_amount']) * 100,
                    'entry_time': pos.get('entry_time', 'Unknown'),
                    'hold_time_minutes': (datetime.now() - pos.get('entry_time', datetime.now())).total_seconds() / 60 if pos.get('entry_time') else 0
                })
            
            # Strategy information
            strategy_info = {
                'total_strategies': len(self.default_strategies),
                'strategies_per_symbol': {symbol: len(strategies) for symbol, strategies in self.strategies.items()}
            }
            
            status = {
                'running': self.running,
                'enhanced_features': True,
                'strategy_info': strategy_info,
                'symbols': trading_config.get('symbols', []),
                'total_trades': self.total_trades,
                'daily_pnl': round(self.daily_pnl, 2),
                'total_unrealized_pnl': round(total_unrealized_pnl, 2),
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
                'symbol_performance': dict(self.symbol_performance),
                'last_trade_times': dict(self.last_trade_time),
                'trend_cache': dict(self.trend_cache),
                'market_data_cache': dict(self.market_data_cache),
                'settings': {
                    'min_trade_interval': self.min_trade_interval,
                    'polling_interval': trading_config.get('polling_interval', 3),
                    'max_positions_per_symbol': trading_config.get('max_positions_per_symbol', 3),
                    'quick_profit_threshold': trading_config.get('quick_profit_threshold', 0.02),
                    'max_hold_time': trading_config.get('max_position_hold_seconds', 600)
                },
                'market_scanner': list(self.recent_opportunities),
                'recent_trades': self.db_manager.get_recent_trades(limit=20)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting enhanced status: {e}")
            return {
                'running': self.running,
                'enhanced_features': True,
                'error': str(e)
            }

    def close_position(self, position_key: str):
        """Manually close a specific position"""
        with self._lock:
            if position_key in self.positions:
                self._close_position(position_key, self.positions[position_key], "Manual Close")
                return True
            return False

    def close_all_positions(self):
        """Manually close all open positions"""
        with self._lock:
            # Create a copy of keys to avoid runtime error during iteration
            keys = list(self.positions.keys())
            for key in keys:
                if key in self.positions:
                    self._close_position(key, self.positions[key], "Manual Close All")
            return True