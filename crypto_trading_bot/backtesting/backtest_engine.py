# backtesting/backtest_engine.py - Backtesting engine for trading strategies
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.core.strategies import StrategyFactory
from crypto_trading_bot.database.models import DatabaseManager
from crypto_trading_bot.utils.binance_data import fetch_historical_data_for_backtest
from crypto_trading_bot.core.adaptive_strategy import EnhancedTradingStrategy

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest result data structure"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    total_return: float
    trades: List[Dict]

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.strategy_factory = StrategyFactory()
        self.db_manager = DatabaseManager(settings.get_database_config().get("path", "trading_bot.db"))
        
        # Backtest parameters
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.initial_balance = 10000.0  # $10,000 starting balance
        self.commission_rate = 0.001  # 0.1% commission
        
        # Results tracking
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def run(self, start_date: str = None, end_date: str = None, strategy_name: str = None):
        """Run backtest"""
        try:
            # Set dates
            if start_date:
                self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                self.start_date = datetime.now() - timedelta(days=30)
                
            if end_date:
                self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                self.end_date = datetime.now()
            
            # Debug: Print loaded config
            strategy_config = self.settings.get_strategy_config()
            logger.info(f"DEBUG: Loaded strategy config: {strategy_config}")
            logger.info(f"DEBUG: Strategy name from config: {strategy_config.get('name', 'NOT_FOUND')}")
            logger.info(f"DEBUG: Strategy parameters from config: {strategy_config.get('parameters', {})}")
            
            # Use specified strategy or default
            strategy_name = strategy_name or self.settings.get_strategy_config().get("name", "moving_average_crossover")
            
            logger.info(f"DEBUG: Final strategy name being used: {strategy_name}")
            logger.info(f"Starting backtest for {strategy_name} from {self.start_date} to {self.end_date}")
            
            # Initialize strategy
            strategy = self.strategy_factory.create_strategy(
                strategy_name,
                self.settings.get_strategy_config().get("parameters", {})
            )
            
            # Get historical data
            historical_data = self._get_historical_data()
            
            # Debug: Print number of data points loaded for each symbol
            for symbol, df in historical_data.items():
                logger.info(f"Loaded {len(df)} data points for {symbol} (head: {df.head(2)})")
            
            if not historical_data:
                logger.error("No historical data available for backtest")
                return None
            
            # Run backtest
            self._run_backtest(strategy, historical_data)
            
            # Calculate results
            results = self._calculate_results()
            
            # Display results
            self._display_results(results)
            
            # Save results to database
            self._save_backtest_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def _get_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Get historical price data for backtesting"""
        backtest_config = self.settings.get_backtest_config()
        use_real_data = backtest_config.get("use_real_data", False)
        fallback_to_synthetic = backtest_config.get("fallback_to_synthetic", True)
        data_interval = backtest_config.get("data_interval", "1h")
        
        symbols = self.settings.get_trading_config().get("symbols", ["BTCUSDT"])
        
        if use_real_data:
            logger.info("Attempting to fetch real historical data from Binance...")
            
            try:
                # Get API credentials
                api_key = self.settings.get_api_key()
                api_secret = self.settings.get_api_secret()
                
                # Fetch real data from Binance
                historical_data = fetch_historical_data_for_backtest(
                    symbols=symbols,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval=data_interval,
                    api_key=api_key,
                    api_secret=api_secret
                )
                
                if historical_data:
                    logger.info(f"Successfully fetched real data for {len(historical_data)} symbols")
                    return historical_data
                else:
                    logger.warning("No real data fetched from Binance")
                    
            except Exception as e:
                logger.error(f"Error fetching real data from Binance: {e}")
        
        # Fallback to synthetic data
        if fallback_to_synthetic:
            logger.info("Using synthetic data for backtesting")
            return self._get_synthetic_data(symbols)
        else:
            logger.error("No data available and fallback to synthetic data is disabled")
            return {}
    
    def _get_synthetic_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get synthetic price data for demonstration"""
        historical_data = {}
        
        for symbol in symbols:
            try:
                # Generate synthetic data for demonstration
                data = self._generate_synthetic_data(symbol)
                historical_data[symbol] = data
                
            except Exception as e:
                logger.error(f"Error generating synthetic data for {symbol}: {e}")
        
        return historical_data
    
    def _generate_synthetic_data(self, symbol: str) -> pd.DataFrame:
        """Generate synthetic price data for demonstration"""
        # This is for demonstration purposes only
        # In a real implementation, use actual historical data
        
        if self.start_date is None or self.end_date is None:
            raise ValueError("Start date and end date must be set")
        
        start_date = self.start_date
        end_date = self.end_date
        
        # Calculate number of days for the period
        days_diff = (end_date - start_date).days
        logger.info(f"Generating {days_diff} days of synthetic data for {symbol}")
        
        # Generate date range with hourly frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducible results
        
        # Start with a base price
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        # Generate price movements with reasonable volatility
        returns = np.random.normal(0, 0.008, len(date_range))  # 0.8% hourly volatility (reduced)
        
        # Add a much smaller trend component
        trend = np.linspace(0, 0.02, len(date_range))  # 2% total trend over period (reduced from 10%)
        seasonality = 0.005 * np.sin(np.linspace(0, 8*np.pi, len(date_range)))  # Smaller weekly cycles
        
        # Add occasional mean reversion
        mean_reversion = np.zeros(len(date_range))
        for i in range(1, len(date_range)):
            if i % 24 == 0:  # Every 24 hours
                mean_reversion[i] = np.random.normal(0, 0.01)  # 1% mean reversion (reduced)
        
        # Add occasional spikes
        spikes = np.zeros(len(date_range))
        for i in range(1, len(date_range)):
            if np.random.random() < 0.005:  # 0.5% chance of spike (reduced)
                spikes[i] = np.random.normal(0, 0.02)  # 2% spike (reduced)
        
        prices = [base_price]
        
        for i, ret in enumerate(returns[1:]):
            # Combine random walk with trend, seasonality, mean reversion, and spikes
            price_change = ret + trend[i] + seasonality[i] + mean_reversion[i] + spikes[i]
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1))  # Ensure price doesn't go negative
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],  # Lower volatility
            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],   # Lower volatility
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(date_range))
        })
        
        logger.info(f"Generated {len(data)} data points for {symbol}")
        return data
    
    def _run_backtest(self, strategy, historical_data: Dict[str, pd.DataFrame]):
        """Run the actual backtest"""
        logger.info("Running backtest simulation...")
        
        # Reset tracking variables
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Get all timestamps
        all_timestamps = set()
        for symbol_data in historical_data.values():
            all_timestamps.update(symbol_data['timestamp'].tolist())
        
        all_timestamps = sorted(list(all_timestamps))
        logger.info(f"Total timestamps to process: {len(all_timestamps)}")
        
        signal_count = 0
        
        adaptive_strategy = EnhancedTradingStrategy()
        
        # Simulate trading over time
        for i, timestamp in enumerate(all_timestamps):
            if i % 1000 == 0:  # Log progress every 1000 timestamps
                logger.info(f"Processing timestamp {i}/{len(all_timestamps)}: {timestamp}")
            
            # Update current prices
            current_prices = {}
            for symbol, data in historical_data.items():
                symbol_data = data[data['timestamp'] == timestamp]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data.iloc[0]['close']
            
            # Check for signals
            for symbol, current_price in current_prices.items():
                symbol_data = historical_data[symbol]
                symbol_data_up_to_now = symbol_data[symbol_data['timestamp'] <= timestamp]
                if len(symbol_data_up_to_now) < 50:
                    logger.debug(f"Not enough data for {symbol} at {timestamp}: {len(symbol_data_up_to_now)} points")
                    continue
                klines = []
                for _, row in symbol_data_up_to_now.tail(100).iterrows():
                    timestamp_ms = int(row['timestamp'].timestamp() * 1000)
                    kline = [
                        timestamp_ms, str(row['open']), str(row['high']), str(row['low']), str(row['close']), str(row['volume']),
                        timestamp_ms, "0", "0", "0", "0"
                    ]
                    klines.append(kline)
                # --- Adaptive strategy selection ---
                closing_prices = [float(k[4]) for k in klines]
                optimal_config = adaptive_strategy.get_optimal_strategy_config(closing_prices)
                strategy_instance = self.strategy_factory.create_strategy(optimal_config['name'], optimal_config['parameters'])
                logger.debug(f"Adaptive strategy for {symbol} at {timestamp}: {optimal_config['name']} {optimal_config['parameters']}")
                signal = strategy_instance.generate_signal(symbol, current_price, klines)
                logger.debug(f"Signal result for {symbol} at {timestamp}: {signal}")
                if signal:
                    signal_count += 1
                    logger.info(f"Signal #{signal_count} generated: {signal.action} {symbol} @ {signal.price}")
                    self._execute_backtest_trade(signal, timestamp)
                else:
                    if len(symbol_data_up_to_now) % 200 == 0:
                        logger.debug(f"No signal for {symbol} at {timestamp}, price: {current_price}")
            
            # Record equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.balance,
                'positions_value': self._calculate_positions_value(current_prices)
            })
        
        logger.info(f"Backtest completed with {len(self.trades)} trades from {signal_count} signals")
    
    def _execute_backtest_trade(self, signal, timestamp):
        """Execute a trade in the backtest"""
        symbol = signal.symbol
        action = signal.action
        price = signal.price
        
        # Calculate quantity based on investment amount
        investment_amount = self.settings.get_trading_config().get('investment_amount', 100.0)
        quantity = investment_amount / price
        
        # Apply commission
        commission = quantity * price * self.commission_rate
        
        if action == 'BUY':
            # Check if we have enough balance
            total_cost = quantity * price + commission
            if total_cost <= self.balance:
                # Execute buy
                self.balance -= total_cost
                
                if symbol in self.positions:
                    # Add to existing position
                    self.positions[symbol]['quantity'] += quantity
                    self.positions[symbol]['avg_price'] = (
                        (self.positions[symbol]['avg_price'] * self.positions[symbol]['quantity'] + 
                         price * quantity) / (self.positions[symbol]['quantity'] + quantity)
                    )
                else:
                    # Create new position
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'entry_time': timestamp
                    }
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'commission': commission,
                    'pnl': 0.0
                })
        
        elif action == 'SELL':
            # Check if we have position to sell
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Calculate P&L
                pnl = (price - position['avg_price']) * quantity - commission
                
                # Execute sell
                self.balance += quantity * price - commission
                
                # Remove position
                del self.positions[symbol]
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'commission': commission,
                    'pnl': pnl
                })
    
    def _calculate_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current value of all positions"""
        total_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['quantity'] * current_prices[symbol]
        return total_value
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results"""
        if not self.trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                total_return=0.0,
                trades=self.trades
            )
        
        # Calculate basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        winning_pnls = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        
        avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
        
        # Calculate total return
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            trades=self.trades
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0.0
        
        peak = self.initial_balance
        max_dd = 0.0
        
        for point in self.equity_curve:
            total_value = point['balance'] + point['positions_value']
            if total_value > peak:
                peak = total_value
            
            drawdown = (peak - total_value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if not self.equity_curve:
            return 0.0
        
        # Calculate returns
        returns = []
        prev_value = self.initial_balance
        
        for point in self.equity_curve:
            total_value = point['balance'] + point['positions_value']
            ret = (total_value - prev_value) / prev_value
            returns.append(ret)
            prev_value = total_value
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return avg_return / std_return * np.sqrt(252)  # Annualized
    
    def _display_results(self, results: BacktestResult):
        """Display backtest results"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        if self.start_date and self.end_date:
            print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total Return: {results.total_return:.2%}")
        print()
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print()
        print(f"Total P&L: ${results.total_pnl:,.2f}")
        print(f"Average Win: ${results.avg_win:,.2f}")
        print(f"Average Loss: ${results.avg_loss:,.2f}")
        print()
        print(f"Maximum Drawdown: {results.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print("="*50)
    
    def _save_backtest_results(self, results: BacktestResult):
        """Save backtest results to database"""
        try:
            # This would save results to a backtest_results table
            # For now, just log the results
            logger.info(f"Backtest completed: {results.total_trades} trades, {results.total_return:.2%} return")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def export_results(self, filepath: str):
        """Export backtest results to CSV"""
        try:
            if not self.trades:
                logger.warning("No trades to export")
                return
            
            # Create DataFrame
            df = pd.DataFrame(self.trades)
            df.to_csv(filepath, index=False)
            logger.info(f"Backtest results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting backtest results: {e}") 