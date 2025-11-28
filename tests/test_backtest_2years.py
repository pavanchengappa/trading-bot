#!/usr/bin/env python3
"""
Enhanced backtest script integrated with portfolio management and GUI
"""

import sys
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, messagebox

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import Settings
from core.portfolio_manager import PortfolioManager
from core.strategies import StrategyFactory
from core.trade_signal import TradeSignal
from utils.logger import setup_logging
from ui.gui import TradingBotGUI
from crypto_trading_bot.utils.binance_data import fetch_historical_data_for_backtest

# Mock Binance client for backtesting
class MockBinanceClient:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.current_index = 0
        
    def get_symbol_ticker(self, symbol):
        if symbol in self.historical_data and self.current_index < len(self.historical_data[symbol]):
            return {'price': str(self.historical_data[symbol][self.current_index][4])}  # Close price
        return {'price': '50000'}  # Default price
    
    def get_ticker(self, symbol):
        return {
            'volume': '1000000000',
            'priceChangePercent': '2.5'
        }
    
    def get_klines(self, symbol, interval, limit):
        if symbol in self.historical_data:
            start_idx = max(0, self.current_index - limit + 1)
            end_idx = self.current_index + 1
            return self.historical_data[symbol][start_idx:end_idx]
        return []

@dataclass
class BacktestResults:
    """Enhanced backtest results with all properties"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    roi_percentage: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration: str = "0:00:00"
    portfolio_summary: Dict = None
    trade_history: List = None
    daily_pnl: List = None

class BacktestEngine:
    """Enhanced backtest engine using portfolio management and strategies"""
    
    def __init__(self, settings: Settings, gui_queue: Optional[queue.Queue] = None):
        self.settings = settings
        self.gui_queue = gui_queue
        
        # Initialize portfolio manager
        trading_config = settings.get_trading_config()
        initial_investment = trading_config.get('initial_investment', 10000.0)
        max_allocation = trading_config.get('max_allocation_per_trade', 0.05)
        
        self.portfolio_manager = PortfolioManager(
            total_investment=initial_investment,
            max_allocation_per_trade=max_allocation
        )
        
        # Initialize strategy factory
        self.strategy_factory = StrategyFactory()
        self._initialize_strategies()
        
        # Tracking
        self.trades = []
        self.positions = {}
        self.daily_pnl_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        strategy_config = self.settings.get_strategy_config()
        default_strategy_name = strategy_config.get('name', 'moving_average_crossover')
        default_strategy_params = strategy_config.get('parameters', {})
        
        self.default_strategy = self.strategy_factory.create_strategy(
            default_strategy_name,
            default_strategy_params
        )
        
        # Symbol-specific strategies
        self.strategies = {}
        symbol_strategies = strategy_config.get('symbol_strategies', {})
        for symbol, strategy_info in symbol_strategies.items():
            strategy_name = strategy_info.get('name', default_strategy_name)
            strategy_params = strategy_info.get('parameters', default_strategy_params)
            self.strategies[symbol] = self.strategy_factory.create_strategy(
                strategy_name,
                strategy_params
            )
    
    def get_strategy_for_symbol(self, symbol: str):
        """Get appropriate strategy for symbol"""
        return self.strategies.get(symbol, self.default_strategy)
    
    def _send_update(self, message: str, progress: float = None):
        """Send update to GUI if available"""
        if self.gui_queue:
            self.gui_queue.put({
                'type': 'update',
                'message': message,
                'progress': progress
            })
    
    def run(self, start_date: str, end_date: str) -> BacktestResults:
        """Run enhanced backtest"""
        self._send_update("Starting backtest...", 0)
        
        try:
            # Load historical data (mock implementation)
            historical_data = self._load_historical_data(start_date, end_date)
            
            if not historical_data:
                self._send_update("No historical data available", 100)
                return BacktestResults()
            
            # Create mock client
            mock_client = MockBinanceClient(historical_data)
            
            symbols = self.settings.get_trading_config().get('symbols', ['BTCUSDT'])
            total_periods = len(next(iter(historical_data.values())))
            
            # Run backtest simulation
            for period_idx in range(total_periods):
                mock_client.current_index = period_idx
                progress = (period_idx / total_periods) * 100
                
                if period_idx % 100 == 0:  # Update every 100 periods
                    self._send_update(f"Processing period {period_idx}/{total_periods}", progress)
                
                # Scan all symbols for opportunities
                for symbol in symbols:
                    self._process_symbol(symbol, mock_client, period_idx)
                
                # Update portfolio values
                self._update_portfolio_values(mock_client)
                
                # Record daily P&L
                if period_idx % 24 == 0:  # Assuming hourly data, record daily
                    portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                    self.daily_pnl_history.append({
                        'date': period_idx,
                        'portfolio_value': portfolio_summary['current_portfolio_value'],
                        'pnl': portfolio_summary['total_pnl']
                    })
            
            # Calculate final results
            results = self._calculate_results()
            self._send_update("Backtest completed!", 100)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}", exc_info=True)
            self._send_update(f"Backtest failed: {str(e)}", 100)
            return BacktestResults()
    
    def _load_historical_data(self, start_date: str, end_date: str) -> Dict:
        """Load historical data from Binance"""
        self.logger.info(f"Loading historical data from {start_date} to {end_date}")

        symbols = self.settings.get_trading_config().get('symbols', ['BTCUSDT'])
        data_interval = self.settings.get_backtest_config().get('data_interval', '1h')
        api_key = self.settings.get_api_key()
        api_secret = self.settings.get_api_secret()

        # Convert string dates to datetime objects
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

        historical_data = fetch_historical_data_for_backtest(
            symbols=symbols,
            start_date=start_datetime,
            end_date=end_datetime,
            interval=data_interval,
            api_key=api_key,
            api_secret=api_secret
        )

        # Convert pandas DataFrames to list of lists (klines format)
        formatted_historical_data = {}
        for symbol, df in historical_data.items():
            # Ensure the DataFrame has the expected columns and order
            # 'Open time', 'Open', 'High', 'Low', 'Close', 'Volume'
            # The fetch_historical_data_for_backtest returns a DataFrame with 'Open time' as index
            # and 'Open', 'High', 'Low', 'Close', 'Volume' as columns.
            # We need to convert it to the klines format:
            # [timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            # The MockBinanceClient expects: [timestamp, open, high, low, close, volume]
            # So we need to extract these and convert timestamp to milliseconds.
            klines_list = []
            for index, row in df.iterrows():
                timestamp_ms = int(index.timestamp() * 1000)
                kline = [
                    timestamp_ms,
                    str(row['Open']),
                    str(row['High']),
                    str(row['Low']),
                    str(row['Close']),
                    str(row['Volume'])
                ]
                klines_list.append(kline)
            formatted_historical_data[symbol] = klines_list

        return formatted_historical_data
    
    def _process_symbol(self, symbol: str, mock_client: MockBinanceClient, period_idx: int):
        """Process trading signals for a symbol"""
        try:
            # Get current price
            ticker = mock_client.get_symbol_ticker(symbol)
            current_price = float(ticker['price'])
            
            # Get historical data for strategy
            klines = mock_client.get_klines(symbol, '1h', 100)
            
            if not klines:
                return
            
            # Get strategy and generate signal
            strategy = self.get_strategy_for_symbol(symbol)
            signal = strategy.generate_signal(symbol, current_price, klines)
            
            if signal:
                self._execute_backtest_trade(signal, current_price)
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
    
    def _execute_backtest_trade(self, signal: TradeSignal, current_price: float):
        """Execute trade in backtest"""
        try:
            symbol = signal.symbol
            
            if signal.action == 'BUY' and symbol not in self.positions:
                # Calculate position size
                max_trade_amount = self.portfolio_manager.get_max_trade_amount()
                
                if max_trade_amount <= 0:
                    return
                
                quantity = max_trade_amount / current_price
                allocation_amount = quantity * current_price
                
                # Allocate funds
                if self.portfolio_manager.allocate_funds(symbol, allocation_amount):
                    self.positions[symbol] = {
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_time': datetime.now(),
                        'allocated_amount': allocation_amount
                    }
                    
                    # Record trade
                    self.trades.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': quantity,
                        'timestamp': datetime.now(),
                        'strategy': signal.strategy
                    })
                    
            elif signal.action == 'SELL' and symbol in self.positions:
                # Close position
                pos = self.positions[symbol]
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                
                # Deallocate funds
                self.portfolio_manager.deallocate_funds(
                    symbol,
                    pos['allocated_amount'],
                    pnl
                )
                
                # Record trade
                self.trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': pos['quantity'],
                    'timestamp': datetime.now(),
                    'strategy': signal.strategy,
                    'pnl': pnl
                })
                
                del self.positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def _update_portfolio_values(self, mock_client: MockBinanceClient):
        """Update portfolio with current market values"""
        current_positions = {}
        
        for symbol, pos in self.positions.items():
            try:
                ticker = mock_client.get_symbol_ticker(symbol)
                current_price = float(ticker['price'])
                current_value = pos['quantity'] * current_price
                
                current_positions[symbol] = {
                    'current_value': current_value,
                    'entry_value': pos['allocated_amount'],
                    'unrealized_pnl': current_value - pos['allocated_amount']
                }
            except:
                current_positions[symbol] = {
                    'current_value': pos['allocated_amount'],
                    'entry_value': pos['allocated_amount'],
                    'unrealized_pnl': 0
                }
        
        self.portfolio_manager.update_unrealized_pnl(current_positions)
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        # Basic trade statistics
        total_trades = len([t for t in self.trades if t['action'] == 'SELL'])
        winning_trades = len([t for t in self.trades if t['action'] == 'SELL' and t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Portfolio summary
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        
        # Calculate additional metrics
        trade_pnls = [t.get('pnl', 0) for t in self.trades if t['action'] == 'SELL']
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio(trade_pnls)
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=portfolio_summary['total_pnl'],
            total_fees=sum(abs(pnl) * 0.001 for pnl in trade_pnls),  # Estimate 0.1% fees
            roi_percentage=portfolio_summary['roi_percentage'],
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            avg_trade_duration=self._calculate_avg_trade_duration(),
            portfolio_summary=portfolio_summary,
            trade_history=self.trades,
            daily_pnl=self.daily_pnl_history
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.daily_pnl_history:
            return 0.0
        
        peak = self.daily_pnl_history[0]['portfolio_value']
        max_dd = 0.0
        
        for record in self.daily_pnl_history:
            value = record['portfolio_value']
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, trade_pnls: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not trade_pnls:
            return 0.0
        
        import numpy as np
        returns = np.array(trade_pnls)
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns)
    
    def _calculate_avg_trade_duration(self) -> str:
        """Calculate average trade duration"""
        # Simplified implementation
        return "2:30:00"  # Mock average

class BacktestGUI(TradingBotGUI):
    """Extended GUI with backtest functionality"""
    
    def __init__(self):
        super().__init__()
        self.backtest_results = None
        self.backtest_queue = queue.Queue()
        self.backtest_thread = None
        
        # Add backtest tab
        self._create_backtest_tab()
    
    def _create_backtest_tab(self):
        """Create backtest tab"""
        backtest_frame = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(backtest_frame, text="Backtest")
        
        # Controls
        controls_frame = ttk.LabelFrame(backtest_frame, text="Backtest Controls", style="Dark.TFrame")
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Date selection
        date_frame = ttk.Frame(controls_frame, style="Dark.TFrame")
        date_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(date_frame, text="Start Date:", style="Dark.TLabel").grid(row=0, column=0, padx=5, pady=5)
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"))
        tk.Entry(date_frame, textvariable=self.start_date_var, bg=self.entry_bg, fg=self.entry_fg).grid(row=0, column=1, padx=5)
        
        ttk.Label(date_frame, text="End Date:", style="Dark.TLabel").grid(row=0, column=2, padx=5, pady=5)
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(date_frame, textvariable=self.end_date_var, bg=self.entry_bg, fg=self.entry_fg).grid(row=0, column=3, padx=5)
        
        # Run button
        tk.Button(
            controls_frame, text="Run Backtest", command=self.run_backtest,
            bg=self.button_bg, fg=self.button_fg, font=('Arial', 12, 'bold'),
            padx=20, pady=10
        ).pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            controls_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        
        # Status label
        self.backtest_status_label = ttk.Label(
            controls_frame, text="Ready to run backtest", style="Dark.TLabel"
        )
        self.backtest_status_label.pack(pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(backtest_frame, text="Backtest Results", style="Dark.TFrame")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Results grid
        self.results_grid = ttk.Frame(results_frame, style="Dark.TFrame")
        self.results_grid.pack(fill='x', padx=10, pady=10)
        
        # Initialize result displays
        self._create_result_displays()
    
    def _create_result_displays(self):
        """Create result display widgets"""
        # Row 1
        self._create_metric_display(self.results_grid, "Total Trades", "0", 0, 0, "bt_total_trades")
        self._create_metric_display(self.results_grid, "Win Rate", "0.00%", 0, 1, "bt_win_rate")
        self._create_metric_display(self.results_grid, "Total P&L", "$0.00", 0, 2, "bt_total_pnl")
        self._create_metric_display(self.results_grid, "ROI", "0.00%", 0, 3, "bt_roi")
        
        # Row 2
        self._create_metric_display(self.results_grid, "Max Drawdown", "0.00%", 1, 0, "bt_max_drawdown")
        self._create_metric_display(self.results_grid, "Sharpe Ratio", "0.000", 1, 1, "bt_sharpe_ratio")
        self._create_metric_display(self.results_grid, "Total Fees", "$0.00", 1, 2, "bt_total_fees")
        self._create_metric_display(self.results_grid, "Avg Duration", "0:00:00", 1, 3, "bt_avg_duration")
    
    def run_backtest(self):
        """Run backtest in separate thread"""
        if self.backtest_thread and self.backtest_thread.is_alive():
            messagebox.showwarning("Warning", "Backtest is already running!")
            return
        
        try:
            self.backtest_status_label.config(text="Starting backtest...")
            self.progress_var.set(0)
            
            self.backtest_thread = threading.Thread(target=self._run_backtest_thread, daemon=True)
            self.backtest_thread.start()
            
            # Start monitoring backtest
            self._monitor_backtest()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start backtest: {str(e)}")
    
    def _run_backtest_thread(self):
        """Run backtest in separate thread"""
        try:
            settings = Settings()
            engine = BacktestEngine(settings, self.backtest_queue)
            
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            
            results = engine.run(start_date, end_date)
            
            self.backtest_queue.put({
                'type': 'complete',
                'results': results
            })
            
        except Exception as e:
            self.backtest_queue.put({
                'type': 'error',
                'message': str(e)
            })
    
    def _monitor_backtest(self):
        """Monitor backtest progress"""
        try:
            while not self.backtest_queue.empty():
                message = self.backtest_queue.get_nowait()
                
                if message['type'] == 'update':
                    self.backtest_status_label.config(text=message['message'])
                    if message.get('progress'):
                        self.progress_var.set(message['progress'])
                
                elif message['type'] == 'complete':
                    self.backtest_results = message['results']
                    self._display_backtest_results()
                    
                elif message['type'] == 'error':
                    messagebox.showerror("Backtest Error", message['message'])
                    self.backtest_status_label.config(text="Backtest failed")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._monitor_backtest)
    
    def _display_backtest_results(self):
        """Display backtest results in GUI"""
        if not self.backtest_results:
            return
        
        results = self.backtest_results
        
        # Update all result displays
        self.bt_total_trades_label.config(text=str(results.total_trades))
        self.bt_win_rate_label.config(
            text=f"{results.win_rate:.2%}",
            foreground=self.success_color if results.win_rate > 0.5 else self.danger_color
        )
        self.bt_total_pnl_label.config(
            text=f"${results.total_pnl:.2f}",
            foreground=self.success_color if results.total_pnl >= 0 else self.danger_color
        )
        self.bt_roi_label.config(
            text=f"{results.roi_percentage:.2f}%",
            foreground=self.success_color if results.roi_percentage >= 0 else self.danger_color
        )
        self.bt_max_drawdown_label.config(text=f"{results.max_drawdown:.2f}%")
        self.bt_sharpe_ratio_label.config(text=f"{results.sharpe_ratio:.3f}")
        self.bt_total_fees_label.config(text=f"${results.total_fees:.2f}")
        self.bt_avg_duration_label.config(text=results.avg_trade_duration)
        
        self.backtest_status_label.config(text="Backtest completed successfully!")
        self.progress_var.set(100)

def main():
    """Main entry point"""
    setup_logging(logging.INFO)
    
    # Start GUI with backtest functionality
    app = BacktestGUI()
    app.run()

if __name__ == "__main__":
    main()
