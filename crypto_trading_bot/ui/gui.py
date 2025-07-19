# ui/gui.py - GUI interface for the trading bot
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import pandas as pd

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.core.bot import TradingBot
from crypto_trading_bot.database.models import DatabaseManager
from crypto_trading_bot.utils.binance_data import BinanceDataFetcher

logger = logging.getLogger(__name__)

class GUI:
    """Graphical user interface for the trading bot"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot = None
        self.bot_thread = None
        self.running = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Cryptocurrency Trading Bot")
        self.root.geometry("1000x700") # Increased size for charts
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create database manager
        self.db_manager = DatabaseManager(settings.get_database_config().get('path', 'trading_bot.db'))
        self.binance_data = BinanceDataFetcher(
            api_key=self.settings.get_api_key(),
            api_secret=self.settings.get_api_secret(),
            testnet=self.settings.get_binance_config().get("testnet", True)
        )
        
        # Setup UI
        self.setup_ui()
        
        # Start update timer
        self.update_status()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_trades_tab()
        self.create_charts_tab() # New charts tab
        self.create_config_tab()
        self.create_logs_tab()
    
    def create_dashboard_tab(self):
        """Create dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Status section
        status_frame = ttk.LabelFrame(dashboard_frame, text="Bot Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: Stopped", font=("Arial", 12, "bold"))
        self.status_label.pack(anchor=tk.W)
        
        self.strategy_label = ttk.Label(status_frame, text="Strategy: None")
        self.strategy_label.pack(anchor=tk.W)
        
        self.symbols_label = ttk.Label(status_frame, text="Symbols: None")
        self.symbols_label.pack(anchor=tk.W)
        
        # Control buttons
        control_frame = ttk.Frame(dashboard_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Bot", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Performance section
        perf_frame = ttk.LabelFrame(dashboard_frame, text="Performance", padding=10)
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for performance data
        columns = ("Metric", "Value")
        self.perf_tree = ttk.Treeview(perf_frame, columns=columns, show="headings", height=8)
        self.perf_tree.heading("Metric", text="Metric")
        self.perf_tree.heading("Value", text="Value")
        self.perf_tree.column("Metric", width=200)
        self.perf_tree.column("Value", width=150)
        self.perf_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for performance tree
        perf_scrollbar = ttk.Scrollbar(perf_frame, orient=tk.VERTICAL, command=self.perf_tree.yview)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.perf_tree.configure(yscrollcommand=perf_scrollbar.set)
    
    def create_trades_tab(self):
        """Create trades tab"""
        trades_frame = ttk.Frame(self.notebook)
        self.notebook.add(trades_frame, text="Trades")
        
        # Controls
        controls_frame = ttk.Frame(trades_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_var = tk.StringVar()
        symbol_entry = ttk.Entry(controls_frame, textvariable=self.symbol_var, width=10)
        symbol_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Refresh", command=self.refresh_trades).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export CSV", command=self.export_trades).pack(side=tk.LEFT, padx=5)
        
        # Trades table
        columns = ("Date", "Symbol", "Side", "Quantity", "Price", "P&L", "Strategy")
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=100)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        trades_scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
        trades_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trades_tree.configure(yscrollcommand=trades_scrollbar.set)
        
        # Load initial trades
        self.refresh_trades()

    def create_charts_tab(self):
        """Create charts tab"""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="Charts")

        # Controls for chart
        chart_controls_frame = ttk.Frame(charts_frame)
        chart_controls_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(chart_controls_frame, text="Symbol:").pack(side=tk.LEFT)
        self.chart_symbol_var = tk.StringVar(value="BTCUSDT") # Default symbol
        self.chart_symbol_entry = ttk.Entry(chart_controls_frame, textvariable=self.chart_symbol_var, width=10)
        self.chart_symbol_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(chart_controls_frame, text="Interval:").pack(side=tk.LEFT)
        self.chart_interval_var = tk.StringVar(value="1h") # Default interval
        self.chart_interval_combo = ttk.Combobox(chart_controls_frame, textvariable=self.chart_interval_var,
                                                values=["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"])
        self.chart_interval_combo.pack(side=tk.LEFT, padx=5)
        self.chart_interval_combo.set("1h") # Set default value

        ttk.Button(chart_controls_frame, text="Refresh Chart", command=self.update_chart).pack(side=tk.LEFT, padx=5)

        # Chart display area
        self.chart_frame = ttk.Frame(charts_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax = None # Initialize ax to None

        # Initial chart load
        self.update_chart()
    
    def create_config_tab(self):
        """Create configuration tab"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")
        
        # Create scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # API Configuration
        api_frame = ttk.LabelFrame(scrollable_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W)
        self.api_key_var = tk.StringVar(value=self.settings.get_api_key())
        ttk.Entry(api_frame, textvariable=self.api_key_var, width=50).grid(row=0, column=1, padx=5)
        
        ttk.Label(api_frame, text="API Secret:").grid(row=1, column=0, sticky=tk.W)
        self.api_secret_var = tk.StringVar(value=self.settings.get_api_secret())
        ttk.Entry(api_frame, textvariable=self.api_secret_var, show="*", width=50).grid(row=1, column=1, padx=5)
        
        self.testnet_var = tk.BooleanVar(value=self.settings.get_binance_config().get("testnet", True))
        ttk.Checkbutton(api_frame, text="Use Testnet", variable=self.testnet_var).grid(row=2, column=1, sticky=tk.W)
        
        # Trading Configuration
        trading_frame = ttk.LabelFrame(scrollable_frame, text="Trading Configuration", padding=10)
        trading_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(trading_frame, text="Symbols:").grid(row=0, column=0, sticky=tk.W)
        trading_config = self.settings.get_trading_config()
        self.symbols_var = tk.StringVar(value=",".join(trading_config.get("symbols", ["BTCUSDT"])))
        ttk.Entry(trading_frame, textvariable=self.symbols_var, width=50).grid(row=0, column=1, padx=5)
        
        ttk.Label(trading_frame, text="Investment Amount ($):").grid(row=1, column=0, sticky=tk.W)
        self.investment_var = tk.DoubleVar(value=trading_config.get("investment_amount", 100.0))
        ttk.Entry(trading_frame, textvariable=self.investment_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(trading_frame, text="Max Daily Loss ($):").grid(row=2, column=0, sticky=tk.W)
        self.max_loss_var = tk.DoubleVar(value=trading_config.get("max_daily_loss", 0.05))
        ttk.Entry(trading_frame, textvariable=self.max_loss_var, width=20).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Strategy Configuration
        strategy_frame = ttk.LabelFrame(scrollable_frame, text="Strategy Configuration", padding=10)
        strategy_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(strategy_frame, text="Strategy:").grid(row=0, column=0, sticky=tk.W)
        strategy_config = self.settings.get_strategy_config()
        self.strategy_var = tk.StringVar(value=strategy_config.get("name", "moving_average_crossover"))
        strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, 
                                    values=["moving_average_crossover", "rsi_strategy", "bollinger_bands"])
        strategy_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Save button
        save_frame = ttk.Frame(scrollable_frame)
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(save_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT)
        ttk.Button(save_frame, text="Validate Configuration", command=self.validate_config).pack(side=tk.LEFT, padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_logs_tab(self):
        """Create logs tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Log text area
        self.log_text = tk.Text(logs_frame, height=20, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar for logs
        log_scrollbar = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Clear logs button
        ttk.Button(logs_frame, text="Clear Logs", command=self.clear_logs).pack(pady=5)
    
    def start_bot(self):
        """Start the trading bot"""
        if not self.validate_config():
            messagebox.showerror("Error", "Configuration is invalid. Please fix the errors.")
            return
        
        try:
            self.bot = TradingBot(self.settings)
            self.bot_thread = threading.Thread(target=self.bot.start, daemon=True)
            self.bot_thread.start()
            
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")
            
            messagebox.showinfo("Success", "Trading bot started successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start bot: {e}")
            logger.error(f"Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the trading bot"""
        if self.bot:
            self.bot.stop()
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")
            
            messagebox.showinfo("Success", "Trading bot stopped successfully!")
    
    def refresh_trades(self):
        """Refresh trades display"""
        # Clear existing items
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
        
        # Get trades
        symbol = self.symbol_var.get().strip() if self.symbol_var.get() else None
        trades = self.db_manager.get_trades(symbol=symbol, limit=100)
        
        # Add trades to treeview
        for trade in trades:
            date_str = trade['timestamp'][:19] if isinstance(trade['timestamp'], str) else str(trade['timestamp'])[:19]
            self.trades_tree.insert("", "end", values=(
                date_str,
                trade['symbol'],
                trade['side'],
                f"{trade['quantity']:.6f}",
                f"{trade['price']:.2f}",
                f"{trade.get('pnl', 0):.2f}",
                trade['strategy']
            ))
    
    def update_chart(self):
        """Update the candlestick chart with live data"""
        symbol = self.chart_symbol_var.get().upper()
        interval = self.chart_interval_var.get()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30) # Fetch last 30 days of data

            df = self.binance_data.get_historical_klines(symbol, start_date, end_date, interval)
            
            if df.empty:
                messagebox.showinfo("No Data", f"No historical data found for {symbol} with interval {interval}.")
                return

            # Clear previous plot
            self.fig.clear()

            # Prepare kwargs for mplfinance
            plot_kwargs = dict(
                type='candle', style='yahoo', volume=True,
                title=f"{symbol} {interval} Candlestick Chart",
                figscale=1.0, figratio=(10,6), panel_ratios=(3,1), 
                datetime_format='%Y-%m-%d %H:%M', 
                xrotation=0, update_width_config=dict(candle_linewidth=0.6, candle_width=0.5)
            )

            mpf.plot(df, **plot_kwargs)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to load chart data: {e}")
            logger.error(f"Failed to load chart data: {e}")
    
    def export_trades(self):
        """Export trades to CSV"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        
        if filepath:
            symbol = self.symbol_var.get().strip() if self.symbol_var.get() else None
            try:
                self.db_manager.export_trades_csv(filepath, symbol)
                messagebox.showinfo("Success", f"Trades exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export trades: {e}")
    
    def save_config(self):
        """Save configuration"""
        try:
            # Update settings from GUI
            self.settings.update_config("binance_config", "api_key", self.api_key_var.get())
            self.settings.update_config("binance_config", "api_secret", self.api_secret_var.get())
            self.settings.update_config("binance_config", "testnet", self.testnet_var.get())
            
            self.settings.update_config("trading_config", "symbols", [s.strip() for s in self.symbols_var.get().split(",")])
            self.settings.update_config("trading_config", "investment_amount", self.investment_var.get())
            self.settings.update_config("trading_config", "max_daily_loss", self.max_loss_var.get())
            
            self.settings.update_config("strategy_config", "name", self.strategy_var.get())
            
            # Save to file
            self.settings.save_config()
            messagebox.showinfo("Success", "Configuration saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def validate_config(self):
        """Validate configuration"""
        try:
            if self.settings.validate_config():
                messagebox.showinfo("Success", "Configuration is valid!")
                return True
            else:
                messagebox.showerror("Error", "Configuration has errors. Please check the settings.")
                return False
        except Exception as e:
            messagebox.showerror("Error", f"Validation failed: {e}")
            return False
    
    def update_status(self):
        """Update status display"""
        try:
            # Update bot status
            if self.bot and self.running:
                status = self.bot.get_status()
                self.strategy_label.config(text=f"Strategy: {status.get('strategy', 'Unknown')}")
                self.symbols_label.config(text=f"Symbols: {', '.join(status.get('symbols', []))}")
                
                # Update performance data
            self.update_performance_display(status)
            
            # Schedule next update
            self.root.after(5000, self.update_status)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def update_performance_display(self, status: Dict):
        """Update performance display"""
        try:
            # Clear existing items
            for item in self.perf_tree.get_children():
                self.perf_tree.delete(item)
            
            # Get performance data
            performance = self.db_manager.get_performance_summary(days=7)
            db_stats = self.db_manager.get_database_stats()
            
            # Add performance data
            perf_data = [
                ("Total Trades", str(status.get('total_trades', 0))),
                ("Total Invested", f"${status.get('total_invested', 0):.2f}"),
                ("Current Portfolio Value", f"${status.get('current_portfolio_value', 0):.2f}"),
                ("Total P&L", f"${status.get('total_pnl', 0):.2f}"),
                ("Daily P&L", f"${status.get('daily_pnl', 0):.2f}"),
                ("Win Rate", f"{performance.get('win_rate', 0):.1%}"),
                ("Average Win", f"${performance.get('avg_win', 0):.2f}"),
                ("Average Loss", f"${performance.get('avg_loss', 0):.2f}"),
                ("Database Size", f"{db_stats.get('database_size_mb', 0)} MB"),
                ("Last Update", datetime.now().strftime("%H:%M:%S"))
            ]
            
            for metric, value in perf_data:
                self.perf_tree.insert("", "end", values=(metric, value))
                
        except Exception as e:
            logger.error(f"Error updating performance display: {e}")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def add_log_message(self, message: str):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            if messagebox.askokcancel("Quit", "Bot is running. Do you want to stop it and quit?"):
                self.stop_bot()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()