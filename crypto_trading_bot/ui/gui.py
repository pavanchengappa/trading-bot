# ui/gui.py - Enhanced GUI with portfolio management display
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import json
from datetime import datetime
from typing import Dict, Optional

from crypto_trading_bot.core.bot import TradingBot
from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

class TradingBotGUI:
    """Enhanced GUI for cryptocurrency trading bot with portfolio display"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crypto Trading Bot - Portfolio Manager")
        self.root.geometry("1200x800")
        
        # Apply modern dark theme
        self.apply_dark_theme()
        
        self.bot = None
        self.bot_thread = None
        self.update_queue = queue.Queue()
        self.settings = Settings()
        
        # Create GUI components
        self._create_widgets()
        
        # Start update loop
        self._update_gui()
        
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.button_bg = "#0d7377"
        self.button_fg = "#ffffff"
        self.entry_bg = "#2d2d2d"
        self.entry_fg = "#ffffff"
        self.success_color = "#28a745"
        self.danger_color = "#dc3545"
        self.warning_color = "#ffc107"
        self.info_color = "#17a2b8"
        
        self.root.configure(bg=self.bg_color)
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("Dark.TFrame", background=self.bg_color)
        style.configure("Dark.TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("Dark.TButton", background=self.button_bg, foreground=self.button_fg)
        style.configure("Success.TLabel", background=self.bg_color, foreground=self.success_color)
        style.configure("Danger.TLabel", background=self.bg_color, foreground=self.danger_color)
        style.configure("Info.TLabel", background=self.bg_color, foreground=self.info_color)
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_dashboard_tab()
        self._create_positions_tab()
        self._create_performance_tab()
        self._create_config_tab()
        self._create_log_tab()
        
    def _create_dashboard_tab(self):
        """Create main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Control panel
        control_frame = ttk.Frame(dashboard_frame, style="Dark.TFrame")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_button = tk.Button(
            control_frame, text="Start Bot", command=self.start_bot,
            bg=self.success_color, fg=self.button_fg, font=('Arial', 12, 'bold'),
            padx=20, pady=10
        )
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = tk.Button(
            control_frame, text="Stop Bot", command=self.stop_bot,
            bg=self.danger_color, fg=self.button_fg, font=('Arial', 12, 'bold'),
            padx=20, pady=10, state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # Portfolio overview frame
        portfolio_frame = ttk.LabelFrame(dashboard_frame, text="Portfolio Overview", style="Dark.TFrame")
        portfolio_frame.pack(fill='x', padx=10, pady=5)
        
        # Portfolio metrics grid
        metrics_frame = ttk.Frame(portfolio_frame, style="Dark.TFrame")
        metrics_frame.pack(fill='x', padx=10, pady=10)
        
        # Row 1 - Main metrics
        self._create_metric_display(metrics_frame, "Initial Investment", "$0.00", 0, 0, "initial_investment")
        self._create_metric_display(metrics_frame, "Current Value", "$0.00", 0, 1, "current_value")
        self._create_metric_display(metrics_frame, "Total P&L", "$0.00", 0, 2, "total_pnl")
        self._create_metric_display(metrics_frame, "ROI %", "0.00%", 0, 3, "roi_percentage")
        
        # Row 2 - Fund allocation
        self._create_metric_display(metrics_frame, "Available Funds", "$0.00", 1, 0, "available_funds")
        self._create_metric_display(metrics_frame, "Allocated Funds", "$0.00", 1, 1, "allocated_funds")
        self._create_metric_display(metrics_frame, "Max Trade Size", "$0.00", 1, 2, "max_trade_size")
        self._create_metric_display(metrics_frame, "Active Positions", "0", 1, 3, "active_positions")
        
        # Trading metrics frame
        trading_frame = ttk.LabelFrame(dashboard_frame, text="Trading Metrics", style="Dark.TFrame")
        trading_frame.pack(fill='x', padx=10, pady=5)
        
        trading_metrics = ttk.Frame(trading_frame, style="Dark.TFrame")
        trading_metrics.pack(fill='x', padx=10, pady=10)
        
        self._create_metric_display(trading_metrics, "Total Trades", "0", 0, 0, "total_trades")
        self._create_metric_display(trading_metrics, "Daily P&L", "$0.00", 0, 1, "daily_pnl")
        self._create_metric_display(trading_metrics, "Bot Status", "Stopped", 0, 2, "bot_status")
        self._create_metric_display(trading_metrics, "Active Symbols", "0", 0, 3, "active_symbols")
        
        # Market scanner frame
        scanner_frame = ttk.LabelFrame(dashboard_frame, text="Market Scanner", style="Dark.TFrame")
        scanner_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for market opportunities
        columns = ('Symbol', 'Price', 'Volume 24h', 'Volatility', 'Score', 'Signal')
        self.market_tree = ttk.Treeview(scanner_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=150)
        
        market_scroll = ttk.Scrollbar(scanner_frame, orient='vertical', command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=market_scroll.set)
        
        self.market_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        market_scroll.pack(side='right', fill='y', padx=(0, 10), pady=10)
        
    def _create_positions_tab(self):
        """Create positions management tab"""
        positions_frame = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(positions_frame, text="Positions")
        
        # Current positions
        current_frame = ttk.LabelFrame(positions_frame, text="Current Positions", style="Dark.TFrame")
        current_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('Symbol', 'Side', 'Quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Duration')
        self.positions_tree = ttk.Treeview(current_frame, columns=columns, show='headings')
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=120)
        
        pos_scroll = ttk.Scrollbar(current_frame, orient='vertical', command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scroll.set)
        
        self.positions_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        pos_scroll.pack(side='right', fill='y', padx=(0, 10), pady=10)
        
        # Position controls
        control_frame = ttk.Frame(positions_frame, style="Dark.TFrame")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            control_frame, text="Close Position", command=self.close_position,
            bg=self.warning_color, fg=self.button_fg, font=('Arial', 10)
        ).pack(side='left', padx=5)
        
        tk.Button(
            control_frame, text="Close All Positions", command=self.close_all_positions,
            bg=self.danger_color, fg=self.button_fg, font=('Arial', 10)
        ).pack(side='left', padx=5)
        
    def _create_performance_tab(self):
        """Create performance analytics tab"""
        performance_frame = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(performance_frame, text="Performance")
        
        # Symbol performance
        symbol_frame = ttk.LabelFrame(performance_frame, text="Symbol Performance", style="Dark.TFrame")
        symbol_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('Symbol', 'Trades', 'Total P&L', 'Win Rate', 'Avg Win', 'Avg Loss', 'Score')
        self.symbol_tree = ttk.Treeview(symbol_frame, columns=columns, show='headings')
        
        for col in columns:
            self.symbol_tree.heading(col, text=col)
            self.symbol_tree.column(col, width=120)
        
        symbol_scroll = ttk.Scrollbar(symbol_frame, orient='vertical', command=self.symbol_tree.yview)
        self.symbol_tree.configure(yscrollcommand=symbol_scroll.set)
        
        self.symbol_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        symbol_scroll.pack(side='right', fill='y', padx=(0, 10), pady=10)
        
        # Recent trades
        trades_frame = ttk.LabelFrame(performance_frame, text="Recent Trades", style="Dark.TFrame")
        trades_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('Time', 'Symbol', 'Side', 'Quantity', 'Price', 'P&L', 'Score')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=120)
        
        trades_scroll = ttk.Scrollbar(trades_frame, orient='vertical', command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=trades_scroll.set)
        
        self.trades_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        trades_scroll.pack(side='right', fill='y', padx=(0, 10), pady=10)
        
    def _create_config_tab(self):
        """Create configuration tab"""
        config_frame = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(config_frame, text="Configuration")
        
        # Portfolio settings
        portfolio_settings = ttk.LabelFrame(config_frame, text="Portfolio Settings", style="Dark.TFrame")
        portfolio_settings.pack(fill='x', padx=10, pady=10)
        
        settings_grid = ttk.Frame(portfolio_settings, style="Dark.TFrame")
        settings_grid.pack(padx=10, pady=10)
        
        # Initial investment
        ttk.Label(settings_grid, text="Initial Investment ($):", style="Dark.TLabel").grid(
            row=0, column=0, sticky='w', padx=5, pady=5
        )
        self.initial_investment_var = tk.StringVar(value="10000")
        tk.Entry(
            settings_grid, textvariable=self.initial_investment_var,
            bg=self.entry_bg, fg=self.entry_fg, font=('Arial', 10)
        ).grid(row=0, column=1, padx=5, pady=5)
        
        # Max allocation per trade
        ttk.Label(settings_grid, text="Max Allocation per Trade (%):", style="Dark.TLabel").grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        self.max_allocation_var = tk.StringVar(value="10")
        tk.Entry(
            settings_grid, textvariable=self.max_allocation_var,
            bg=self.entry_bg, fg=self.entry_fg, font=('Arial', 10)
        ).grid(row=1, column=1, padx=5, pady=5)
        
        # Min opportunity score
        ttk.Label(settings_grid, text="Min Opportunity Score:", style="Dark.TLabel").grid(
            row=2, column=0, sticky='w', padx=5, pady=5
        )
        self.min_score_var = tk.StringVar(value="60")
        tk.Entry(
            settings_grid, textvariable=self.min_score_var,
            bg=self.entry_bg, fg=self.entry_fg, font=('Arial', 10)
        ).grid(row=2, column=1, padx=5, pady=5)
        
        # Symbols selection
        symbols_frame = ttk.LabelFrame(config_frame, text="Trading Symbols", style="Dark.TFrame")
        symbols_frame.pack(fill='x', padx=10, pady=10)
        
        # Available symbols
        self.symbol_vars = {}
        common_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
            'XRPUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'LTCUSDT'
        ]
        
        symbol_grid = ttk.Frame(symbols_frame, style="Dark.TFrame")
        symbol_grid.pack(padx=10, pady=10)
        
        for i, symbol in enumerate(common_symbols):
            var = tk.BooleanVar(value=True)
            self.symbol_vars[symbol] = var
            tk.Checkbutton(
                symbol_grid, text=symbol, variable=var,
                bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg,
                font=('Arial', 10)
            ).grid(row=i//5, column=i%5, sticky='w', padx=5, pady=2)
        
        # Save button
        tk.Button(
            config_frame, text="Save Configuration", command=self.save_config,
            bg=self.success_color, fg=self.button_fg, font=('Arial', 12, 'bold'),
            padx=20, pady=10
        ).pack(pady=20)
        
    def _create_log_tab(self):
        """Create log viewer tab"""
        log_frame = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(log_frame, text="Logs")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=30,
            bg=self.entry_bg, fg=self.entry_fg, font=('Courier', 9)
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Log controls
        control_frame = ttk.Frame(log_frame, style="Dark.TFrame")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            control_frame, text="Clear Logs", command=self.clear_logs,
            bg=self.button_bg, fg=self.button_fg, font=('Arial', 10)
        ).pack(side='left', padx=5)
        
        tk.Button(
            control_frame, text="Export Logs", command=self.export_logs,
            bg=self.button_bg, fg=self.button_fg, font=('Arial', 10)
        ).pack(side='left', padx=5)
        
    def _create_metric_display(self, parent, label, value, row, col, key):
        """Create a metric display widget"""
        frame = ttk.Frame(parent, style="Dark.TFrame")
        frame.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
        
        ttk.Label(frame, text=label, style="Dark.TLabel", font=('Arial', 10)).pack()
        
        value_label = ttk.Label(frame, text=value, font=('Arial', 14, 'bold'))
        value_label.pack()
        
        # Store reference for updates
        setattr(self, f"{key}_label", value_label)
        
        # Color code based on metric type
        if 'pnl' in key or 'roi' in key:
            value_label.configure(style="Success.TLabel")
        elif 'status' in key:
            value_label.configure(style="Info.TLabel")
        else:
            value_label.configure(style="Dark.TLabel")
    
    def start_bot(self):
        """Start the trading bot"""
        if self.bot and self.bot.running:
            messagebox.showwarning("Warning", "Bot is already running!")
            return
        
        try:
            self.settings = Settings()
            self.bot = TradingBot(self.settings)
            
            # Start bot in separate thread
            self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
            self.bot_thread.start()
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.bot_status_label.config(text="Running", foreground=self.success_color)
            
            self.add_log("Bot started successfully", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start bot: {str(e)}")
            self.add_log(f"Failed to start bot: {str(e)}", "error")
    
    def stop_bot(self):
        """Stop the trading bot"""
        if not self.bot or not self.bot.running:
            messagebox.showwarning("Warning", "Bot is not running!")
            return
        
        try:
            self.bot.stop()
            
            # Update UI
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.bot_status_label.config(text="Stopped", foreground=self.danger_color)
            
            self.add_log("Bot stopped", "info")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop bot: {str(e)}")
            self.add_log(f"Failed to stop bot: {str(e)}", "error")
    
    def _run_bot(self):
        """Run bot in separate thread"""
        try:
            self.bot.start()
        except Exception as e:
            self.update_queue.put(('error', f"Bot error: {str(e)}"))
    
    def save_config(self):
        """Save configuration settings"""
        try:
            # Update trading config
            trading_config = self.settings.get_trading_config()
            trading_config['initial_investment'] = float(self.initial_investment_var.get())
            trading_config['max_allocation_per_trade'] = float(self.max_allocation_var.get()) / 100
            trading_config['min_opportunity_score'] = float(self.min_score_var.get())
            
            # Update symbols
            selected_symbols = [sym for sym, var in self.symbol_vars.items() if var.get()]
            trading_config['symbols'] = selected_symbols
            
            # Save configuration
            self.settings.config['trading_config'] = trading_config
            self.settings.save_config()
            
            messagebox.showinfo("Success", "Configuration saved successfully!")
            self.add_log("Configuration saved", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
            self.add_log(f"Failed to save configuration: {str(e)}", "error")
    
    def close_position(self):
        """Close selected position"""
        selection = self.positions_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a position to close")
            return
        
        position_key = selection[0]  # We use position_key as iid
        
        if self.bot and self.bot.running:
            if self.bot.close_position(position_key):
                self.add_log(f"Closed position {position_key}", "success")
                messagebox.showinfo("Success", "Position closed successfully")
            else:
                messagebox.showerror("Error", "Failed to close position")
        else:
            messagebox.showwarning("Warning", "Bot is not running")
    
    def close_all_positions(self):
        """Close all open positions"""
        if not messagebox.askyesno("Confirm", "Are you sure you want to close all positions?"):
            return
        
        if self.bot and self.bot.running:
            if self.bot.close_all_positions():
                self.add_log("Closed all positions", "success")
                messagebox.showinfo("Success", "All positions closed successfully")
            else:
                messagebox.showerror("Error", "Failed to close positions")
        else:
            messagebox.showwarning("Warning", "Bot is not running")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def export_logs(self):
        """Export logs to file"""
        # TODO: Implement log export
        messagebox.showinfo("Info", "Log export not yet implemented")
    
    def add_log(self, message, level="info"):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Color code based on level
        if level == "error":
            color = self.danger_color
        elif level == "success":
            color = self.success_color
        elif level == "warning":
            color = self.warning_color
        else:
            color = self.fg_color
        
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def _update_gui(self):
        """Update GUI with latest bot status"""
        try:
            # Process update queue
            while not self.update_queue.empty():
                msg_type, message = self.update_queue.get_nowait()
                if msg_type == 'error':
                    self.add_log(message, "error")
                else:
                    self.add_log(message, "info")
            
            # Update status if bot is running
            if self.bot and self.bot.running:
                status = self.bot.get_status()
                self._update_dashboard(status)
                self._update_positions(status)
                self._update_performance(status)
            
        except Exception as e:
            logger.error(f"Error updating GUI: {e}")
        
        # Schedule next update
        self.root.after(1000, self._update_gui)  # Update every second
    
    def _update_dashboard(self, status):
        """Update dashboard with latest status"""
        try:
            portfolio = status.get('portfolio', {})
            
            # Update portfolio metrics
            self.initial_investment_label.config(text=f"${portfolio.get('initial_investment', 0):,.2f}")
            self.current_value_label.config(text=f"${portfolio.get('current_value', 0):,.2f}")
            
            total_pnl = portfolio.get('total_pnl', 0)
            self.total_pnl_label.config(
                text=f"${total_pnl:,.2f}",
                foreground=self.success_color if total_pnl >= 0 else self.danger_color
            )
            
            roi = portfolio.get('roi_percentage', 0)
            self.roi_percentage_label.config(
                text=f"{roi:+.2f}%",
                foreground=self.success_color if roi >= 0 else self.danger_color
            )
            
            self.available_funds_label.config(text=f"${portfolio.get('available_funds', 0):,.2f}")
            self.allocated_funds_label.config(text=f"${portfolio.get('allocated_funds', 0):,.2f}")
            self.max_trade_size_label.config(text=f"${portfolio.get('max_trade_amount', 0):,.2f}")
            self.active_positions_label.config(text=str(status.get('active_positions', 0)))
            
            # Update trading metrics
            self.total_trades_label.config(text=str(status.get('total_trades', 0)))
            
            daily_pnl = status.get('daily_pnl', 0)
            self.daily_pnl_label.config(
                text=f"${daily_pnl:,.2f}",
                foreground=self.success_color if daily_pnl >= 0 else self.danger_color
            )
            
            self.active_symbols_label.config(text=str(len(status.get('symbols', []))))
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def _update_positions(self, status):
        """Update positions display"""
        try:
            # Clear current items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Add current positions
            positions = status.get('positions', [])
            for pos in positions:
                entry_price = pos.get('entry_price', 0)
                current_price = pos.get('current_price', entry_price)
                quantity = pos.get('quantity', 0)
                
                pnl = (current_price - entry_price) * quantity
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                entry_time = pos.get('entry_time', 'Unknown')
                if isinstance(entry_time, datetime):
                    duration = str(datetime.now() - entry_time).split('.')[0]
                else:
                    duration = "Unknown"
                
                values = (
                    pos.get('symbol', ''),
                    'LONG',  # Assuming long positions for now
                    f"{quantity:.6f}",
                    f"${entry_price:.2f}",
                    f"${current_price:.2f}",
                    f"${pnl:.2f}",
                    f"{pnl_percent:+.2f}%",
                    duration
                )
                
                # Color code based on P&L
                tag = 'profit' if pnl >= 0 else 'loss'
                # Use position_key as iid for easy retrieval
                self.positions_tree.insert('', 'end', iid=pos.get('position_key'), values=values, tags=(tag,))
            
            # Configure tags
            self.positions_tree.tag_configure('profit', foreground=self.success_color)
            self.positions_tree.tag_configure('loss', foreground=self.danger_color)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _update_performance(self, status):
        """Update performance displays"""
        try:
            # Update symbol performance
            symbol_perf = status.get('symbol_performance', {})
            
            # Clear current items
            for item in self.symbol_tree.get_children():
                self.symbol_tree.delete(item)
            
            # Add symbol performance data
            for symbol, perf in symbol_perf.items():
                if perf.get('trades', 0) > 0:
                    values = (
                        symbol,
                        perf.get('trades', 0),
                        f"${perf.get('pnl', 0):.2f}",
                        f"{perf.get('win_rate', 0)*100:.1f}%",
                        f"${0:.2f}",  # TODO: Calculate avg win
                        f"${0:.2f}",  # TODO: Calculate avg loss
                        f"{0:.1f}"    # TODO: Calculate score
                    )
                    
                    tag = 'profit' if perf.get('pnl', 0) >= 0 else 'loss'
                    self.symbol_tree.insert('', 'end', values=values, tags=(tag,))
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """Main entry point for GUI"""
    app = TradingBotGUI()
    app.run()

if __name__ == "__main__":
    main()