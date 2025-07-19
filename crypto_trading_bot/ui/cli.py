# ui/cli.py - Command-line interface for the trading bot
import logging
import sys
from typing import Dict, List, Any
import click
import getpass

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.database.models import DatabaseManager
from crypto_trading_bot.core.strategies import StrategyFactory

logger = logging.getLogger(__name__)

class CLI:
    """Command-line interface for the trading bot"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_manager = DatabaseManager(settings.get_database_config().get('path', 'trading_bot.db'))
        self.strategy_factory = StrategyFactory()
    
    def configure(self):
        """Interactive configuration setup"""
        print("=== Cryptocurrency Trading Bot Configuration ===\n")
        
        # Configure API credentials
        self._configure_api_credentials()
        
        # Configure trading parameters
        self._configure_trading()
        
        # Configure strategy
        self._configure_strategy()
        
        # Configure backtesting
        self._configure_backtesting()
        
        # Configure notifications
        self._configure_notifications()
        
        print("\nConfiguration completed!")
    
    def _configure_api_credentials(self):
        """Configure Binance API credentials"""
        print("1. Binance API Configuration")
        print("-" * 30)
        
        use_testnet = input("Use Binance Testnet? (y/n, default: y): ").lower() != 'n'
        
        api_key = input("Enter your Binance API Key (or press Enter to skip): ").strip()
        api_secret = ""
        
        if api_key:
            api_secret = getpass.getpass("Enter your Binance API Secret: ").strip()
            
            if api_key and api_secret:
                self.settings.set_api_credentials(api_key, api_secret)
                print("✓ API credentials saved and encrypted")
            else:
                print("⚠ API credentials not provided")
        
        self.settings.update_config("binance_config", "testnet", use_testnet)
        print(f"✓ Testnet setting: {'Enabled' if use_testnet else 'Disabled'}")
        print()
    
    def _configure_trading(self):
        """Configure trading parameters"""
        print("2. Trading Configuration")
        print("-" * 30)
        
        # Trading symbols
        symbols_input = input("Enter trading symbols (comma-separated, default: BTCUSDT,ETHUSDT): ").strip()
        if symbols_input:
            symbols = [s.strip() for s in symbols_input.split(",")]
        else:
            symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Investment amount
        investment_input = input("Enter investment amount per trade (default: 100.0): ").strip()
        investment_amount = float(investment_input) if investment_input else 100.0
        
        # Stop loss
        stop_loss_input = input("Enter stop loss percentage (default: 0.05): ").strip()
        stop_loss = float(stop_loss_input) if stop_loss_input else 0.05
        
        # Take profit
        take_profit_input = input("Enter take profit percentage (default: 0.10): ").strip()
        take_profit = float(take_profit_input) if take_profit_input else 0.10
        
        # Max drawdown
        max_drawdown_input = input("Enter maximum drawdown percentage (default: 0.20): ").strip()
        max_drawdown = float(max_drawdown_input) if max_drawdown_input else 0.20
        
        # Update configuration
        self.settings.update_config("trading_config", "symbols", symbols)
        self.settings.update_config("trading_config", "investment_amount", investment_amount)
        self.settings.update_config("trading_config", "stop_loss", stop_loss)
        self.settings.update_config("trading_config", "take_profit", take_profit)
        self.settings.update_config("trading_config", "max_drawdown", max_drawdown)
        
        print(f"✓ Trading symbols: {', '.join(symbols)}")
        print(f"✓ Investment amount: ${investment_amount}")
        print(f"✓ Stop loss: {stop_loss:.1%}")
        print(f"✓ Take profit: {take_profit:.1%}")
        print(f"✓ Max drawdown: {max_drawdown:.1%}")
        print()
    
    def _configure_strategy(self):
        """Configure trading strategy"""
        print("3. Strategy Configuration")
        print("-" * 30)
        
        strategies = ["moving_average_crossover", "rsi_strategy", "bollinger_bands", "random_strategy"]
        
        print("Available strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
        
        strategy_choice = input(f"Select strategy (1-{len(strategies)}, default: 1): ").strip()
        
        try:
            strategy_index = int(strategy_choice) - 1 if strategy_choice else 0
            if 0 <= strategy_index < len(strategies):
                strategy_name = strategies[strategy_index]
            else:
                strategy_name = strategies[0]
        except ValueError:
            strategy_name = strategies[0]
        
        # Configure strategy parameters based on selection
        parameters = self._configure_strategy_parameters(strategy_name)
        
        self.settings.update_config("strategy_config", "name", strategy_name)
        self.settings.update_config("strategy_config", "parameters", parameters)
        
        print(f"✓ Selected strategy: {strategy_name}")
        print()
    
    def _configure_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Configure parameters for specific strategy"""
        if strategy_name == "moving_average_crossover":
            short_window = input("Enter short MA window (default: 5): ").strip()
            long_window = input("Enter long MA window (default: 15): ").strip()
            
            return {
                "short_window": int(short_window) if short_window else 5,
                "long_window": int(long_window) if long_window else 15,
                "min_crossover_strength": 0.0001
            }
        
        elif strategy_name == "rsi_strategy":
            rsi_period = input("Enter RSI period (default: 7): ").strip()
            overbought = input("Enter overbought threshold (default: 80): ").strip()
            oversold = input("Enter oversold threshold (default: 20): ").strip()
            
            return {
                "rsi_period": int(rsi_period) if rsi_period else 7,
                "rsi_overbought": int(overbought) if overbought else 80,
                "rsi_oversold": int(oversold) if oversold else 20,
                "confirmation_periods": 0
            }
        
        elif strategy_name == "bollinger_bands":
            period = input("Enter BB period (default: 20): ").strip()
            std_dev = input("Enter standard deviation (default: 2): ").strip()
            
            return {
                "period": int(period) if period else 20,
                "std_dev": int(std_dev) if std_dev else 2,
                "min_breakout_strength": 0.001
            }
        
        elif strategy_name == "random_strategy":
            probability = input("Enter signal probability (default: 0.01): ").strip()
            
            return {
                "signal_probability": float(probability) if probability else 0.01
            }
        
        return {}
    
    def _configure_backtesting(self):
        """Configure backtesting settings"""
        print("4. Backtesting Configuration")
        print("-" * 30)
        
        # Data source choice
        use_real_data = input("Use real Binance data for backtesting? (y/n, default: n): ").lower() == 'y'
        
        if use_real_data:
            print("Note: You need valid Binance API credentials to use real data")
        
        # Data interval
        intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
        print("Available intervals:")
        for i, interval in enumerate(intervals, 1):
            print(f"  {i}. {interval}")
        
        interval_choice = input(f"Select interval (1-{len(intervals)}, default: 6 for 1h): ").strip()
        
        try:
            interval_index = int(interval_choice) - 1 if interval_choice else 5
            if 0 <= interval_index < len(intervals):
                data_interval = intervals[interval_index]
            else:
                data_interval = "1h"
        except ValueError:
            data_interval = "1h"
        
        # Fallback option
        fallback = input("Fallback to synthetic data if real data fails? (y/n, default: y): ").lower() != 'n'
        
        self.settings.update_config("backtest_config", "use_real_data", use_real_data)
        self.settings.update_config("backtest_config", "data_interval", data_interval)
        self.settings.update_config("backtest_config", "fallback_to_synthetic", fallback)
        
        print(f"✓ Data source: {'Real Binance data' if use_real_data else 'Synthetic data'}")
        print(f"✓ Data interval: {data_interval}")
        print(f"✓ Fallback to synthetic: {'Enabled' if fallback else 'Disabled'}")
        print()
    
    def _configure_notifications(self):
        """Configure notification settings"""
        print("5. Notification Configuration")
        print("-" * 30)
        
        # Email notifications
        email_enabled = input("Enable email notifications? (y/n, default: n): ").lower() == 'y'
        
        if email_enabled:
            email = input("Enter email address: ").strip()
            self.settings.update_config("notification_config", "email", email)
            print(f"✓ Email notifications enabled for: {email}")
        else:
            self.settings.update_config("notification_config", "email", "")
        
        self.settings.update_config("notification_config", "email_enabled", email_enabled)
        
        # Telegram notifications
        telegram_enabled = input("Enable Telegram notifications? (y/n, default: n): ").lower() == 'y'
        
        if telegram_enabled:
            bot_token = input("Enter Telegram bot token: ").strip()
            chat_id = input("Enter Telegram chat ID: ").strip()
            
            self.settings.update_config("notification_config", "telegram_bot_token", bot_token)
            self.settings.update_config("notification_config", "telegram_chat_id", chat_id)
            print(f"✓ Telegram notifications enabled")
        else:
            self.settings.update_config("notification_config", "telegram_bot_token", "")
            self.settings.update_config("notification_config", "telegram_chat_id", "")
        
        self.settings.update_config("notification_config", "telegram_enabled", telegram_enabled)
        print()
    
    def show_config(self):
        """Display current configuration"""
        print("=== Current Configuration ===\n")
        
        # API Configuration
        binance_config = self.settings.get_binance_config()
        print("Binance API:")
        print(f"  Testnet: {binance_config.get('testnet', True)}")
        print(f"  API Key: {'✓ Set' if self.settings.get_api_key() else '✗ Not set'}")
        print(f"  API Secret: {'✓ Set' if self.settings.get_api_secret() else '✗ Not set'}")
        print()
        
        # Trading Configuration
        trading_config = self.settings.get_trading_config()
        print("Trading:")
        print(f"  Symbols: {', '.join(trading_config.get('symbols', []))}")
        print(f"  Investment Amount: ${trading_config.get('investment_amount', 0)}")
        print(f"  Stop Loss: {trading_config.get('stop_loss', 0):.1%}")
        print(f"  Take Profit: {trading_config.get('take_profit', 0):.1%}")
        print(f"  Max Drawdown: {trading_config.get('max_drawdown', 0):.1%}")
        print()
        
        # Strategy Configuration
        strategy_config = self.settings.get_strategy_config()
        print("Strategy:")
        print(f"  Name: {strategy_config.get('name', 'None')}")
        print(f"  Parameters: {strategy_config.get('parameters', {})}")
        print()
        
        # Backtest Configuration
        backtest_config = self.settings.get_backtest_config()
        print("Backtesting:")
        print(f"  Use Real Data: {backtest_config.get('use_real_data', False)}")
        print(f"  Data Interval: {backtest_config.get('data_interval', '1h')}")
        print(f"  Fallback to Synthetic: {backtest_config.get('fallback_to_synthetic', True)}")
        print()
        
        # Notification Configuration
        notification_config = self.settings.get_notification_config()
        print("Notifications:")
        print(f"  Email: {'✓ Enabled' if notification_config.get('email_enabled', False) else '✗ Disabled'}")
        print(f"  Telegram: {'✓ Enabled' if notification_config.get('telegram_enabled', False) else '✗ Disabled'}")
        print()
    
    def show_status(self):
        """Show current bot status and configuration"""
        click.echo("🤖 Trading Bot Status")
        click.echo("=" * 30)
        
        # Configuration status
        click.echo(f"API Configured: {'✅' if self.settings.get_api_key() else '❌'}")
        click.echo(f"Testnet Mode: {'✅' if self.settings.get_binance_config().get('testnet', False) else '❌'}")
        click.echo(f"Strategy: {self.settings.get_strategy_config().get('name', 'Unknown')}")
        click.echo(f"Trading Symbols: {', '.join(self.settings.get_trading_config().get('symbols', []))}")
        click.echo(f"Investment Amount: ${self.settings.get_trading_config().get('investment_amount', 0)}")
        
        # Database stats
        db_stats = self.db_manager.get_database_stats()
        click.echo(f"\n📊 Database Statistics:")
        click.echo(f"Total Trades: {db_stats.get('trades_count', 0)}")
        click.echo(f"Total Orders: {db_stats.get('orders_count', 0)}")
        click.echo(f"Database Size: {db_stats.get('database_size_mb', 0)} MB")
        
        # Recent performance
        performance = self.db_manager.get_performance_summary(days=7)
        if performance:
            click.echo(f"\n📈 Last 7 Days Performance:")
            click.echo(f"Total Trades: {performance.get('total_trades', 0)}")
            click.echo(f"Win Rate: {performance.get('win_rate', 0):.1%}")
            click.echo(f"Total P&L: ${performance.get('total_pnl', 0):.2f}")
    
    def show_trades(self, symbol: str = None, limit: int = 10):
        """Show recent trades"""
        trades = self.db_manager.get_trades(symbol=symbol, limit=limit)
        
        if not trades:
            click.echo("No trades found.")
            return
        
        click.echo(f"📋 Recent Trades (Last {len(trades)}):")
        click.echo("-" * 80)
        click.echo(f"{'Date':<20} {'Symbol':<10} {'Side':<6} {'Quantity':<12} {'Price':<10} {'P&L':<10}")
        click.echo("-" * 80)
        
        for trade in trades:
            date_str = trade['timestamp'][:19] if isinstance(trade['timestamp'], str) else str(trade['timestamp'])[:19]
            click.echo(
                f"{date_str:<20} {trade['symbol']:<10} {trade['side']:<6} "
                f"{trade['quantity']:<12.6f} {trade['price']:<10.2f} {trade.get('pnl', 0):<10.2f}"
            )
    
    def export_trades(self, filepath: str, symbol: str = None):
        """Export trades to CSV"""
        try:
            self.db_manager.export_trades_csv(filepath, symbol)
            click.echo(f"✅ Trades exported to {filepath}")
        except Exception as e:
            click.echo(f"❌ Error exporting trades: {e}")
    
    def show_orders(self, symbol: str = None, status: str = None):
        """Show recent orders"""
        orders = self.db_manager.get_orders(symbol=symbol, status=status)
        
        if not orders:
            click.echo("No orders found.")
            return
        
        click.echo(f"📋 Recent Orders (Last {len(orders)}):")
        click.echo("-" * 90)
        click.echo(f"{'Date':<20} {'Symbol':<10} {'Side':<6} {'Type':<8} {'Quantity':<12} {'Price':<10} {'Status':<10}")
        click.echo("-" * 90)
        
        for order in orders:
            date_str = order['timestamp'][:19] if isinstance(order['timestamp'], str) else str(order['timestamp'])[:19]
            click.echo(
                f"{date_str:<20} {order['symbol']:<10} {order['side']:<6} "
                f"{order['order_type']:<8} {order['quantity']:<12.6f} "
                f"{order.get('price', 'N/A'):<10} {order['status']:<10}"
            )
    
    def validate_config(self):
        """Validate current configuration"""
        click.echo("🔍 Validating Configuration...")
        
        if self.settings.validate_config():
            click.echo("✅ Configuration is valid!")
        else:
            click.echo("❌ Configuration has errors. Please run 'configure' to fix them.")
            return False
        
        return True
