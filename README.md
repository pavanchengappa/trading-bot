# Cryptocurrency Trading Bot

A locally-deployed cryptocurrency trading bot that leverages the Binance API to automatically execute trades based on configurable strategies. Built with Python, this bot includes comprehensive risk management, backtesting capabilities, and both CLI and GUI interfaces.

## 🚀 Features

### Core Features
- **Binance API Integration**: Secure connection to Binance spot trading endpoints with rate limiting and error recovery
- **Multiple Trading Strategies**: 
  - Moving Average Crossover
  - RSI-based overbought/oversold
  - Bollinger Bands breakout
- **Automated Trading**: Scheduler to poll price data and execute orders automatically
- **Risk Management**: Stop-loss, take-profit, daily loss limits, and maximum drawdown controls
- **Transaction Recording**: SQLite database for storing all trades and performance data

### User Interface
- **CLI Interface**: Interactive command-line interface for configuration and monitoring
- **GUI Interface**: Lightweight Tkinter-based graphical interface
- **Real-time Monitoring**: Live status updates and performance tracking

### Advanced Features
- **Backtesting Module**: Test strategies on historical data
- **Notifications**: Email and Telegram alerts for trades and errors
- **Performance Analytics**: Comprehensive reporting and CSV export
- **Configuration Management**: Encrypted storage of API keys and settings

## 📋 Requirements

- Python 3.8 or higher
- Binance account with API access
- Stable internet connection
- Windows, macOS, or Linux

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd crypto-bot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Binance API**:
   - Create a Binance account
   - Generate API key and secret
   - Enable spot trading permissions
   - For testing, use Binance Testnet

## ⚙️ Configuration

### Initial Setup

Run the configuration wizard:
```bash
python crypto_trading_bot/main.py --mode config
```

This will guide you through:
- API key and secret setup
- Trading symbol selection
- Strategy configuration
- Risk management settings
- Notification setup

### Configuration Options

The bot supports the following configuration options:

#### Trading Parameters
- **Symbols**: List of cryptocurrency pairs to trade (e.g., BTCUSDT, ETHUSDT)
- **Investment Amount**: Amount per trade in USD
- **Max Daily Loss**: Maximum daily loss limit
- **Stop Loss**: Percentage-based stop loss
- **Take Profit**: Percentage-based take profit

#### Strategy Parameters

**Moving Average Crossover**:
- Short window (default: 10)
- Long window (default: 30)
- Minimum crossover strength

**RSI Strategy**:
- RSI period (default: 14)
- Overbought threshold (default: 70)
- Oversold threshold (default: 30)

**Bollinger Bands**:
- Window size (default: 20)
- Standard deviation multiplier (default: 2.0)
- Minimum breakout strength

## 🚀 Usage

### Starting the Bot

**CLI Mode**:
```bash
python crypto_trading_bot/main.py --mode trade
```

**GUI Mode**:
```bash
python crypto_trading_bot/main.py --mode trade --gui
```
python -m crypto_trading_bot.main --mode trade --gui 

**Backtesting Mode**:
```bash
python crypto_trading_bot/main.py --mode backtest
```

### Command Line Options

```bash
python crypto_trading_bot/main.py [OPTIONS]

Options:
  --config, -c PATH    Path to configuration file
  --mode, -m CHOICE    Bot operation mode [trade|backtest|config]
  --gui                Launch GUI interface
  --verbose, -v        Verbose logging
  --help               Show help message
```

### GUI Interface

The GUI provides:
- Real-time bot status
- Performance dashboard
- Trade history
- Configuration management
- Log viewer

### CLI Commands

Once the bot is running, you can use these commands:

```bash
# Show bot status
python crypto_trading_bot/main.py --mode config --status

# View recent trades
python crypto_trading_bot/main.py --mode config --trades

# Export trades to CSV
python crypto_trading_bot/main.py --mode config --export-trades output.csv

# Validate configuration
python crypto_trading_bot/main.py --mode config --validate
```

## 📊 Backtesting

Test your strategies on historical data:

```bash
# Run backtest with default settings
python crypto_trading_bot/main.py --mode backtest

# Run backtest with custom parameters
python crypto_trading_bot/main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

Backtest results include:
- Total trades and win rate
- Profit/Loss analysis
- Maximum drawdown
- Sharpe ratio
- Performance charts

## 🔔 Notifications

### Email Notifications
Configure SMTP settings in the configuration:
- Gmail SMTP (smtp.gmail.com:587)
- App password required for Gmail

### Telegram Notifications
1. Create a Telegram bot via @BotFather
2. Get your chat ID
3. Configure in the bot settings

## 📁 Project Structure

```
crypto-bot/
├── crypto_trading_bot/
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtest_engine.py
│   ├── config/
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── adaptive_strategy.py
│   │   ├── bot.py
│   │   ├── portfolio_manager.py
│   │   ├── risk_manager.py
│   │   ├── strategies.py
│   │   └── trade_signal.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── logs/
│   ├── main.py
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── notifier.py
│   ├── optimize_adaptive_params.py
│   ├── set_binance_api.py
│   ├── test_backtest_2years.py
│   ├── test_binance_data.py
│   ├── test_random_strategy.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── gui.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── binance_data.py
│   │   └── logger.py
├── logs/
├── README.md
├── config.json
├── requirements.txt
├── setup.py
├── test_api.py
├── test_backtest_fix.py
├── test_installation.py
```

## 🔒 Security

### API Key Security
- API keys are encrypted using Fernet encryption
- Encryption key is stored separately from configuration
- Never commit API keys to version control

### Risk Management
- Built-in stop-loss and take-profit mechanisms
- Daily loss limits
- Maximum drawdown protection
- Position size limits

## 📈 Performance Monitoring

### Key Metrics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Overall profit/loss
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure

### Logging
- Comprehensive logging to both console and file
- Rotating log files (10MB max, 5 backups)
- Different log levels for debugging

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=crypto_trading_bot tests/
```

## 🐛 Troubleshooting

### Common Issues

**API Connection Errors**:
- Verify API key and secret
- Check internet connection
- Ensure API permissions are correct
- Use testnet for initial testing

**Configuration Errors**:
- Run configuration wizard: `python crypto_trading_bot/main.py --mode config`
- Validate configuration: `python crypto_trading_bot/main.py --mode config --validate`

**Database Errors**:
- Check file permissions
- Ensure sufficient disk space
- Verify SQLite installation

### Log Files
Check log files in the `logs/` directory for detailed error information.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. The value of cryptocurrencies can go down as well as up, and you may lose some or all of your investment.**

- Past performance does not guarantee future results
- Always test thoroughly on testnet before using real funds
- Never invest more than you can afford to lose
- Consider consulting with a financial advisor

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review log files for error details
- Create an issue on GitHub

## 🔄 Updates

To update the bot:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**Happy Trading! 🚀** 