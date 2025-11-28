import unittest
from decimal import Decimal
from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np

from crypto_trading_bot.core.risk_manager import RiskManager, Position
from crypto_trading_bot.core.portfolio_manager import PortfolioManager
from crypto_trading_bot.core.trade_signal import TradeSignal
from crypto_trading_bot.config.settings import Settings

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        self.settings = MagicMock(spec=Settings)
        self.settings.get_trading_config.return_value = {
            'max_daily_loss': 0.05,
            'max_drawdown': 0.20,
            'stop_loss_percentage': 0.05,
            'take_profit_percentage': 0.10,
            'max_volatility': 10.0,
            'atr_stop_multiplier': 2.0,
            'max_portfolio_correlation': 0.8
        }
        self.risk_manager = RiskManager(self.settings)
        self.portfolio_manager = PortfolioManager(total_investment=10000.0)
        self.portfolio_manager.use_kelly_criterion = True

    def test_atr_stop_loss(self):
        """Test that ATR-based stop loss is calculated correctly"""
        signal = TradeSignal(
            symbol='BTCUSDT',
            action='BUY',
            price=50000.0,
            quantity=0.1,
            timestamp=datetime.now(),
            strategy='test',
            confidence=0.8,
            atr=1000.0
        )
        
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 50000.0,
            'signal': signal
        }
        
        self.risk_manager.record_trade(trade_data)
        
        position = self.risk_manager.positions['BTCUSDT']
        expected_stop = 50000.0 - (1000.0 * 2.0)  # Price - (ATR * Multiplier)
        self.assertEqual(position.stop_loss, expected_stop)

    def test_kelly_criterion(self):
        """Test Kelly Criterion position sizing"""
        # Test case 1: Positive expectancy
        # Win rate 50%, Win/Loss 2.0
        # Kelly = (2.0 * 0.5 - 0.5) / 2.0 = 0.5 / 2.0 = 0.25 (25%)
        # Half-Kelly = 12.5%
        # But capped at max_allocation_per_trade (5%)
        
        allocation = self.portfolio_manager.calculate_kelly_allocation(win_rate=0.5, win_loss_ratio=2.0)
        self.assertEqual(allocation, 0.05)
        
        # Test case 2: Negative expectancy
        # Win rate 30%, Win/Loss 1.0
        # Kelly = (1.0 * 0.3 - 0.7) / 1.0 = -0.4
        allocation = self.portfolio_manager.calculate_kelly_allocation(win_rate=0.3, win_loss_ratio=1.0)
        self.assertEqual(allocation, 0.0)

    def test_correlation_guard(self):
        """Test Portfolio Correlation Guard"""
        # Create a dummy position
        self.risk_manager.positions['BTCUSDT'] = Position(
            symbol='BTCUSDT', side='BUY', quantity=1, entry_price=50000,
            entry_time=datetime.now(), stop_loss=49000, take_profit=55000
        )
        
        # Correlated prices (ETH following BTC)
        btc_prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        eth_prices = [10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9]
        
        correlation_data = {
            'BTCUSDT': btc_prices,
            'ETHUSDT': eth_prices
        }
        
        signal = TradeSignal(
            symbol='ETHUSDT', action='BUY', price=2000, quantity=1,
            timestamp=datetime.now(), strategy='test', confidence=0.8
        )
        
        # Should fail due to high correlation
        result = self.risk_manager.check_risk_limits(signal, correlation_data=correlation_data)
        self.assertFalse(result)
        
        # Uncorrelated prices
        random_prices = [10, 9, 11, 8, 12, 7, 13, 6, 14, 5]
        correlation_data['SOLUSDT'] = random_prices
        
        signal_sol = TradeSignal(
            symbol='SOLUSDT', action='BUY', price=100, quantity=1,
            timestamp=datetime.now(), strategy='test', confidence=0.8
        )
        
        # Should pass (assuming correlation < 0.8)
        result = self.risk_manager.check_risk_limits(signal_sol, correlation_data=correlation_data)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
