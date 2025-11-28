import unittest
from unittest.mock import MagicMock, patch
from crypto_trading_bot.core.bot import TradingBot, MarketOpportunity
from crypto_trading_bot.core.trade_signal import TradeSignal

class TestSellLogic(unittest.TestCase):
    def setUp(self):
        # Mock settings and dependencies
        self.mock_settings = MagicMock()
        self.mock_settings.get_trading_config.return_value = {
            'initial_investment': 1000.0,
            'max_allocation_per_trade': 0.1,
            'min_trade_amount': 10.0
        }
        self.mock_settings.get_database_config.return_value = {'path': ':memory:'}
        
        # Patch dependencies to avoid real connections
        with patch('crypto_trading_bot.core.bot.Client'), \
             patch('crypto_trading_bot.core.bot.DatabaseManager'), \
             patch('crypto_trading_bot.core.bot.RiskManager'), \
             patch('crypto_trading_bot.core.bot.Notifier'):
            self.bot = TradingBot(self.mock_settings)
            
        # Mock internal components
        self.bot.client = MagicMock()
        self.bot.portfolio_manager = MagicMock()
        self.bot._close_position = MagicMock()
        self.bot._place_order = MagicMock()

    def test_phantom_sell_ignored(self):
        """Test that a SELL signal without an open position is ignored"""
        # Setup
        self.bot.positions = {}  # No positions
        signal = TradeSignal(
            symbol="BTCUSDT", 
            action="SELL", 
            price=50000.0, 
            timestamp=12345,
            quantity=0.0,
            strategy="TestStrategy",
            confidence=0.9
        )
        opportunity = MarketOpportunity(symbol="BTCUSDT", signal=signal, score=80.0, volume_24h=1000.0, volatility=0.02)
        
        # Execute
        self.bot._execute_opportunity(opportunity)
        
        # Verify
        self.bot._close_position.assert_not_called()
        self.bot._place_order.assert_not_called()
        print("\nTest 1 Passed: Phantom SELL ignored (no order placed)")

    def test_valid_sell_closes_position(self):
        """Test that a SELL signal with an open position triggers close_position"""
        # Setup
        self.bot.positions = {
            "BTCUSDT_123": {"symbol": "BTCUSDT", "quantity": 0.1}
        }
        signal = TradeSignal(
            symbol="BTCUSDT", 
            action="SELL", 
            price=50000.0, 
            timestamp=12345,
            quantity=0.1,
            strategy="TestStrategy",
            confidence=0.9
        )
        opportunity = MarketOpportunity(symbol="BTCUSDT", signal=signal, score=80.0, volume_24h=1000.0, volatility=0.02)
        
        # Execute
        self.bot._execute_opportunity(opportunity)
        
        # Verify
        self.bot._close_position.assert_called_once()
        self.bot._place_order.assert_not_called() # Should handle close via _close_position, not _place_order directly
        print("\nTest 2 Passed: Valid SELL triggered close_position")

if __name__ == '__main__':
    unittest.main()
