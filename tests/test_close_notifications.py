import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from crypto_trading_bot.core.bot import TradingBot

class TestCloseNotifications(unittest.TestCase):
    def setUp(self):
        self.mock_settings = MagicMock()
        self.mock_settings.get_trading_config.return_value = {}
        self.mock_settings.get_database_config.return_value = {'path': ':memory:'}
        
        with patch('crypto_trading_bot.core.bot.Client'), \
             patch('crypto_trading_bot.core.bot.DatabaseManager'), \
             patch('crypto_trading_bot.core.bot.RiskManager'), \
             patch('crypto_trading_bot.core.bot.Notifier'):
            self.bot = TradingBot(self.mock_settings)
            
        self.bot.client = MagicMock()
        self.bot.client.get_symbol_ticker.return_value = {'price': '55000.0'}
        self.bot.portfolio_manager = MagicMock()
        self.bot.portfolio_manager.get_portfolio_summary.return_value = {'current_portfolio_value': 10000.0}
        self.bot.db_manager = MagicMock()
        self.bot.notifier = MagicMock()

    def test_close_position_sends_notification_and_records_db(self):
        # Setup dummy position
        position_key = "BTCUSDT_123"
        self.bot.positions = {
            position_key: {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.1,
                "entry_price": 50000.0,
                "allocated_amount": 5000.0
            }
        }
        
        # Execute close
        self.bot._close_position(position_key, self.bot.positions[position_key], "Test Reason")
        
        # Verify DB recording
        self.bot.db_manager.record_trade.assert_called_once()
        call_args = self.bot.db_manager.record_trade.call_args[0][0]
        self.assertEqual(call_args['symbol'], "BTCUSDT")
        self.assertEqual(call_args['side'], "SELL")
        self.assertEqual(call_args['pnl'], 500.0)  # (55000 - 50000) * 0.1
        
        # Verify Notification
        self.bot.notifier.send_notification.assert_called_once()
        notif_args = self.bot.notifier.send_notification.call_args[0]
        self.assertIn("Position Closed", notif_args[0])
        self.assertIn("P&L: $500.00", notif_args[1])
        
        print("\nTest Passed: Notification sent and Trade recorded.")

if __name__ == '__main__':
    unittest.main()
