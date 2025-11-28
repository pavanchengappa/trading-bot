import unittest
import os
from crypto_trading_bot.database.models import DatabaseManager

class TestDBFix(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_db.sqlite"
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_recent_trades(self):
        """Test get_recent_trades method existence and execution"""
        # Should return empty list for new DB
        trades = self.db.get_recent_trades(limit=5)
        self.assertIsInstance(trades, list)
        self.assertEqual(len(trades), 0)
        print("\nTest Passed: get_recent_trades exists and returns list")

if __name__ == '__main__':
    unittest.main()
