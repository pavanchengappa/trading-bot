import unittest
from decimal import Decimal
from crypto_trading_bot.core.portfolio_manager import PortfolioManager

class TestDecimalFix(unittest.TestCase):
    def setUp(self):
        self.pm = PortfolioManager(total_investment=1000.0)

    def test_allocate_funds_float(self):
        """Test allocate_funds with float input"""
        success = self.pm.allocate_funds("BTCUSDT", 100.50)
        self.assertTrue(success)
        self.assertIsInstance(self.pm.allocated_funds, Decimal)
        self.assertEqual(self.pm.allocated_funds, Decimal('100.50'))
        print("\nTest 1 Passed: allocate_funds handled float input")

    def test_deallocate_funds_float(self):
        """Test deallocate_funds with float input"""
        self.pm.allocate_funds("BTCUSDT", 100.50)
        success = self.pm.deallocate_funds("BTCUSDT", 50.25, pnl=10.0)
        self.assertTrue(success)
        self.assertEqual(self.pm.allocated_funds, Decimal('50.25'))
        self.assertEqual(self.pm.realized_pnl, Decimal('10.0'))
        print("\nTest 2 Passed: deallocate_funds handled float input")

if __name__ == '__main__':
    unittest.main()
