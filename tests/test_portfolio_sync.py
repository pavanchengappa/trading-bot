import unittest
from decimal import Decimal
from crypto_trading_bot.core.portfolio_manager import PortfolioManager

class TestPortfolioSync(unittest.TestCase):
    def setUp(self):
        self.pm = PortfolioManager(total_investment=10000.0)

    def test_reconcile_state_fixes_drift(self):
        # 1. Simulate normal allocation
        symbol = "BTCUSDT"
        amount = 100.0
        self.pm.allocate_funds(symbol, amount)
        
        # Verify initial state
        self.assertEqual(float(self.pm.allocated_funds), 100.0)
        self.assertEqual(self.pm.position_allocations[symbol], 100.0)
        
        # 2. Simulate drift (bug) - manually modify internal state
        self.pm.allocated_funds = Decimal('500.0')  # Ghost funds!
        self.assertEqual(float(self.pm.allocated_funds), 500.0)
        
        # 3. Create active positions dict (truth)
        active_positions = {
            "BTCUSDT_123456": {
                "symbol": "BTCUSDT",
                "allocated_amount": 100.0
            }
        }
        
        # 4. Reconcile
        self.pm.reconcile_state(active_positions)
        
        # 5. Verify fixed state
        self.assertEqual(float(self.pm.allocated_funds), 100.0)
        self.assertEqual(self.pm.position_allocations[symbol], 100.0)
        print("\nTest 1 Passed: Drift corrected from 500.0 to 100.0")

    def test_reconcile_empty_state(self):
        # 1. Simulate drift with no active positions
        self.pm.allocated_funds = Decimal('200.0')
        self.pm.position_allocations = {"ETHUSDT": 200.0}
        
        # 2. Reconcile with empty positions
        self.pm.reconcile_state({})
        
        # 3. Verify
        self.assertEqual(float(self.pm.allocated_funds), 0.0)
        self.assertEqual(len(self.pm.position_allocations), 0)
        print("\nTest 2 Passed: Ghost allocation cleared to 0.0")

if __name__ == '__main__':
    unittest.main()
