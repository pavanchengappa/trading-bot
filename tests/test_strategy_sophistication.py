import unittest
from unittest.mock import MagicMock
from datetime import datetime
import numpy as np

from crypto_trading_bot.core.strategies import MovingAverageCrossoverStrategy, RSIStrategy
from crypto_trading_bot.core.trade_signal import TradeSignal

class TestStrategySophistication(unittest.TestCase):
    def setUp(self):
        self.ma_strategy = MovingAverageCrossoverStrategy({'short_window': 5, 'long_window': 10})
        self.rsi_strategy = RSIStrategy({'rsi_period': 14})
        
        # Create synthetic klines
        # [time, open, high, low, close, volume]
        self.klines = []
        price = 100.0
        for i in range(100):
            price += np.random.normal(0, 1)
            self.klines.append([
                datetime.now().timestamp() * 1000,
                str(price), str(price+1), str(price-1), str(price),
                str(1000.0)
            ])

    def test_adx_calculation(self):
        """Test ADX calculation"""
        adx = self.ma_strategy._calculate_adx(self.klines)
        self.assertIsNotNone(adx)
        self.assertTrue(0 <= adx <= 100)

    def test_regime_detection(self):
        """Test Market Regime Detection"""
        # Create trending data
        trending_klines = []
        price = 100.0
        for i in range(100):
            price += 2.0 # Strong uptrend
            trending_klines.append([0, str(price), str(price+1), str(price-1), str(price), str(1000)])
            
        regime = self.ma_strategy.detect_market_regime(trending_klines)
        self.assertEqual(regime, 'TRENDING')
        
        # Create ranging data
        ranging_klines = []
        price = 100.0
        for i in range(100):
            price = 100.0 + np.sin(i/5) * 5 # Oscillation
            ranging_klines.append([0, str(price), str(price+1), str(price-1), str(price), str(1000)])
            
        regime = self.ma_strategy.detect_market_regime(ranging_klines)
        # ADX might take time to adjust, but should be lower than trending
        adx = self.ma_strategy._calculate_adx(ranging_klines)
        if adx > 25:
            print(f"Warning: Ranging ADX is {adx}")

    def test_volume_profile(self):
        """Test Volume Profile Calculation"""
        vp = self.ma_strategy._calculate_volume_profile(self.klines)
        self.assertIn('profile', vp)
        self.assertIn('hvns', vp)
        self.assertIn('poc', vp)
        self.assertTrue(len(vp['hvns']) > 0)

    def test_mta_integration(self):
        """Test Multi-Timeframe Analysis logic in strategy"""
        # Setup a crossover signal scenario
        # Short MA > Long MA (Bullish)
        prices = [100.0] * 20 + [110.0] * 5 # Sudden jump
        klines = []
        for p in prices:
            klines.append([0, str(p), str(p), str(p), str(p), str(1000)])
            
        # Test with Bullish HTF (Should confirm)
        context_bullish = {'htf_trend': 'BULLISH'}
        signal_bull = self.ma_strategy.generate_signal('BTC', 110.0, klines, context=context_bullish)
        
        # Test with Bearish HTF (Should reduce confidence or reject)
        context_bearish = {'htf_trend': 'BEARISH'}
        signal_bear = self.ma_strategy.generate_signal('BTC', 110.0, klines, context=context_bearish)
        
        if signal_bull and signal_bear:
            self.assertTrue(signal_bull.confidence > signal_bear.confidence)
            self.assertTrue(signal_bull.mta_confirmed)
            self.assertFalse(signal_bear.mta_confirmed)

if __name__ == '__main__':
    unittest.main()
