# core/strategies.py - Trading strategies implementation
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

from crypto_trading_bot.core.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, parameters: Dict):
        self.parameters = parameters
    
    @abstractmethod
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate trading signal based on strategy logic"""
        pass
    
    def _calculate_sma(self, prices: List[float], window: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window
    
    def _calculate_ema(self, prices: List[float], window: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < window:
            return None
        
        prices_array = np.array(prices)
        alpha = 2 / (window + 1)
        ema = [prices_array[0]]
        
        for price in prices_array[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        
        return float(ema[-1])
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: List[float], window: int = 20, std_dev: float = 2) -> Optional[Dict]:
        """Calculate Bollinger Bands"""
        if len(prices) < window:
            return None
        
        prices_array = np.array(prices[-window:])
        sma = np.mean(prices_array)
        std = np.std(prices_array)
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy with RSI filter"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.short_window = parameters.get('short_window', 5)   # Shorter window
        self.long_window = parameters.get('long_window', 15)    # Shorter window
        self.min_crossover_strength = parameters.get('min_crossover_strength', 0.0001)  # Much lower threshold
        # RSI filter parameters
        self.rsi_period = parameters.get('rsi_period', 14)
        self.rsi_overbought = parameters.get('rsi_overbought', 70)
        self.rsi_oversold = parameters.get('rsi_oversold', 30)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate signal based on MA crossover with RSI filter"""
        try:
            logger.debug(f"[MA Crossover] generate_signal called for {symbol} at price {current_price} with {len(klines)} klines")
            # Extract closing prices
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < max(self.long_window, self.rsi_period + 1):
                logger.debug(f"[MA Crossover] Not enough closing prices: {len(closing_prices)} < {max(self.long_window, self.rsi_period + 1)}")
                return None
            
            # Calculate moving averages
            short_ma = self._calculate_sma(closing_prices, self.short_window)
            long_ma = self._calculate_sma(closing_prices, self.long_window)
            
            if short_ma is None or long_ma is None:
                logger.debug(f"[MA Crossover] short_ma or long_ma is None: short_ma={short_ma}, long_ma={long_ma}")
                return None
            
            # Calculate previous moving averages
            prev_short_ma = self._calculate_sma(closing_prices[:-1], self.short_window)
            prev_long_ma = self._calculate_sma(closing_prices[:-1], self.long_window)
            
            if prev_short_ma is None or prev_long_ma is None:
                logger.debug(f"[MA Crossover] prev_short_ma or prev_long_ma is None: prev_short_ma={prev_short_ma}, prev_long_ma={prev_long_ma}")
                return None
            
            # Calculate RSI
            current_rsi = self._calculate_rsi(closing_prices, self.rsi_period)
            if current_rsi is None:
                logger.debug(f"[MA Crossover] current_rsi is None")
                return None
            
            # Check for crossover
            current_diff = short_ma - long_ma
            prev_diff = prev_short_ma - prev_long_ma
            
            # Debug logging
            if len(closing_prices) % 100 == 0:  # Log every 100th check
                logger.debug(f"MA Check - Short: {short_ma:.2f}, Long: {long_ma:.2f}, Current Diff: {current_diff:.6f}, Prev Diff: {prev_diff:.6f}, RSI: {current_rsi:.2f}")
            
            # Bullish crossover (short MA crosses above long MA)
            if prev_diff <= 0 and current_diff > 0:
                if abs(current_diff) / long_ma >= self.min_crossover_strength:
                    # RSI filter: only buy if not overbought
                    if current_rsi < self.rsi_overbought:
                        logger.info(f"BUY Signal: Short MA ({short_ma:.2f}) crossed above Long MA ({long_ma:.2f}), RSI={current_rsi:.2f}")
                        logger.debug(f"[MA Crossover] Returning BUY signal for {symbol} at {current_price}")
                        return TradeSignal(
                            symbol=symbol,
                            action='BUY',
                            price=current_price,
                            quantity=0,  # Will be calculated by bot
                            timestamp=datetime.now(),
                            strategy='moving_average_crossover_rsi',
                            confidence=min(abs(current_diff) / long_ma, 1.0)
                        )
            
            # Bearish crossover (short MA crosses below long MA)
            elif prev_diff >= 0 and current_diff < 0:
                if abs(current_diff) / long_ma >= self.min_crossover_strength:
                    # RSI filter: only sell if not oversold
                    if current_rsi > self.rsi_oversold:
                        logger.info(f"SELL Signal: Short MA ({short_ma:.2f}) crossed below Long MA ({long_ma:.2f}), RSI={current_rsi:.2f}")
                        logger.debug(f"[MA Crossover] Returning SELL signal for {symbol} at {current_price}")
                        return TradeSignal(
                            symbol=symbol,
                            action='SELL',
                            price=current_price,
                            quantity=0,  # Will be calculated by bot
                            timestamp=datetime.now(),
                            strategy='moving_average_crossover_rsi',
                            confidence=min(abs(current_diff) / long_ma, 1.0)
                        )
            
            logger.debug(f"[MA Crossover] No signal generated for {symbol} at {current_price}, RSI={current_rsi:.2f}")
            return None
            
        except Exception as e:
            logger.error(f"Error in MA crossover strategy with RSI filter: {e}")
            return None

class RSIStrategy(BaseStrategy):
    """RSI-based Strategy"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.rsi_period = parameters.get('rsi_period', 7)        # Shorter period
        self.overbought_threshold = parameters.get('rsi_overbought', 80)  # More conservative
        self.oversold_threshold = parameters.get('rsi_oversold', 20)      # More conservative
        self.confirmation_periods = parameters.get('confirmation_periods', 0)  # No confirmation needed
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate signal based on RSI levels"""
        try:
            logger.debug(f"[RSI] generate_signal called for {symbol} at price {current_price} with {len(klines)} klines")
            # Extract closing prices
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.rsi_period + 1:
                logger.debug(f"[RSI] Not enough closing prices: {len(closing_prices)} < {self.rsi_period + 1}")
                return None
            
            # Calculate current RSI
            current_rsi = self._calculate_rsi(closing_prices, self.rsi_period)
            
            if current_rsi is None:
                logger.debug(f"[RSI] current_rsi is None")
                return None
            
            # Check for oversold condition (buy signal)
            if current_rsi < self.oversold_threshold:
                confidence = (self.oversold_threshold - current_rsi) / self.oversold_threshold
                logger.debug(f"[RSI] Returning BUY signal for {symbol} at {current_price}, RSI={current_rsi}")
                return TradeSignal(
                    symbol=symbol,
                    action='BUY',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='rsi_strategy',
                    confidence=min(confidence, 1.0)
                )
            
            # Check for overbought condition (sell signal)
            elif current_rsi > self.overbought_threshold:
                confidence = (current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
                logger.debug(f"[RSI] Returning SELL signal for {symbol} at {current_price}, RSI={current_rsi}")
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='rsi_strategy',
                    confidence=min(confidence, 1.0)
                )
            
            logger.debug(f"[RSI] No signal generated for {symbol} at {current_price}, RSI={current_rsi}")
            return None
            
        except Exception as e:
            logger.error(f"Error in RSI strategy: {e}")
            return None

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.window = parameters.get('bb_window', 20)
        self.std_dev = parameters.get('bb_std_dev', 2.0)
        self.min_breakout_strength = parameters.get('min_breakout_strength', 0.01)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate signal based on Bollinger Bands"""
        try:
            # Extract closing prices
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.window:
                return None
            
            # Calculate Bollinger Bands
            bb = self._calculate_bollinger_bands(closing_prices, self.window, self.std_dev)
            
            if bb is None:
                return None
            
            # Check for breakout signals
            upper_breakout = current_price > bb['upper']
            lower_breakout = current_price < bb['lower']
            
            # Calculate breakout strength
            if upper_breakout:
                breakout_strength = (current_price - bb['upper']) / bb['upper']
                if breakout_strength >= self.min_breakout_strength:
                    return TradeSignal(
                        symbol=symbol,
                        action='SELL',  # Price above upper band suggests overbought
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='bollinger_bands',
                        confidence=min(breakout_strength, 1.0)
                    )
            
            elif lower_breakout:
                breakout_strength = (bb['lower'] - current_price) / bb['lower']
                if breakout_strength >= self.min_breakout_strength:
                    return TradeSignal(
                        symbol=symbol,
                        action='BUY',  # Price below lower band suggests oversold
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='bollinger_bands',
                        confidence=min(breakout_strength, 1.0)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands strategy: {e}")
            return None

class RandomStrategy(BaseStrategy):
    """Random Strategy for testing - generates signals randomly"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.signal_probability = parameters.get('signal_probability', 0.01)  # 1% chance per check
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate random signals for testing"""
        try:
            # Generate random signal with given probability
            if np.random.random() < self.signal_probability:
                action = 'BUY' if np.random.random() > 0.5 else 'SELL'
                logger.info(f"Random {action} signal generated for {symbol} @ {current_price}")
                return TradeSignal(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='moving_average_crossover',
                    confidence=0.5
                )
            return None
            
        except Exception as e:
            logger.error(f"Error in random strategy: {e}")
            return None

class StrategyFactory:
    """Factory for creating trading strategies"""
    
    @staticmethod
    def create_strategy(strategy_name: str, parameters: Dict) -> BaseStrategy:
        """Create a strategy instance"""
        if strategy_name == 'moving_average_crossover':
            return MovingAverageCrossoverStrategy(parameters)
        elif strategy_name == 'rsi_strategy':
            return RSIStrategy(parameters)
        elif strategy_name == 'bollinger_bands':
            return BollingerBandsStrategy(parameters)
        elif strategy_name == 'random_strategy':
            return RandomStrategy(parameters)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    @staticmethod
    def get_default_parameters(strategy_name: str) -> Dict:
        """Get default parameters for a strategy"""
        if strategy_name == 'moving_average_crossover':
            return {
                'short_window': 5,
                'long_window': 15,
                'min_crossover_strength': 0.0001
            }
        elif strategy_name == 'rsi_strategy':
            return {
                'rsi_period': 7,
                'rsi_overbought': 80,
                'rsi_oversold': 20,
                'confirmation_periods': 0
            }
        elif strategy_name == 'bollinger_bands':
            return {
                'period': 20,
                'std_dev': 2,
                'min_breakout_strength': 0.001
            }
        elif strategy_name == 'random_strategy':
            return {
                'signal_probability': 0.01  # 1% chance per check
            }
        else:
            return {} 