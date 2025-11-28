# core/strategies.py - Enhanced trading strategies with increased signal generation
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import random

from crypto_trading_bot.core.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Enhanced base class for all trading strategies"""
    
    def __init__(self, parameters: Dict):
        self.parameters = parameters
        self.signal_count = 0
        self.last_signal_time = 0
    
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
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
        """Calculate MACD"""
        if len(prices) < slow:
            return None
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # For simplicity, use SMA for signal line (should be EMA of MACD)
        if len(prices) >= slow + signal:
            macd_values = []
            for i in range(slow, len(prices)):
                fast_ema = self._calculate_ema(prices[:i+1], fast)
                slow_ema = self._calculate_ema(prices[:i+1], slow)
                if fast_ema and slow_ema:
                    macd_values.append(fast_ema - slow_ema)
            
            signal_line = self._calculate_sma(macd_values, signal) if len(macd_values) >= signal else macd_line
        else:
            signal_line = macd_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }
    
    def _detect_price_patterns(self, prices: List[float]) -> Dict:
        """Detect basic price patterns"""
        if len(prices) < 5:
            return {}
        
        recent_prices = prices[-5:]
        patterns = {}
        
        # Trend detection
        if all(recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices))):
            patterns['strong_uptrend'] = True
        elif all(recent_prices[i] < recent_prices[i-1] for i in range(1, len(recent_prices))):
            patterns['strong_downtrend'] = True
        
        # Volatility detection
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, len(recent_prices))]
        avg_change = sum(price_changes) / len(price_changes)
        
        if avg_change > 0.01:  # 1% average change
            patterns['high_volatility'] = True
        
        # Support/Resistance levels
        if len(prices) >= 10:
            recent_high = max(prices[-10:])
            recent_low = min(prices[-10:])
            current_price = prices[-1]
            
            if abs(current_price - recent_high) / recent_high < 0.005:  # Within 0.5% of high
                patterns['near_resistance'] = True
            elif abs(current_price - recent_low) / recent_low < 0.005:  # Within 0.5% of low
                patterns['near_support'] = True
        
        return patterns

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Enhanced Moving Average Crossover Strategy with multiple timeframes"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.short_window = parameters.get('short_window', 7)
        self.long_window = parameters.get('long_window', 25)
        self.min_crossover_strength = parameters.get('min_crossover_strength', 0.002)  # Even lower threshold
        # Additional parameters for enhanced signals
        self.use_volume_confirmation = parameters.get('use_volume_confirmation', False)
        self.trend_strength_threshold = parameters.get('trend_strength_threshold', 0.001)
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 30)  # Reduced cooldown
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Enhanced signal generation with multiple criteria"""
        try:
            current_time = datetime.now().timestamp()
            
            # Check signal cooldown
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            closing_prices = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines] if self.use_volume_confirmation else None
            
            if len(closing_prices) < self.long_window:
                return None
            
            # Calculate multiple moving averages for better signals
            short_ma = self._calculate_sma(closing_prices, self.short_window)
            long_ma = self._calculate_sma(closing_prices, self.long_window)
            
            # Add EMA for confirmation
            short_ema = self._calculate_ema(closing_prices, self.short_window)
            long_ema = self._calculate_ema(closing_prices, self.long_window)
            
            if None in [short_ma, long_ma, short_ema, long_ema]:
                return None
            
            # Calculate previous values
            prev_short_ma = self._calculate_sma(closing_prices[:-1], self.short_window)
            prev_long_ma = self._calculate_sma(closing_prices[:-1], self.long_window)
            
            if prev_short_ma is None or prev_long_ma is None:
                return None
            
            # Detect price patterns
            patterns = self._detect_price_patterns(closing_prices)
            
            # Check for crossover
            current_diff = short_ma - long_ma
            prev_diff = prev_short_ma - prev_long_ma
            ema_diff = short_ema - long_ema
            
            # Volume confirmation
            volume_boost = 1.0
            if volumes and len(volumes) >= 5:
                current_volume = volumes[-1]
                avg_volume = sum(volumes[-5:]) / 5
                if current_volume > avg_volume * 1.2:  # 20% above average
                    volume_boost = 1.2
            
            signal = None
            confidence_base = min(abs(current_diff) / long_ma * 100, 1.0)  # Base confidence
            
            # Enhanced bullish crossover detection
            if prev_diff <= 0 and current_diff > 0:
                if abs(current_diff) / long_ma >= self.min_crossover_strength:
                    confidence = confidence_base * volume_boost
                    
                    # Pattern-based confidence boost
                    if patterns.get('strong_uptrend'):
                        confidence *= 1.3
                    elif patterns.get('high_volatility'):
                        confidence *= 1.1
                    elif patterns.get('near_support'):
                        confidence *= 1.2
                    
                    # EMA confirmation
                    if ema_diff > 0:
                        confidence *= 1.1
                    
                    signal = TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='enhanced_ma_crossover',
                        confidence=min(confidence, 1.0)
                    )
                    
                    self.last_signal_time = current_time
                    self.signal_count += 1
                    logger.info(f"Enhanced BUY Signal: {symbol} - MA Crossover (Score: {confidence:.3f})")
            
            # Enhanced bearish crossover detection  
            elif prev_diff >= 0 and current_diff < 0:
                if abs(current_diff) / long_ma >= self.min_crossover_strength:
                    confidence = confidence_base * volume_boost
                    
                    # Pattern-based confidence boost
                    if patterns.get('strong_downtrend'):
                        confidence *= 1.3
                    elif patterns.get('high_volatility'):
                        confidence *= 1.1
                    elif patterns.get('near_resistance'):
                        confidence *= 1.2
                    
                    # EMA confirmation
                    if ema_diff < 0:
                        confidence *= 1.1
                    
                    signal = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='enhanced_ma_crossover',
                        confidence=min(confidence, 1.0)
                    )
                    
                    self.last_signal_time = current_time
                    self.signal_count += 1
                    logger.info(f"Enhanced SELL Signal: {symbol} - MA Crossover (Score: {confidence:.3f})")
            
            # Additional trend following signals (not just crossovers)
            elif not signal and len(closing_prices) >= 10:
                trend_strength = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5]
                
                # Strong trend continuation signals
                if abs(trend_strength) > self.trend_strength_threshold:
                    if trend_strength > 0 and current_diff > 0 and patterns.get('high_volatility'):
                        # Strong uptrend continuation
                        signal = TradeSignal(
                            symbol=symbol,
                            action='BUY',
                            price=current_price,
                            quantity=0,
                            timestamp=datetime.now(),
                            strategy='enhanced_ma_trend_follow',
                            confidence=min(abs(trend_strength) * 20, 0.8)  # Max 0.8 confidence for trend following
                        )
                        self.last_signal_time = current_time
                        logger.info(f"Trend Follow BUY: {symbol} - Strength: {trend_strength:.4f}")
                        
                    elif trend_strength < 0 and current_diff < 0 and patterns.get('high_volatility'):
                        # Strong downtrend continuation
                        signal = TradeSignal(
                            symbol=symbol,
                            action='SELL',
                            price=current_price,
                            quantity=0,
                            timestamp=datetime.now(),
                            strategy='enhanced_ma_trend_follow',
                            confidence=min(abs(trend_strength) * 20, 0.8)
                        )
                        self.last_signal_time = current_time
                        logger.info(f"Trend Follow SELL: {symbol} - Strength: {trend_strength:.4f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in enhanced MA crossover strategy: {e}")
            return None

class RSIStrategy(BaseStrategy):
    """Enhanced RSI Strategy with dynamic thresholds"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.base_overbought = parameters.get('rsi_overbought', 75)
        self.base_oversold = parameters.get('rsi_oversold', 25)
        self.dynamic_thresholds = parameters.get('dynamic_thresholds', True)
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 20)  # Reduced cooldown
        # Additional RSI levels for more signals
        self.moderate_overbought = parameters.get('moderate_overbought', 65)
        self.moderate_oversold = parameters.get('moderate_oversold', 35)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Enhanced RSI signal generation with dynamic thresholds"""
        try:
            current_time = datetime.now().timestamp()
            
            # Check signal cooldown
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.rsi_period + 5:  # Need extra for pattern detection
                return None
            
            current_rsi = self._calculate_rsi(closing_prices, self.rsi_period)
            if current_rsi is None:
                return None
            
            # Calculate RSI trend
            prev_prices = closing_prices[:-1]
            prev_rsi = self._calculate_rsi(prev_prices, self.rsi_period)
            rsi_trend = current_rsi - prev_rsi if prev_rsi else 0
            
            # Detect price patterns for dynamic threshold adjustment
            patterns = self._detect_price_patterns(closing_prices)
            
            # Dynamic threshold adjustment
            overbought_threshold = self.base_overbought
            oversold_threshold = self.base_oversold
            
            if self.dynamic_thresholds:
                if patterns.get('high_volatility'):
                    overbought_threshold -= 5  # Lower threshold in volatile markets
                    oversold_threshold += 5
                elif patterns.get('strong_uptrend'):
                    oversold_threshold += 10  # Harder to get oversold in uptrend
                elif patterns.get('strong_downtrend'):
                    overbought_threshold -= 10  # Easier to get overbought in downtrend
            
            signal = None
            
            # Extreme oversold condition (strong buy signal)
            if current_rsi < oversold_threshold:
                confidence = (oversold_threshold - current_rsi) / oversold_threshold
                
                # RSI divergence boost
                if rsi_trend > 0 and (closing_prices[-1] - closing_prices[-3]) / closing_prices[-3] < 0:
                    confidence *= 1.3  # Positive RSI divergence
                
                signal = TradeSignal(
                    symbol=symbol,
                    action='BUY',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='enhanced_rsi_extreme',
                    confidence=min(confidence, 1.0)
                )
                self.last_signal_time = current_time
                logger.info(f"Enhanced RSI BUY: {symbol} - RSI: {current_rsi:.1f} (Extreme)")
            
            # Extreme overbought condition (strong sell signal)
            elif current_rsi > overbought_threshold:
                confidence = (current_rsi - overbought_threshold) / (100 - overbought_threshold)
                
                # RSI divergence boost
                if rsi_trend < 0 and (closing_prices[-1] - closing_prices[-3]) / closing_prices[-3] > 0:
                    confidence *= 1.3  # Negative RSI divergence
                
                signal = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='enhanced_rsi_extreme',
                    confidence=min(confidence, 1.0)
                )
                self.last_signal_time = current_time
                logger.info(f"Enhanced RSI SELL: {symbol} - RSI: {current_rsi:.1f} (Extreme)")
            
            # Moderate levels for additional trading opportunities
            elif current_rsi < self.moderate_oversold and patterns.get('high_volatility'):
                confidence = (self.moderate_oversold - current_rsi) / self.moderate_oversold * 0.7  # Lower confidence
                signal = TradeSignal(
                    symbol=symbol,
                    action='BUY',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='enhanced_rsi_moderate',
                    confidence=min(confidence, 0.7)
                )
                self.last_signal_time = current_time
                logger.info(f"Enhanced RSI BUY: {symbol} - RSI: {current_rsi:.1f} (Moderate)")
                
            elif current_rsi > self.moderate_overbought and patterns.get('high_volatility'):
                confidence = (current_rsi - self.moderate_overbought) / (100 - self.moderate_overbought) * 0.7
                signal = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='enhanced_rsi_moderate',
                    confidence=min(confidence, 0.7)
                )
                self.last_signal_time = current_time
                logger.info(f"Enhanced RSI SELL: {symbol} - RSI: {current_rsi:.1f} (Moderate)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in enhanced RSI strategy: {e}")
            return None

class BollingerBandsStrategy(BaseStrategy):
    """Enhanced Bollinger Bands Strategy with squeeze detection"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.window = parameters.get('bb_window', 20)
        self.std_dev = parameters.get('bb_std_dev', 2.0)  # Tighter bands for more signals
        self.min_breakout_strength = parameters.get('min_breakout_strength', 0.005)
        self.squeeze_threshold = parameters.get('squeeze_threshold', 0.02)  # 2% band width
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 25)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Enhanced Bollinger Bands with squeeze detection"""
        try:
            current_time = datetime.now().timestamp()
            
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.window:
                return None
            
            bb = self._calculate_bollinger_bands(closing_prices, self.window, self.std_dev)
            if bb is None:
                return None
            
            # Calculate band width for squeeze detection
            band_width = (bb['upper'] - bb['lower']) / bb['middle']
            
            # Detect squeeze condition (tight bands = upcoming volatility)
            is_squeeze = band_width < self.squeeze_threshold
            
            # Get price position within bands
            price_position = (current_price - bb['lower']) / (bb['upper'] - bb['lower'])
            
            # Detect patterns
            patterns = self._detect_price_patterns(closing_prices)
            
            signal = None
            
            # Upper band breakout (sell signal)
            if current_price > bb['upper']:
                breakout_strength = (current_price - bb['upper']) / bb['upper']
                if breakout_strength >= self.min_breakout_strength:
                    confidence = min(breakout_strength * 50, 0.9)
                    
                    # Squeeze breakout bonus
                    if is_squeeze:
                        confidence *= 1.4
                    
                    signal = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='enhanced_bb_breakout',
                        confidence=min(confidence, 1.0)
                    )
                    self.last_signal_time = current_time
                    logger.info(f"BB SELL Breakout: {symbol} - Strength: {breakout_strength:.4f}")
            
            # Lower band breakout (buy signal)
            elif current_price < bb['lower']:
                breakout_strength = (bb['lower'] - current_price) / bb['lower']
                if breakout_strength >= self.min_breakout_strength:
                    confidence = min(breakout_strength * 50, 0.9)
                    
                    # Squeeze breakout bonus
                    if is_squeeze:
                        confidence *= 1.4
                    
                    signal = TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='enhanced_bb_breakout',
                        confidence=min(confidence, 1.0)
                    )
                    self.last_signal_time = current_time
                    logger.info(f"BB BUY Breakout: {symbol} - Strength: {breakout_strength:.4f}")
            
            # Mean reversion signals when not breaking out
            elif not signal:
                # Near upper band but not breaking out (potential reversal)
                if price_position > 0.8 and patterns.get('high_volatility'):
                    signal = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='enhanced_bb_reversal',
                        confidence=min((price_position - 0.8) * 2.5, 0.6)
                    )
                    self.last_signal_time = current_time
                    logger.info(f"BB SELL Reversal: {symbol} - Position: {price_position:.2f}")
                
                # Near lower band but not breaking out (potential reversal)
                elif price_position < 0.2 and patterns.get('high_volatility'):
                    signal = TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='enhanced_bb_reversal',
                        confidence=min((0.2 - price_position) * 2.5, 0.6)
                    )
                    self.last_signal_time = current_time
                    logger.info(f"BB BUY Reversal: {symbol} - Position: {price_position:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in enhanced Bollinger Bands strategy: {e}")
            return None

class MACDStrategy(BaseStrategy):
    """MACD Strategy for additional signal generation"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.fast_period = parameters.get('fast_period', 8)   # Faster periods
        self.slow_period = parameters.get('slow_period', 17)
        self.signal_period = parameters.get('signal_period', 6)
        self.min_histogram_strength = parameters.get('min_histogram_strength', 0.0001)
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 30)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate MACD signals"""
        try:
            current_time = datetime.now().timestamp()
            
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.slow_period + self.signal_period:
                return None
            
            macd = self._calculate_macd(closing_prices, self.fast_period, self.slow_period, self.signal_period)
            if macd is None:
                return None
            
            # Calculate previous MACD for crossover detection
            prev_macd = self._calculate_macd(closing_prices[:-1], self.fast_period, self.slow_period, self.signal_period)
            if prev_macd is None:
                return None
            
            current_histogram = macd['histogram']
            prev_histogram = prev_macd['histogram']
            
            signal = None
            
            # MACD line crosses above signal line (bullish)
            if prev_histogram <= 0 and current_histogram > 0:
                if abs(current_histogram) >= self.min_histogram_strength:
                    confidence = min(abs(current_histogram) * 1000, 0.8)
                    signal = TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='macd_crossover',
                        confidence=confidence
                    )
                    self.last_signal_time = current_time
                    logger.info(f"MACD BUY: {symbol} - Histogram: {current_histogram:.6f}")
            
            # MACD line crosses below signal line (bearish)
            elif prev_histogram >= 0 and current_histogram < 0:
                if abs(current_histogram) >= self.min_histogram_strength:
                    confidence = min(abs(current_histogram) * 1000, 0.8)
                    signal = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='macd_crossover',
                        confidence=confidence
                    )
                    self.last_signal_time = current_time
                    logger.info(f"MACD SELL: {symbol} - Histogram: {current_histogram:.6f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in MACD strategy: {e}")
            return None

class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy for quick signals"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.momentum_period = parameters.get('momentum_period', 3)
        self.min_momentum_threshold = parameters.get('min_momentum_threshold', 0.005)  # 0.5%
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 15)  # Very short cooldown
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate momentum-based signals"""
        try:
            current_time = datetime.now().timestamp()
            
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.momentum_period + 1:
                return None
            
            # Calculate momentum
            current_momentum = (closing_prices[-1] - closing_prices[-self.momentum_period-1]) / closing_prices[-self.momentum_period-1]
            
            # Detect patterns for confirmation
            patterns = self._detect_price_patterns(closing_prices)
            
            signal = None
            
            # Strong positive momentum
            if current_momentum > self.min_momentum_threshold and patterns.get('high_volatility'):
                confidence = min(current_momentum * 20, 0.7)  # Max 0.7 for momentum signals
                signal = TradeSignal(
                    symbol=symbol,
                    action='BUY',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='momentum',
                    confidence=confidence
                )
                self.last_signal_time = current_time
                logger.info(f"Momentum BUY: {symbol} - Momentum: {current_momentum:.4f}")
            
            # Strong negative momentum
            elif current_momentum < -self.min_momentum_threshold and patterns.get('high_volatility'):
                confidence = min(abs(current_momentum) * 20, 0.7)
                signal = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='momentum',
                    confidence=confidence
                )
                self.last_signal_time = current_time
                logger.info(f"Momentum SELL: {symbol} - Momentum: {current_momentum:.4f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return None

class VolatilityBreakoutStrategy(BaseStrategy):
    """Strategy that trades on volatility breakouts"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.volatility_period = parameters.get('volatility_period', 10)
        self.volatility_multiplier = parameters.get('volatility_multiplier', 1.5)
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 20)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate volatility breakout signals"""
        try:
            current_time = datetime.now().timestamp()
            
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            closing_prices = [float(k[4]) for k in klines]
            
            if len(closing_prices) < self.volatility_period + 1:
                return None
            
            # Calculate recent volatility
            recent_returns = [(closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1] 
                            for i in range(-self.volatility_period, 0)]
            current_volatility = np.std(recent_returns)
            
            # Calculate historical volatility
            if len(closing_prices) >= self.volatility_period * 2:
                hist_returns = [(closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1] 
                              for i in range(-self.volatility_period*2, -self.volatility_period)]
                historical_volatility = np.std(hist_returns)
            else:
                return None
            
            # Detect volatility breakout
            volatility_ratio = current_volatility / (historical_volatility + 1e-6)  # Avoid division by zero
            
            if volatility_ratio > self.volatility_multiplier:
                # High volatility detected, determine direction
                recent_change = (closing_prices[-1] - closing_prices[-3]) / closing_prices[-3]
                
                signal = None
                confidence = min(volatility_ratio / self.volatility_multiplier * 0.6, 0.8)
                
                if recent_change > 0:
                    signal = TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='volatility_breakout',
                        confidence=confidence
                    )
                    logger.info(f"Volatility BUY: {symbol} - Ratio: {volatility_ratio:.2f}")
                else:
                    signal = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        price=current_price,
                        quantity=0,
                        timestamp=datetime.now(),
                        strategy='volatility_breakout',
                        confidence=confidence
                    )
                    logger.info(f"Volatility SELL: {symbol} - Ratio: {volatility_ratio:.2f}")
                
                self.last_signal_time = current_time
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volatility breakout strategy: {e}")
            return None

class RandomStrategy(BaseStrategy):
    """Enhanced random strategy with controlled chaos"""
    
    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.signal_probability = parameters.get('signal_probability', 0.02)  # 2% chance
        self.signal_cooldown = parameters.get('signal_cooldown_seconds', 45)
        self.favor_trend = parameters.get('favor_trend', True)
    
    def generate_signal(self, symbol: str, current_price: float, klines: List) -> Optional[TradeSignal]:
        """Generate controlled random signals"""
        try:
            current_time = datetime.now().timestamp()
            
            if current_time - self.last_signal_time < self.signal_cooldown:
                return None
            
            if random.random() < self.signal_probability:
                closing_prices = [float(k[4]) for k in klines]
                
                # Determine action
                if self.favor_trend and len(closing_prices) >= 5:
                    # Favor the current trend
                    trend = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5]
                    if trend > 0:
                        action = 'BUY' if random.random() > 0.3 else 'SELL'  # 70% chance to follow uptrend
                    else:
                        action = 'SELL' if random.random() > 0.3 else 'BUY'   # 70% chance to follow downtrend
                else:
                    action = 'BUY' if random.random() > 0.5 else 'SELL'
                
                signal = TradeSignal(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    quantity=0,
                    timestamp=datetime.now(),
                    strategy='controlled_random',
                    confidence=random.uniform(0.3, 0.6)  # Random confidence
                )
                
                self.last_signal_time = current_time
                logger.info(f"Random {action} signal generated for {symbol}")
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in random strategy: {e}")
            return None

class StrategyFactory:
    """Enhanced factory for creating trading strategies"""
    
    @staticmethod
    def create_strategy(strategy_name: str, parameters: Dict) -> BaseStrategy:
        """Create a strategy instance"""
        if strategy_name == 'moving_average_crossover':
            return MovingAverageCrossoverStrategy(parameters)
        elif strategy_name == 'rsi_strategy':
            return RSIStrategy(parameters)
        elif strategy_name == 'bollinger_bands':
            return BollingerBandsStrategy(parameters)
        elif strategy_name == 'macd_strategy':
            return MACDStrategy(parameters)
        elif strategy_name == 'momentum_strategy':
            return MomentumStrategy(parameters)
        elif strategy_name == 'volatility_breakout':
            return VolatilityBreakoutStrategy(parameters)
        elif strategy_name == 'random_strategy':
            return RandomStrategy(parameters)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    @staticmethod
    def get_default_parameters(strategy_name: str) -> Dict:
        """Get default parameters for a strategy"""
        defaults = {
            'moving_average_crossover': {
                'short_window': 3,
                'long_window': 8,
                'min_crossover_strength': 0.00005,
                'use_volume_confirmation': False,
                'signal_cooldown_seconds': 30
            },
            'rsi_strategy': {
                'rsi_period': 7,
                'rsi_overbought': 75,
                'rsi_oversold': 25,
                'moderate_overbought': 65,
                'moderate_oversold': 35,
                'dynamic_thresholds': True,
                'signal_cooldown_seconds': 20
            },
            'bollinger_bands': {
                'bb_window': 15,
                'bb_std_dev': 1.5,
                'min_breakout_strength': 0.005,
                'squeeze_threshold': 0.02,
                'signal_cooldown_seconds': 25
            },
            'macd_strategy': {
                'fast_period': 8,
                'slow_period': 17,
                'signal_period': 6,
                'min_histogram_strength': 0.0001,
                'signal_cooldown_seconds': 30
            },
            'momentum_strategy': {
                'momentum_period': 3,
                'min_momentum_threshold': 0.005,
                'signal_cooldown_seconds': 15
            },
            'volatility_breakout': {
                'volatility_period': 10,
                'volatility_multiplier': 1.5,
                'signal_cooldown_seconds': 20
            },
            'random_strategy': {
                'signal_probability': 0.02,
                'signal_cooldown_seconds': 45,
                'favor_trend': True
            }
        }
        
        return defaults.get(strategy_name, {})
    
    @staticmethod
    def get_all_strategy_names() -> List[str]:
        """Get list of all available strategy names"""
        return [
            'moving_average_crossover',
            'rsi_strategy', 
            'bollinger_bands',
            'macd_strategy',
            'momentum_strategy',
            'volatility_breakout',
            'random_strategy'
        ]