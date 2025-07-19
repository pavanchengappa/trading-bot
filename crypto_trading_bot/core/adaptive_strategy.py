import numpy as np
from typing import Dict, List
from enum import Enum

class MarketCondition(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"

class EnhancedTradingStrategy:
    """
    Enhanced multi-strategy system that adapts to market conditions
    """
    def __init__(self):
        self.strategy_configs = {
            MarketCondition.TRENDING_UP: {
                "name": "moving_average_crossover",
                "parameters": {
                    "short_window": 8,
                    "long_window": 21,
                    "min_crossover_strength": 0.001
                },
                "weight": 0.6
            },
            MarketCondition.TRENDING_DOWN: {
                "name": "rsi_strategy",
                "parameters": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "confirmation_periods": 1
                },
                "weight": 0.7
            },
            MarketCondition.SIDEWAYS: {
                "name": "bollinger_bands",
                "parameters": {
                    "bb_window": 14,
                    "bb_std_dev": 1.8,
                    "min_breakout_strength": 0.005
                },
                "weight": 0.8
            },
            MarketCondition.HIGH_VOLATILITY: {
                "name": "bollinger_bands",
                "parameters": {
                    "bb_window": 10,
                    "bb_std_dev": 2.2,
                    "min_breakout_strength": 0.008
                },
                "weight": 0.5
            }
        }

    def detect_market_condition(self, prices: List[float], lookback: int = 20) -> MarketCondition:
        if len(prices) < lookback:
            return MarketCondition.SIDEWAYS
        recent_prices = prices[-lookback:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        returns = [recent_prices[i] / recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
        volatility = np.std(returns)
        if volatility > 0.03:
            return MarketCondition.HIGH_VOLATILITY
        elif price_change > 0.05:
            return MarketCondition.TRENDING_UP
        elif price_change < -0.05:
            return MarketCondition.TRENDING_DOWN
        else:
            return MarketCondition.SIDEWAYS

    def get_optimal_strategy_config(self, prices: List[float]) -> Dict:
        market_condition = self.detect_market_condition(prices)
        return self.strategy_configs[market_condition]

    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 0.02) -> float:
        return account_balance * risk_per_trade

    def apply_filters(self, signal: str, prices: List[float]) -> bool:
        if len(prices) < 50:
            return True
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-30:])
        if signal == "BUY" and short_ma < long_ma:
            return False
        elif signal == "SELL" and short_ma > long_ma:
            return False
        return True

# Example usage and backtesting improvements
class OptimizedBacktester:
    """
    Enhanced backtester with better risk management and adaptive strategy selection
    """
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.strategy = EnhancedTradingStrategy()

    def backtest_with_dynamic_strategy(self, price_data: List[float]) -> Dict:
        results = {
            'trades': [],
            'equity_curve': [],
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'total_return': 0
        }
        capital = self.initial_capital
        position = 0
        peak_capital = capital
        for i in range(50, len(price_data)):
            current_prices = price_data[:i+1]
            strategy_config = self.strategy.get_optimal_strategy_config(current_prices)
            signal = self.simulate_signal_generation(current_prices, strategy_config)
            if signal and self.strategy.apply_filters(signal, current_prices):
                if signal == "BUY" and position == 0:
                    position_size = self.strategy.calculate_position_size(capital)
                    shares = position_size / price_data[i]
                    position = shares
                    capital -= position_size
                elif signal == "SELL" and position > 0:
                    capital += position * price_data[i]
                    position = 0
            current_equity = capital + (position * price_data[i] if position > 0 else 0)
            results['equity_curve'].append(current_equity)
            if current_equity > peak_capital:
                peak_capital = current_equity
            else:
                drawdown = (peak_capital - current_equity) / peak_capital
                results['max_drawdown'] = max(results['max_drawdown'], drawdown)
        final_equity = results['equity_curve'][-1] if results['equity_curve'] else self.initial_capital
        results['total_return'] = (final_equity - self.initial_capital) / self.initial_capital
        return results

    def simulate_signal_generation(self, prices: List[float], strategy_config: Dict) -> str:
        if len(prices) < 20:
            return None
        if strategy_config["name"] == "bollinger_bands":
            recent_prices = prices[-20:]
            ma = np.mean(recent_prices)
            std = np.std(recent_prices)
            current_price = prices[-1]
            if current_price < ma - 2 * std:
                return "BUY"
            elif current_price > ma + 2 * std:
                return "SELL"
        return None

# Recommended strategy configurations for maximum profit
PROFIT_OPTIMIZED_CONFIGS = {
    "conservative": {
        "name": "bollinger_bands",
        "parameters": {
            "bb_window": 16,
            "bb_std_dev": 1.9,
            "min_breakout_strength": 0.004
        }
    },
    "aggressive": {
        "name": "rsi_strategy", 
        "parameters": {
            "rsi_period": 10,
            "rsi_overbought": 78,
            "rsi_oversold": 22,
            "confirmation_periods": 0
        }
    },
    "balanced": {
        "name": "moving_average_crossover",
        "parameters": {
            "short_window": 7,
            "long_window": 18,
            "min_crossover_strength": 0.0015
        }
    }
} 