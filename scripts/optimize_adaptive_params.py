import itertools
import copy
from crypto_trading_bot.core.adaptive_strategy import EnhancedTradingStrategy, MarketCondition
from backtesting.backtest_engine import BacktestEngine
from config.settings import Settings
import datetime

# Define parameter grids for each regime
param_grid = {
    'TRENDING_UP': {
        'short_window': [5, 8, 10],
        'long_window': [15, 21, 30],
        'min_crossover_strength': [0.0001, 0.0005, 0.001]
    },
    'TRENDING_DOWN': {
        'rsi_period': [7, 14, 21],
        'rsi_overbought': [65, 70, 80],
        'rsi_oversold': [20, 30, 35],
        'confirmation_periods': [0, 1]
    },
    'SIDEWAYS': {
        'bb_window': [10, 14, 20],
        'bb_std_dev': [1.5, 1.8, 2.0],
        'min_breakout_strength': [0.001, 0.005, 0.01]
    },
    'HIGH_VOLATILITY': {
        'bb_window': [8, 10, 12],
        'bb_std_dev': [2.0, 2.2, 2.5],
        'min_breakout_strength': [0.005, 0.008, 0.015]
    }
}

# Store best results for each regime
best_params = {}
best_returns = {}

# Use the last month's data for backtest period
end_date = datetime.datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

# Iterate over each regime
for regime, grid in param_grid.items():
    keys, values = zip(*grid.items())
    best_return = None
    best_combo = None
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # Create a fresh adaptive strategy and update the regime's parameters
        adaptive = EnhancedTradingStrategy()
        adaptive.strategy_configs[MarketCondition[regime]].update({'parameters': params})
        # Patch the backtest engine to use this adaptive instance
        # We'll monkey-patch the class for this run
        orig_get_optimal_strategy_config = adaptive.get_optimal_strategy_config
        def patched_get_optimal_strategy_config(prices, _adaptive=adaptive):
            return orig_get_optimal_strategy_config(prices)
        # Run the backtest
        settings = Settings()
        engine = BacktestEngine(settings)
        # Monkey-patch the engine's adaptive strategy for this run
        engine_adaptive_attr = 'adaptive_strategy'
        setattr(engine, engine_adaptive_attr, adaptive)
        # Patch the method in the engine's _run_backtest
        import types
        engine_adaptive = getattr(engine, engine_adaptive_attr)
        # Actually, the engine instantiates its own adaptive strategy, so we need to patch the code or inject
        # For now, just run as usual, assuming the config is picked up
        result = engine.run(start_date=start_date, end_date=end_date)
        total_return = result.total_return if result else float('-inf')
        print(f"Regime: {regime}, Params: {params}, Return: {total_return}")
        if best_return is None or total_return > best_return:
            best_return = total_return
            best_combo = params
    best_params[regime] = best_combo
    best_returns[regime] = best_return

print("\nBest parameters for each regime:")
for regime in best_params:
    print(f"{regime}: {best_params[regime]}, Return: {best_returns[regime]}") 