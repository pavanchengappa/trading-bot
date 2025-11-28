# core/risk_manager.py - Risk management and controls
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass

from crypto_trading_bot.config.settings import Settings
from crypto_trading_bot.core.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Position tracking data structure"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float

class RiskManager:
    """Risk management system for the trading bot"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.positions: Dict[str, Position] = {}
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 50  # Maximum trades per day
        self.last_reset_date = datetime.now().date()
        
        # Risk limits
        trading_config = settings.get_trading_config()
        self.max_daily_loss = trading_config.get('max_daily_loss', 0.05)
        self.max_drawdown = trading_config.get('max_drawdown', 0.20)
        self.stop_loss_pct = trading_config.get('stop_loss_percentage', 0.05)
        self.take_profit_pct = trading_config.get('take_profit_percentage', 0.10)
        self.max_volatility = trading_config.get('max_volatility', 10.0)  # Max 10% volatility
        self.atr_multiplier = trading_config.get('atr_stop_multiplier', 2.0)  # Chandelier Exit multiplier
        self.max_correlation = trading_config.get('max_portfolio_correlation', 0.8)  # Max correlation threshold
    
    def check_risk_limits(self, signal: TradeSignal, volatility: float = 0.0, correlation_data: Dict = None) -> bool:
        """Check if trade signal meets risk management criteria"""
        try:
            # Reset daily stats if needed
            self._reset_daily_stats()
            
            # Check daily loss limit
            if self.daily_loss >= self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {self.daily_loss}")
                return False
            
            # Check daily trade limit
            if self.daily_trades >= self.max_daily_trades:
                logger.warning(f"Daily trade limit reached: {self.daily_trades}")
                return False
            
            # Check position limits
            if not self._check_position_limits(signal):
                return False
            
            # Check drawdown limits
            if not self._check_drawdown_limits():
                return False
            
            # Check market volatility
            if not self._check_volatility_limits(signal, volatility):
                return False
            
            # Check portfolio correlation
            if correlation_data and not self._check_correlation_limits(signal, correlation_data):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk limit check: {e}")
            return False
    
    def _check_position_limits(self, signal: TradeSignal) -> bool:
        """Check position-based risk limits"""
        symbol = signal.symbol
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Don't allow same-side trades
            if position.side == signal.action:
                logger.warning(f"Already have {position.side} position in {symbol}")
                return False
            
            # Check if enough time has passed since last trade
            time_since_last_trade = datetime.now() - position.entry_time
            if time_since_last_trade < timedelta(minutes=5):  # Minimum 5 minutes between trades
                logger.warning(f"Too soon to trade {symbol} again")
                return False
        
        return True
    
    def _check_drawdown_limits(self) -> bool:
        """Check maximum drawdown limits"""
        # This is a simplified drawdown calculation
        # In a real implementation, you'd track portfolio value over time
        if self.daily_loss > 0:
            drawdown_pct = (self.daily_loss / self.settings.get_trading_config().get('investment_amount', 100.0)) * 100
            if drawdown_pct > self.max_drawdown:
                logger.warning(f"Maximum drawdown exceeded: {drawdown_pct:.2f}%")
                return False
        
        return True
    
    def _check_volatility_limits(self, signal: TradeSignal, volatility: float) -> bool:
        """Check volatility-based risk limits"""
        if volatility > self.max_volatility:
            logger.warning(f"Volatility too high for {signal.symbol}: {volatility:.2f}% > {self.max_volatility}%")
            return False
        return True
    
    def _check_correlation_limits(self, signal: TradeSignal, correlation_data: Dict) -> bool:
        """Check if new position is too correlated with existing portfolio"""
        try:
            if not correlation_data or len(self.positions) == 0:
                return True
                
            symbol = signal.symbol
            new_prices = correlation_data.get(symbol, [])
            
            if not new_prices or len(new_prices) < 10:
                return True
                
            import numpy as np
            
            # Check correlation against each active position
            for pos_symbol, pos_prices in correlation_data.items():
                if pos_symbol == symbol:
                    continue
                    
                if pos_symbol in self.positions:
                    if not pos_prices or len(pos_prices) < 10:
                        continue
                        
                    # Ensure same length
                    min_len = min(len(new_prices), len(pos_prices))
                    s1 = new_prices[-min_len:]
                    s2 = pos_prices[-min_len:]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(s1, s2)[0, 1]
                    
                    if correlation > self.max_correlation:
                        logger.warning(f"High correlation detected between {symbol} and {pos_symbol}: {correlation:.2f}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation: {e}")
            return True  # Fail open to avoid blocking trades on error
    
    def record_trade(self, trade_data: Dict):
        """Record a completed trade for risk tracking"""
        try:
            symbol = trade_data['symbol']
            side = trade_data['side']
            quantity = trade_data['quantity']
            price = trade_data['price']
            
            # Update daily statistics
            self.daily_trades += 1
            
            # Calculate P&L for this trade
            if symbol in self.positions:
                position = self.positions[symbol]
                if side != position.side:  # Closing position
                    if side == 'SELL' and position.side == 'BUY':
                        # Closing long position
                        pnl = (price - position.entry_price) * quantity
                    else:
                        # Closing short position
                        pnl = (position.entry_price - price) * quantity
                    
                    self.daily_loss += pnl
                    
                    # Remove closed position
                    del self.positions[symbol]
                    
                    logger.info(f"Closed position in {symbol}, P&L: {pnl:.2f}")
            
            # Record new position
            if side == 'BUY':
                # Use ATR-based stop loss if available (Chandelier Exit)
                if hasattr(trade_data.get('signal'), 'atr') and trade_data['signal'].atr:
                    atr = trade_data['signal'].atr
                    stop_loss = price - (atr * self.atr_multiplier)
                    logger.info(f"Using ATR-based stop loss: {stop_loss:.2f} (ATR: {atr:.4f})")
                else:
                    stop_loss = price * (1 - self.stop_loss_pct / 100)
                
                take_profit = price * (1 + self.take_profit_pct / 100)
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                logger.info(f"Opened {side} position in {symbol} @ {price:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Dict:
        """Check if stop-loss or take-profit should be triggered"""
        if symbol not in self.positions:
            return {'action': None, 'reason': None}
        
        position = self.positions[symbol]
        
        # Check stop-loss
        if current_price <= position.stop_loss:
            return {
                'action': 'SELL' if position.side == 'BUY' else 'BUY',
                'reason': 'stop_loss',
                'price': current_price
            }
        
        # Check take-profit
        if current_price >= position.take_profit:
            return {
                'action': 'SELL' if position.side == 'BUY' else 'BUY',
                'reason': 'take_profit',
                'price': current_price
            }
        
        return {'action': None, 'reason': None}
    
    def _reset_daily_stats(self):
        """Reset daily statistics if it's a new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_loss = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            logger.info("Daily risk statistics reset")
    
    def get_risk_status(self) -> Dict:
        """Get current risk management status"""
        return {
            'daily_loss': self.daily_loss,
            'daily_trades': self.daily_trades,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'active_positions': len(self.positions),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for pos in self.positions.values()
            ]
        }
    
    def update_position_stop_loss(self, symbol: str, new_stop_loss: float):
        """Update stop-loss for an existing position"""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = new_stop_loss
            logger.info(f"Updated stop-loss for {symbol}: {new_stop_loss:.2f}")
    
    def update_position_take_profit(self, symbol: str, new_take_profit: float):
        """Update take-profit for an existing position"""
        if symbol in self.positions:
            self.positions[symbol].take_profit = new_take_profit
            logger.info(f"Updated take-profit for {symbol}: {new_take_profit:.2f}")
    
    def close_position(self, symbol: str, reason: str = "manual"):
        """Manually close a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            logger.info(f"Closing position in {symbol} ({reason})")
            del self.positions[symbol]
            return True
        return False 