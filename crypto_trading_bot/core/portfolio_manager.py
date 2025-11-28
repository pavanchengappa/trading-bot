# core/portfolio_manager.py - Manages total investment pool
import logging
from typing import Dict, Optional
from decimal import Decimal
from datetime import datetime

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages the total investment pool and allocates funds per trade"""
    
    def __init__(self, total_investment: float, max_allocation_per_trade: float = 0.05):
        """
        Initialize portfolio manager
        
        Args:
            total_investment: Total investment pool amount
            max_allocation_per_trade: Maximum percentage of total pool per trade (default 5%)
        """
        self.total_investment = Decimal(str(total_investment))
        self.max_allocation_per_trade = Decimal(str(max_allocation_per_trade))
        self.allocated_funds = Decimal('0')  # Currently allocated to open positions
        self.realized_pnl = Decimal('0')  # Realized profit/loss
        self.unrealized_pnl = Decimal('0')  # Unrealized profit/loss from open positions
        
        # Track individual position allocations
        self.position_allocations = {}  # {symbol: allocated_amount}
        
        logger.info(f"Portfolio initialized with ${total_investment:,.2f}, max allocation per trade: {float(self.max_allocation_per_trade)*100:.1f}%")
    
    def get_available_funds(self) -> float:
        """Get currently available funds for new investments"""
        available = self.total_investment + self.realized_pnl - self.allocated_funds
        return float(max(0, available))
    
    def get_max_trade_amount(self) -> float:
        """Get maximum amount that can be invested in a single trade"""
        current_portfolio_value = self.get_current_portfolio_value()
        max_trade = Decimal(str(current_portfolio_value)) * self.max_allocation_per_trade
        available = Decimal(str(self.get_available_funds()))
        return float(min(max_trade, available))
    
    def get_current_portfolio_value(self) -> float:
        """Get current total portfolio value including unrealized P&L"""
        total_value = self.total_investment + self.realized_pnl + self.unrealized_pnl
        return float(total_value)
    
    def can_invest(self, amount: float) -> bool:
        """Check if we can invest the specified amount"""
        return amount <= self.get_available_funds()
    
    def allocate_funds(self, symbol: str, amount: float) -> bool:
        """
        Allocate funds for a new position
        
        Args:
            symbol: Trading symbol
            amount: Amount to allocate
            
        Returns:
            True if allocation successful, False otherwise
        """
        if not self.can_invest(amount):
            logger.warning(f"Cannot allocate ${amount:.2f} - insufficient available funds (${self.get_available_funds():.2f})")
            return False
        
        # Add to existing allocation or create new
        current_allocation = self.position_allocations.get(symbol, Decimal('0'))
        amount_dec = Decimal(str(amount))
        new_allocation = current_allocation + amount_dec
        
        self.position_allocations[symbol] = new_allocation
        self.allocated_funds += amount_dec
        
        logger.info(f"Allocated ${amount:.2f} to {symbol}. Total allocated: ${float(self.allocated_funds):.2f}")
        return True
    
    def deallocate_funds(self, symbol: str, amount: float, pnl: float = 0.0) -> bool:
        """
        Deallocate funds when closing a position
        
        Args:
            symbol: Trading symbol
            amount: Original allocated amount to deallocate
            pnl: Realized profit/loss from the position
            
        Returns:
            True if deallocation successful, False otherwise
        """
        if symbol not in self.position_allocations:
            logger.warning(f"No allocation found for {symbol}")
            return False
        
        current_allocation = self.position_allocations[symbol]
        amount_dec = Decimal(str(amount))
        
        if amount_dec > current_allocation:
            logger.warning(f"Cannot deallocate ${amount:.2f} from {symbol} - only ${float(current_allocation):.2f} allocated")
            amount_dec = current_allocation
        
        # Update allocations
        new_allocation = current_allocation - amount_dec
        if new_allocation <= 0:
            del self.position_allocations[symbol]
        else:
            self.position_allocations[symbol] = new_allocation
        
        # Update funds
        self.allocated_funds -= amount_dec
        self.realized_pnl += Decimal(str(pnl))
        
        logger.info(f"Deallocated ${amount:.2f} from {symbol} with P&L: ${pnl:.2f}. Realized P&L: ${float(self.realized_pnl):.2f}")
        return True
    
    def update_unrealized_pnl(self, positions: Dict) -> None:
        """
        Update unrealized P&L based on current positions
        
        Args:
            positions: Dict of current positions with current market values
        """
        total_unrealized = Decimal('0')
        
        for symbol, position_data in positions.items():
            if symbol in self.position_allocations:
                # Calculate unrealized P&L for this position
                current_value = Decimal(str(position_data.get('current_value', 0)))
                allocated_amount = Decimal(str(self.position_allocations[symbol]))
                unrealized = current_value - allocated_amount
                total_unrealized += unrealized
        
        self.unrealized_pnl = total_unrealized
        logger.debug(f"Updated unrealized P&L: ${float(self.unrealized_pnl):.2f}")

    def reconcile_state(self, active_positions: Dict) -> None:
        """
        Reconcile internal state with actual active positions to prevent drift
        
        Args:
            active_positions: Dictionary of currently active positions from the bot
        """
        try:
            old_allocated = float(self.allocated_funds)
            
            # Reset state
            self.allocated_funds = Decimal('0')
            self.position_allocations = {}
            
            # Rebuild from active positions
            for position_key, position_data in active_positions.items():
                symbol = position_data.get('symbol', position_key.split('_')[0])
                allocated = Decimal(str(position_data.get('allocated_amount', 0)))
                
                # Update total allocated
                self.allocated_funds += allocated
                
                # Update per-symbol allocation
                current_symbol_alloc = self.position_allocations.get(symbol, Decimal('0'))
                self.position_allocations[symbol] = current_symbol_alloc + allocated
            
            new_allocated = float(self.allocated_funds)
            
            if abs(new_allocated - old_allocated) > 0.01:
                logger.warning(f"Portfolio state reconciled. Allocated funds corrected from ${old_allocated:.2f} to ${new_allocated:.2f}")
                
        except Exception as e:
            logger.error(f"Error reconciling portfolio state: {e}")
    
    def get_position_allocation(self, symbol: str) -> float:
        """Get current allocation for a specific symbol"""
        return self.position_allocations.get(symbol, 0.0)
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        available_funds = self.get_available_funds()
        current_value = self.get_current_portfolio_value()
        max_trade = self.get_max_trade_amount()
        
        return {
            'total_investment': float(self.total_investment),
            'current_portfolio_value': current_value,
            'available_funds': available_funds,
            'allocated_funds': float(self.allocated_funds),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.realized_pnl + self.unrealized_pnl),
            'max_trade_amount': max_trade,
            'allocation_percentage': float(self.allocated_funds / self.total_investment * 100) if self.total_investment > 0 else 0,
            'active_positions': len(self.position_allocations),
            'position_allocations': dict(self.position_allocations),
            'roi_percentage': float((self.realized_pnl + self.unrealized_pnl) / self.total_investment * 100) if self.total_investment > 0 else 0
        }
    
    def reset_daily_stats(self):
        """Reset any daily tracking (if needed)"""
        # This method can be extended if you want to track daily performance
        pass
    
    def adjust_total_investment(self, new_amount: float, reason: str = "Manual adjustment"):
        """
        Adjust the total investment amount (e.g., adding/withdrawing funds)
        
        Args:
            new_amount: New total investment amount
            reason: Reason for adjustment
        """
        old_amount = float(self.total_investment)
        self.total_investment = Decimal(str(new_amount))
        
        logger.info(f"Total investment adjusted from ${old_amount:.2f} to ${new_amount:.2f}. Reason: {reason}")
    
    def can_afford_position_size(self, symbol: str, price: float, desired_quantity: float) -> tuple[bool, float]:
        """
        Check if we can afford a position and return the affordable quantity
        
        Args:
            symbol: Trading symbol
            price: Current price
            desired_quantity: Desired quantity to buy
            
        Returns:
            (can_afford: bool, affordable_quantity: float)
        """
        price_dec = Decimal(str(price))
        desired_quantity_dec = Decimal(str(desired_quantity))
        required_amount = price_dec * desired_quantity_dec
        max_affordable = Decimal(str(self.get_max_trade_amount()))
        if required_amount <= max_affordable:
            return True, float(desired_quantity_dec)
        else:
            # Calculate maximum affordable quantity
            if price_dec == 0:
                affordable_quantity = Decimal('0')
            else:
                affordable_quantity = max_affordable / price_dec
            return False, float(affordable_quantity)