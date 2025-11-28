from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    timestamp: datetime
    strategy: str
    confidence: float
    atr: Optional[float] = None
    regime: str = "UNKNOWN"
    mta_confirmed: bool = False