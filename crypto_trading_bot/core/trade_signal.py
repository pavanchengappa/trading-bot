from dataclasses import dataclass
from datetime import datetime

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