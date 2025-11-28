# database/models.py - Database models and management
import sqlite3
import logging
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for storing trading data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        strategy TEXT NOT NULL,
                        status TEXT NOT NULL,
                        fees REAL DEFAULT 0.0,
                        pnl REAL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create orders table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL,
                        status TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        avg_win REAL DEFAULT 0.0,
                        avg_loss REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date)
                    )
                ''')
                
                # Create positions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time DATETIME NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        status TEXT DEFAULT 'OPEN',
                        closed_at DATETIME,
                        pnl REAL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def record_trade(self, trade_data: Dict):
        """Record a completed trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (
                        order_id, symbol, side, quantity, price, timestamp, 
                        strategy, status, fees, pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['order_id'],
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['quantity'],
                    trade_data['price'],
                    trade_data['timestamp'],
                    trade_data['strategy'],
                    trade_data['status'],
                    trade_data.get('fees', 0.0),
                    trade_data.get('pnl', 0.0)
                ))
                
                conn.commit()
                logger.info(f"Trade recorded: {trade_data['symbol']} {trade_data['side']}")
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def record_order(self, order_data: Dict):
        """Record an order"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO orders (
                        order_id, symbol, side, order_type, quantity, price, 
                        status, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order_data['order_id'],
                    order_data['symbol'],
                    order_data['side'],
                    order_data['order_type'],
                    order_data['quantity'],
                    order_data.get('price'),
                    order_data['status'],
                    order_data['timestamp']
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording order: {e}")
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute('''
                        SELECT * FROM trades 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM trades 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ''', (limit,))
                
                trades = [dict(row) for row in cursor.fetchall()]
                return trades
                
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get most recent trades across all symbols"""
        return self.get_trades(limit=limit)
    
    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """Get order history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM orders WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                orders = [dict(row) for row in cursor.fetchall()]
                return orders
                
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get trades for the period
                start_date = datetime.now() - timedelta(days=days)
                cursor.execute('''
                    SELECT * FROM trades 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (start_date,))
                
                trades = [dict(row) for row in cursor.fetchall()]
                
                if not trades:
                    return {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0
                    }
                
                # Calculate statistics
                total_trades = len(trades)
                total_pnl = sum(trade['pnl'] for trade in trades)
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] < 0]
                
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
                avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss
                }
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def get_daily_performance(self, date: Optional[datetime] = None) -> Dict:
        """Get performance for a specific date"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                target_date = date.date() if date else datetime.now().date()
                
                cursor.execute('''
                    SELECT * FROM performance 
                    WHERE date = ?
                ''', (target_date,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                else:
                    return {
                        'date': target_date,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0,
                        'max_drawdown': 0.0
                    }
                
        except Exception as e:
            logger.error(f"Error getting daily performance: {e}")
            return {}
    
    def update_daily_performance(self, date: datetime, performance_data: Dict):
        """Update daily performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO performance (
                        date, total_trades, winning_trades, losing_trades,
                        total_pnl, win_rate, avg_win, avg_loss, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date.date(),
                    performance_data.get('total_trades', 0),
                    performance_data.get('winning_trades', 0),
                    performance_data.get('losing_trades', 0),
                    performance_data.get('total_pnl', 0.0),
                    performance_data.get('win_rate', 0.0),
                    performance_data.get('avg_win', 0.0),
                    performance_data.get('avg_loss', 0.0),
                    performance_data.get('max_drawdown', 0.0)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating daily performance: {e}")
    
    def export_trades_csv(self, filepath: str, symbol: Optional[str] = None):
        """Export trades to CSV file"""
        try:
            import csv
            
            trades = self.get_trades(symbol=symbol, limit=10000)
            
            with open(filepath, 'w', newline='') as csvfile:
                if trades:
                    fieldnames = trades[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(trades)
            
            logger.info(f"Trades exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
    
    def backup_database(self):
        """Create a backup of the database"""
        try:
            backup_path = f"{self.db_path}.backup"
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table counts
                cursor.execute("SELECT COUNT(*) FROM trades")
                trades_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM orders")
                orders_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'")
                open_positions = cursor.fetchone()[0]
                
                # Get database size
                db_size = Path(self.db_path).stat().st_size
                
                return {
                    'trades_count': trades_count,
                    'orders_count': orders_count,
                    'open_positions': open_positions,
                    'database_size_mb': round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}