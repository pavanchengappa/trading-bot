# utils/binance_data.py - Binance API data fetching utilities
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    """Utility class for fetching historical data from Binance"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """Initialize Binance client"""
        try:
            if testnet:
                self.client = Client(api_key, api_secret, testnet=True)
            else:
                self.client = Client(api_key, api_secret)
            logger.info("Binance client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            raise
    
    def get_historical_klines(self, 
                            symbol: str, 
                            start_date: datetime, 
                            end_date: datetime,
                            interval: str = Client.KLINE_INTERVAL_1HOUR) -> pd.DataFrame:
        """
        Fetch historical klines data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start date for data
            end_date: End date for data
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Convert dates to milliseconds timestamp
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            # Fetch klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ts,
                end_str=end_ts
            )
            
            if not klines:
                logger.warning(f"No data found for {symbol} in the specified date range")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df.set_index('Open time', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING']
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid and trading"""
        try:
            symbols = self.get_symbols()
            return symbol in symbols
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_recent_klines(self, symbol: str, limit: int = 100, interval: str = Client.KLINE_INTERVAL_1HOUR) -> pd.DataFrame:
        """
        Get recent klines data (useful for testing)
        
        Args:
            symbol: Trading pair
            limit: Number of klines to fetch (max 1000)
            interval: Kline interval
        
        Returns:
            DataFrame with recent klines
        """
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            if not klines:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Convert string values to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            logger.error(f"Error fetching recent klines: {e}")
            return pd.DataFrame()

def fetch_historical_data_for_backtest(symbols: List[str], 
                                     start_date: datetime, 
                                     end_date: datetime,
                                     interval: str = Client.KLINE_INTERVAL_1HOUR,
                                     api_key: str = None,
                                     api_secret: str = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch historical data for multiple symbols
    
    Args:
        symbols: List of trading symbols
        start_date: Start date
        end_date: End date
        interval: Kline interval
        api_key: Binance API key (optional for public data)
        api_secret: Binance API secret (optional for public data)
    
    Returns:
        Dictionary mapping symbols to their historical data DataFrames
    """
    fetcher = BinanceDataFetcher(api_key, api_secret)
    historical_data = {}
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        data = fetcher.get_historical_klines(symbol, start_date, end_date, interval)
        
        if not data.empty:
            historical_data[symbol] = data
            logger.info(f"Successfully loaded {len(data)} data points for {symbol}")
        else:
            logger.warning(f"No data available for {symbol}")
    
    return historical_data 