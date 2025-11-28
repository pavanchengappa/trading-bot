from crypto_trading_bot.database.models import DatabaseManager
import os

db_path = "verify_db.sqlite"
if os.path.exists(db_path):
    os.remove(db_path)

try:
    print("Initializing DatabaseManager...")
    db = DatabaseManager(db_path)
    
    print("Calling get_recent_trades...")
    trades = db.get_recent_trades(limit=5)
    
    print(f"Result type: {type(trades)}")
    print(f"Result length: {len(trades)}")
    
    if isinstance(trades, list):
        print("SUCCESS: get_recent_trades works as expected.")
    else:
        print("FAILURE: get_recent_trades did not return a list.")

except AttributeError as e:
    print(f"FAILURE: AttributeError: {e}")
except Exception as e:
    print(f"FAILURE: Exception: {e}")
finally:
    # Don't try to remove db here to avoid locking issues, just leave it
    pass
